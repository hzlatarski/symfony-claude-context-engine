"""EmbeddingGemma 2 text embedder (ONNX, 4-bit) for the articles collection.

Why this model: on 400 English questions over the live knowledge base, hybrid
search found the right article in the top 5 for 88.8% of questions with this
embedder versus 81.0% with Chroma's default all-MiniLM-L6-v2 (48 wins, 17
losses, p<0.001). See knowledge/research/embeddinggemma-2-vs-minilm.md.

Why ONNX q4 and not sentence-transformers: the UserPromptSubmit hook searches
from a fresh process on every prompt. PyTorch needs ~15 s to load the model;
this ONNX file loads in ~0.3 s, embeds a query in ~40 ms on CPU, and matches
the fp32 model's retrieval quality on the same benchmark. onnxruntime,
tokenizers and huggingface_hub already ship with chromadb, so this adds no
dependency.

The model is trained with task prefixes, so queries and documents are embedded
differently: queries as ``task: search result | query: ...`` and documents as
``title: ... | text: ...``. Mixing the two, or comparing these vectors with
MiniLM vectors, silently degrades ranking — so every collection is opened with
exactly one embedding function (see vector_store.articles_collection).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

# Imported eagerly for the same reason vector_store imports chromadb eagerly:
# a first import from a FastMCP worker thread can deadlock on the import lock.
import onnxruntime as ort
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
from filelock import FileLock
from tokenizers import Tokenizer

MODEL_REPO = "onnx-community/embeddinggemma-2-ONNX"
# Pinned so a re-upload upstream can never change vectors under a built index.
MODEL_REVISION = "daa72c51243991dfcaf9f9137d2c573d8f7790c0"
MODEL_FILE = "onnx/model_q4.onnx"
MODEL_FILES = ("tokenizer.json", MODEL_FILE, f"{MODEL_FILE}_data")

DIMENSIONS = 768
# Articles are embedded per Truth zone; p90 article is ~1,700 words, so 2,048
# tokens covers nearly every zone while bounding CPU time on outliers.
MAX_TOKENS = 2048
BATCH_SIZE = 8
QUERY_PREFIX = "task: search result | query: "

_FEATURE_WIDTH = 512  # The text-only path still requires empty modality inputs.


class EmbedderUnavailable(RuntimeError):
    """The model files are missing and could not be downloaded."""


def model_dir() -> Path:
    override = os.environ.get("MC_EMBEDDER_CACHE")
    base = Path(override) if override else Path.home() / ".cache" / "memory-compiler"
    return base / "embeddinggemma-2-onnx" / MODEL_REVISION


def _model_present(directory: Path) -> bool:
    return all((directory / name).is_file() for name in MODEL_FILES)


def ensure_model(download: bool = True) -> Path:
    """Return the model directory, downloading the pinned files once.

    ``download=False`` (the search path) never touches the network: a search
    runs inside the per-prompt hook's time budget, so a missing model must
    fail fast and let BM25 answer, not wait on a lock or a 170 MB download.

    ``local_dir`` is required: files resolved through the Hugging Face symlink
    cache fail onnxruntime's external-data path check ("path escapes model
    directory").
    """
    directory = model_dir()
    if _model_present(directory):
        return directory
    if not download:
        raise EmbedderUnavailable(
            f"EmbeddingGemma 2 model files are missing from {directory}. Fix: run "
            "`uv run python scripts/embedder.py --download` (a reindex also fetches "
            "them), or set MC_EMBEDDER=minilm to use the old collection."
        )
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory.parent / f".{MODEL_REVISION}.download.lock")):
        if _model_present(directory):
            return directory
        try:
            from huggingface_hub import hf_hub_download

            for name in MODEL_FILES:
                hf_hub_download(
                    MODEL_REPO, name, revision=MODEL_REVISION, local_dir=str(directory)
                )
        except Exception as exc:
            raise EmbedderUnavailable(
                f"EmbeddingGemma 2 model files are missing from {directory} and the "
                f"download failed ({exc.__class__.__name__}: {exc}). Fix: connect to "
                "the internet and run `uv run python scripts/embedder.py --download`, "
                "or set MC_EMBEDDER=minilm to use the old collection."
            ) from exc
    return directory


def document_text(title: str | None, text: str) -> str:
    title = (title or "").strip() or "none"
    return f"title: {title} | text: {text}"


def query_text(query: str) -> str:
    return f"{QUERY_PREFIX}{query}"


class GemmaOnnx:
    """Thread-safe wrapper around one ONNX session."""

    def __init__(self, directory: Path) -> None:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(directory / MODEL_FILE), options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self._tokenizer.enable_truncation(MAX_TOKENS)
        self._tokenizer.no_padding()
        self._empty = np.zeros((0, _FEATURE_WIDTH), dtype=np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed raw (already prefixed) texts; rows are L2-normalized."""
        if not texts:
            return np.zeros((0, DIMENSIONS), dtype=np.float32)
        out = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
        # Longest first so each batch pads to similar lengths.
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            encodings = self._tokenizer.encode_batch([texts[i] for i in idx])
            width = max(len(e.ids) for e in encodings)
            ids = np.zeros((len(idx), width), dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            for row, encoding in enumerate(encodings):
                ids[row, : len(encoding.ids)] = encoding.ids
                mask[row, : len(encoding.ids)] = 1
            vectors = self._session.run(
                ["sentence_embedding"],
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "image_features": self._empty,
                    "video_features": self._empty,
                    "audio_features": self._empty,
                },
            )[0]
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            out[idx] = vectors / np.maximum(norms, 1e-12)
        return out

    def embed_documents(self, titles: list[str | None], texts: list[str]) -> np.ndarray:
        return self.encode([document_text(t, x) for t, x in zip(titles, texts)])

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self.encode([query_text(q) for q in queries])


_model: GemmaOnnx | None = None
_model_lock = threading.Lock()


def get_model(download: bool = False) -> GemmaOnnx:
    """The shared model. Only write paths pass ``download=True``."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = GemmaOnnx(ensure_model(download=download))
    return _model


@register_embedding_function
class GemmaChromaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma adapter: documents via ``__call__``, queries via ``embed_query``.

    Article writes pass explicit title-aware embeddings, so ``__call__`` only
    runs for callers that hand Chroma bare documents; those get ``title: none``.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return list(get_model().embed_documents([None] * len(input), list(input)))

    def embed_query(self, input: Documents) -> Embeddings:
        return list(get_model().embed_queries(list(input)))

    @staticmethod
    def name() -> str:
        return "mc_embeddinggemma2_onnx_q4"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

    def get_config(self) -> dict[str, Any]:
        return {"repo": MODEL_REPO, "revision": MODEL_REVISION, "file": MODEL_FILE}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "GemmaChromaEmbeddingFunction":
        return GemmaChromaEmbeddingFunction()


def main() -> int:
    parser = argparse.ArgumentParser(description="EmbeddingGemma 2 model management")
    parser.add_argument("--download", action="store_true", help="Fetch the pinned model files")
    args = parser.parse_args()
    if args.download:
        print(ensure_model())
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
