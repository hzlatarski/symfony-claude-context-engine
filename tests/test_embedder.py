"""Tests for the EmbeddingGemma 2 ONNX embedder (REQ-1)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import embedder

REFERENCE = Path(__file__).parent / "fixtures" / "embeddinggemma2_reference.json"


@pytest.fixture(scope="module")
def model():
    return embedder.get_model()


def test_query_prefix():
    assert embedder.query_text("where is X") == "task: search result | query: where is X"


def test_document_prefix_uses_title_or_none():
    assert embedder.document_text("Bridge Lock", "body") == "title: Bridge Lock | text: body"
    assert embedder.document_text("", "body") == "title: none | text: body"
    assert embedder.document_text(None, "body") == "title: none | text: body"


def test_vectors_are_768d_and_unit_length(model):
    vectors = model.embed_documents(["A", None], ["first text", "second text"])
    assert vectors.shape == (2, 768)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_empty_input_returns_empty_matrix(model):
    assert model.encode([]).shape == (0, 768)


def test_batch_order_is_preserved(model):
    texts = ["short", "a much longer text " * 40, "medium length text here"]
    together = model.encode(texts)
    alone = np.vstack([model.encode([t]) for t in texts])
    assert np.allclose(together, alone, atol=1e-4)


def test_input_is_truncated_at_max_tokens(model):
    """Text past MAX_TOKENS is cut off: a tail after it changes nothing.
    The same tail on a short text moves the vector a lot (cosine ~0.82)."""
    tail = " volcanoes erupt in iceland every spring"
    long_text = "word " * (embedder.MAX_TOKENS * 2)
    cut = model.encode([long_text, long_text + tail])
    assert cut.shape == (2, 768) and np.isfinite(cut).all()
    assert float((cut[0] * cut[1]).sum()) > 0.9999
    short = model.encode(["word " * 50, "word " * 50 + tail])
    assert float((short[0] * short[1]).sum()) < 0.95


def test_matches_fp32_reference_vectors(model):
    """AC-1.4: the q4 model stays close to the full-precision model."""
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    ours = model.encode(reference["texts"])
    cosine = (ours * np.array(reference["vectors"], dtype=np.float32)).sum(axis=1)
    assert (cosine >= 0.98).all(), cosine


def test_query_and_document_embeddings_differ(model):
    as_query = model.embed_queries(["stimulus naming"])
    as_doc = model.embed_documents([None], ["stimulus naming"])
    # Measured 0.973-0.979: the prefixes really change the vector.
    assert float((as_query * as_doc).sum()) < 0.99


def test_chroma_adapter_routes_queries_and_documents(model):
    ef = embedder.GemmaChromaEmbeddingFunction()
    doc = np.array(ef(["stimulus naming"])[0])
    query = np.array(ef.embed_query(["stimulus naming"])[0])
    assert np.allclose(doc, model.embed_documents([None], ["stimulus naming"])[0], atol=1e-5)
    assert np.allclose(query, model.embed_queries(["stimulus naming"])[0], atol=1e-5)
    assert ef.name() == "mc_embeddinggemma2_onnx_q4"
    assert ef.default_space() == "cosine"


def test_missing_model_without_network_names_path_and_fix(tmp_path, monkeypatch):
    """AC-1.6: an offline first run fails with an actionable message."""
    monkeypatch.setenv("MC_EMBEDDER_CACHE", str(tmp_path))

    import huggingface_hub

    def offline(*_args, **_kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(embedder.EmbedderUnavailable) as raised:
        embedder.ensure_model()
    message = str(raised.value)
    assert str(embedder.model_dir()) in message
    assert "scripts/embedder.py --download" in message
    assert "MC_EMBEDDER=minilm" in message


def test_model_is_downloaded_with_local_dir(tmp_path, monkeypatch):
    """AC-1.5: plain files under the revision folder, not the HF symlink cache."""
    monkeypatch.setenv("MC_EMBEDDER_CACHE", str(tmp_path))
    calls = []

    import huggingface_hub

    def fake_download(repo, name, revision, local_dir):
        calls.append((repo, name, revision, local_dir))
        target = Path(local_dir) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    directory = embedder.ensure_model()
    assert directory == tmp_path / "embeddinggemma-2-onnx" / embedder.MODEL_REVISION
    assert sorted(c[1] for c in calls) == sorted(embedder.MODEL_FILES)
    assert {c[2] for c in calls} == {embedder.MODEL_REVISION}
    assert {c[3] for c in calls} == {str(directory)}

    calls.clear()
    embedder.ensure_model()
    assert calls == []  # present files are never fetched again


def test_search_path_never_downloads(tmp_path, monkeypatch):
    """A missing model must fail fast inside the prompt hook, not fetch 170 MB."""
    monkeypatch.setenv("MC_EMBEDDER_CACHE", str(tmp_path))
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(embedder, "_model", None)
    with pytest.raises(embedder.EmbedderUnavailable, match="scripts/embedder.py --download"):
        embedder.get_model()
    assert calls == []
