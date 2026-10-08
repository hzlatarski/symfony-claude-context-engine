"""Tests for the missing-vector check and repair (scripts/vector_health.py)."""
from __future__ import annotations

import threading
from contextlib import contextmanager

import pytest

import config
import vector_health


class FakeCollection:
    """Records whose id is in ``lost`` raise on embedding reads, like the live store."""

    def __init__(self, ids, lost):
        self.records = {i: (f"doc {i}", {"title": f"T{i}"}) for i in ids}
        self.lost = set(lost)
        self.deleted, self.upserted = [], []

    def get(self, ids=None, include=()):
        ids = list(self.records) if ids is None else ids
        if "embeddings" in include and self.lost.intersection(ids):
            raise RuntimeError("Error executing plan: Internal error: Error finding id")
        return {
            "ids": ids,
            "documents": [self.records[i][0] for i in ids],
            "metadatas": [self.records[i][1] for i in ids],
        }

    def delete(self, ids):
        self.deleted.extend(ids)

    def upsert(self, ids, documents, metadatas, embeddings):
        self.upserted.append((ids, documents, metadatas, embeddings))
        self.lost.difference_update(ids)


class BrokenCollection:
    def get(self, ids=None, include=()):
        raise RuntimeError("Failed to apply logs to the hnsw segment writer")


def test_bisect_finds_exactly_the_lost_ids():
    ids = [f"id-{n}" for n in range(1000)]
    lost = {"id-3", "id-417", "id-418", "id-999"}
    col = FakeCollection(ids, lost)
    assert sorted(vector_health.unreadable_ids(col, ids)) == sorted(lost)


def test_healthy_collection_reports_nothing():
    ids = [f"id-{n}" for n in range(300)]
    assert vector_health.unreadable_ids(FakeCollection(ids, set()), ids) == []


@pytest.fixture
def fake_store(monkeypatch):
    """Route get_client / write_session to fake collections; record lock state."""
    state = {"sessions": [], "in_session": False, "embedded_in_session": []}

    def install(collections):
        class Client:
            def get_collection(self, name, embedding_function=None):
                return collections[name]

            def list_collections(self):
                return list(collections)

        @contextmanager
        def write_session(name):
            state["sessions"].append(name)
            state["in_session"] = True
            try:
                yield Client()
            finally:
                state["in_session"] = False

        monkeypatch.setattr(vector_health.chroma_client, "write_session", write_session)
        monkeypatch.setattr(vector_health.chroma_client, "get_client", lambda: Client())
        monkeypatch.setattr(vector_health.chroma_client, "lock", threading.RLock())

        def fake_embed(name, docs, metas):
            state["embedded_in_session"].append(state["in_session"])
            return [[0.5] * 4 for _ in docs]

        monkeypatch.setattr(vector_health, "_embed", fake_embed)
        return state

    return install


def test_repair_upserts_same_ids_texts_and_metadata_without_deleting(fake_store):
    col = FakeCollection(["a", "b", "c"], {"b", "c"})
    state = fake_store({"daily_chunks_v2": col})

    assert vector_health.repair_collection("daily_chunks_v2", ["b", "c"]) == 2
    assert state["sessions"] == ["daily_chunks_v2"]  # every write goes through the safe session
    assert col.deleted == []  # a crash mid-repair can never lose a record
    ids, docs, metas, embeddings = col.upserted[0]
    assert ids == ["b", "c"]
    assert docs == ["doc b", "doc c"]
    assert metas == [{"title": "Tb"}, {"title": "Tc"}]
    assert embeddings == [[0.5] * 4, [0.5] * 4]
    assert vector_health.unreadable_ids(col, ["a", "b", "c"]) == []


def test_vectors_are_computed_outside_the_write_lock(fake_store):
    col = FakeCollection(["x", "y"], {"x", "y"})
    state = fake_store({"codebase": col})
    vector_health.repair_collection("codebase", ["x", "y"])
    assert state["embedded_in_session"] == [False]


def test_gemma_embed_uses_titles(monkeypatch):
    seen = []

    class Model:
        def embed_documents(self, titles, texts):
            seen.append((titles, texts))
            return [[0.0] * 768 for _ in texts]

    monkeypatch.setattr(vector_health.embedder, "get_model", lambda download=False: Model())
    out = vector_health._embed(config.CHROMA_COLLECTION_ARTICLES_GEMMA, ["doc x"], [{"title": "Tx"}])
    assert seen == [(["Tx"], ["doc x"])]
    assert out == [[0.0] * 768]


def test_nothing_damaged_means_no_writes(fake_store):
    state = fake_store({"codebase": FakeCollection(["a"], set())})
    assert vector_health.repair_collection("codebase", []) == 0
    assert state["sessions"] == []


def test_default_check_covers_only_collections_in_use(monkeypatch):
    monkeypatch.setenv("MC_EMBEDDER", "gemma2-q4")
    assert vector_health.in_use() == [
        config.CHROMA_COLLECTION_ARTICLES_GEMMA,
        config.CHROMA_COLLECTION_DAILY,
        config.CHROMA_COLLECTION_CODEBASE,
    ]
    assert "daily_chunks" not in vector_health.in_use()  # the abandoned, unreadable one


def test_broken_or_missing_collection_is_reported_not_raised(fake_store):
    fake_store({"codebase": FakeCollection(["a"], {"a"}), "daily_chunks_v2": BrokenCollection()})
    report = vector_health.check(["daily_chunks_v2", "articles_gemma2_q4", "codebase"])
    assert report["daily_chunks_v2"][2].startswith("UNREADABLE")
    assert report["articles_gemma2_q4"][2] == "missing (not built yet)"
    assert report["codebase"] == (1, ["a"], None)
