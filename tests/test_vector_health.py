"""Tests for the missing-vector check and repair (scripts/vector_health.py)."""
from __future__ import annotations

from contextlib import contextmanager

import pytest

import vector_health


class FakeCollection:
    """Records whose id is in ``lost`` raise on embedding reads, like the live store."""

    def __init__(self, ids, lost):
        self.records = {i: (f"doc {i}", {"title": f"T{i}"}) for i in ids}
        self.lost = set(lost)
        self.deleted, self.added = [], []

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

    def add(self, ids, documents, metadatas, embeddings=None):
        self.added.append((ids, documents, metadatas, embeddings))
        self.lost.difference_update(ids)


def test_bisect_finds_exactly_the_lost_ids():
    ids = [f"id-{n}" for n in range(1000)]
    lost = {"id-3", "id-417", "id-418", "id-999"}
    col = FakeCollection(ids, lost)
    assert sorted(vector_health.unreadable_ids(col, ids)) == sorted(lost)


def test_healthy_collection_reports_nothing():
    ids = [f"id-{n}" for n in range(300)]
    assert vector_health.unreadable_ids(FakeCollection(ids, set()), ids) == []


@pytest.fixture
def fake_session(monkeypatch):
    sessions = []

    def install(collection):
        class Client:
            def get_collection(self, name, embedding_function=None):
                return collection

        @contextmanager
        def write_session(name):
            sessions.append(name)
            yield Client()

        monkeypatch.setattr(vector_health.chroma_client, "write_session", write_session)
        return sessions

    return install


def test_repair_re_adds_same_ids_texts_and_metadata(fake_session):
    col = FakeCollection(["a", "b", "c"], {"b", "c"})
    sessions = fake_session(col)

    assert vector_health.repair_collection("daily_chunks", ["b", "c"]) == 2
    assert sessions == ["daily_chunks"]  # every write goes through the safe session
    assert col.deleted == ["b", "c"]
    ids, docs, metas, embeddings = col.added[0]
    assert ids == ["b", "c"]
    assert docs == ["doc b", "doc c"]
    assert metas == [{"title": "Tb"}, {"title": "Tc"}]
    assert embeddings is None  # MiniLM collections re-embed through Chroma
    assert vector_health.unreadable_ids(col, ["a", "b", "c"]) == []


def test_gemma_repair_embeds_with_titles(fake_session, monkeypatch):
    col = FakeCollection(["x"], {"x"})
    fake_session(col)
    seen = []

    class Model:
        def embed_documents(self, titles, texts):
            seen.append((titles, texts))
            return [[0.0] * 768]

    monkeypatch.setattr(vector_health.embedder, "get_model", lambda: Model())
    vector_health.repair_collection("articles_gemma2_q4", ["x"])
    assert seen == [(["Tx"], ["doc x"])]
    assert col.added[0][3] == [[0.0] * 768]


def test_nothing_damaged_means_no_writes(fake_session):
    sessions = fake_session(FakeCollection(["a"], set()))
    assert vector_health.repair_collection("codebase", []) == 0
    assert sessions == []
