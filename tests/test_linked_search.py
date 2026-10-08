"""Tests for cross-project Chroma path resolution."""

from __future__ import annotations


def test_linked_client_opens_active_store(tmp_path, monkeypatch) -> None:
    import linked_search

    active = tmp_path / "knowledge" / "chroma" / "active"
    active.mkdir(parents=True)
    opened: list[str] = []
    sentinel = object()
    monkeypatch.setattr(
        linked_search.chromadb,
        "PersistentClient",
        lambda *, path: opened.append(path) or sentinel,
    )
    linked_search._client_cache.clear()

    assert linked_search._client_for(tmp_path) is sentinel
    assert opened == [str(active)]


import chromadb
import pytest


@pytest.fixture
def fresh_chroma():
    from chromadb.api.shared_system_client import SharedSystemClient

    SharedSystemClient._identifier_to_system = {}
    yield
    SharedSystemClient._identifier_to_system = {}


def _sibling(tmp_path, name):
    path = tmp_path / name / "knowledge" / "chroma" / "active"
    path.mkdir(parents=True)
    return tmp_path / name, chromadb.PersistentClient(path=str(path))


def test_legacy_sibling_is_queried_with_minilm(tmp_path, fresh_chroma):
    import linked_search

    _, client = _sibling(tmp_path, "legacy")
    client.get_or_create_collection("articles", metadata={"hnsw:space": "cosine"}).add(
        ids=["concepts/x::observed"], documents=["stimulus naming rules"],
        metadatas=[{"slug": "concepts/x", "quarantined": False}],
    )
    coll = linked_search._linked_articles_collection(client)
    assert coll.name == "articles"
    assert coll.query(query_texts=["stimulus"], n_results=1)["ids"] == [["concepts/x::observed"]]


def test_migrated_sibling_is_queried_with_gemma(tmp_path, fresh_chroma):
    import embedder
    import linked_search

    _, client = _sibling(tmp_path, "migrated")
    client.get_or_create_collection("articles", metadata={"hnsw:space": "cosine"})
    client.get_or_create_collection(
        "articles_gemma2_q4", metadata={"hnsw:space": "cosine"},
        embedding_function=embedder.GemmaChromaEmbeddingFunction(),
    ).add(ids=["concepts/y::observed"], documents=["bridge session lock"],
          metadatas=[{"slug": "concepts/y", "quarantined": False}])
    coll = linked_search._linked_articles_collection(client)
    assert coll.name == "articles_gemma2_q4"
    assert coll.query(query_texts=["session lock"], n_results=1)["ids"] == [["concepts/y::observed"]]


def test_empty_sibling_is_skipped_and_left_untouched(tmp_path, fresh_chroma):
    import linked_search

    _, client = _sibling(tmp_path, "empty")
    assert linked_search._linked_articles_collection(client) is None
    assert client.list_collections() == []


def test_linked_search_merges_a_legacy_sibling(tmp_path, monkeypatch, fresh_chroma):
    import linked_search

    root, client = _sibling(tmp_path, "legacy")
    client.get_or_create_collection("articles", metadata={"hnsw:space": "cosine"}).add(
        ids=["concepts/x::observed"], documents=["stimulus naming rules"],
        metadatas=[{"slug": "concepts/x", "quarantined": False}],
    )
    linked_search._client_cache.clear()
    monkeypatch.setattr(linked_search, "LINKED_PROJECTS", [root])
    monkeypatch.setattr(linked_search.hybrid_search, "search_articles", lambda **_kw: [])

    hits = linked_search.search_articles_linked(query="stimulus naming", limit=3)
    assert [(h["project"], h["slug"]) for h in hits] == [("legacy", "concepts/x")]
