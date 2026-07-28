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
