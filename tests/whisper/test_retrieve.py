"""Unit tests for whisper.retrieve.

Mocks the three search_*_impl functions so we can verify:
  - channel selection honors the scope list
  - parallel execution does NOT duplicate calls
  - RRF merge reranks across channels
  - Hit conversion populates source/category correctly
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest


def _article_hit(slug: str, title: str, category: str = "captured-memory"):
    return {
        "slug": slug,
        "title": title,
        "snippet": f"snippet for {slug}",
        "distance": 0.1,
        "metadata": {"category": category, "confidence": 0.9},
    }


def _code_hit(path: str, preview: str, symbols: list[str] | None = None):
    return {
        "path": path,
        "preview": preview,
        "symbols": symbols or [],
    }


def _daily_hit(slug: str, title: str):
    return {
        "slug": slug,
        "title": title,
        "snippet": f"daily snippet {slug}",
        "distance": 0.15,
        "metadata": {"date": "2026-04-15"},
    }


@pytest.fixture
def mock_searches(monkeypatch):
    ksearch = MagicMock(return_value=[])
    csearch = MagicMock(return_value=[])
    dsearch = MagicMock(return_value=[])
    import whisper.retrieve as r
    monkeypatch.setattr(r, "_search_knowledge_impl", ksearch)
    monkeypatch.setattr(r, "_search_codebase_impl", csearch)
    monkeypatch.setattr(r, "_search_raw_daily_impl", dsearch)
    return ksearch, csearch, dsearch


def test_retrieve_only_fires_channels_in_scope(mock_searches):
    ksearch, csearch, dsearch = mock_searches
    from whisper.retrieve import retrieve

    retrieve(queries=["q1"], scope=["articles"])

    assert ksearch.call_count == 1  # one query × articles
    assert csearch.call_count == 0
    assert dsearch.call_count == 0


def test_retrieve_fans_out_queries_across_selected_channels(mock_searches):
    ksearch, csearch, dsearch = mock_searches
    from whisper.retrieve import retrieve

    retrieve(queries=["q1", "q2", "q3"], scope=["articles", "code"])

    assert ksearch.call_count == 3
    assert csearch.call_count == 3
    assert dsearch.call_count == 0


def test_retrieve_converts_article_hits_to_hit_dataclass(mock_searches):
    ksearch, _, _ = mock_searches
    ksearch.return_value = [_article_hit("concepts/foo", "Foo article", category="governance")]
    from whisper.retrieve import retrieve

    hits = retrieve(queries=["q1"], scope=["articles"])

    assert len(hits) == 1
    assert hits[0].source == "article"
    assert hits[0].category == "governance"
    assert hits[0].path == "concepts/foo"
    assert hits[0].title == "Foo article"


def test_retrieve_converts_code_hits_preserving_symbols(mock_searches):
    _, csearch, _ = mock_searches
    csearch.return_value = [
        _code_hit("src/Service/Foo.php:1-42", "class Foo {...}", symbols=["Foo", "bar"])
    ]
    from whisper.retrieve import retrieve

    hits = retrieve(queries=["q1"], scope=["code"])

    assert len(hits) == 1
    assert hits[0].source == "code"
    assert hits[0].category is None
    assert hits[0].path == "src/Service/Foo.php:1-42"
    assert hits[0].symbols == ["Foo", "bar"]


def test_retrieve_rrf_merges_duplicates_from_multiple_queries(mock_searches):
    ksearch, _, _ = mock_searches
    # Same article returned by two queries — should merge, not duplicate
    ksearch.side_effect = [
        [_article_hit("concepts/foo", "Foo")],
        [_article_hit("concepts/foo", "Foo")],
    ]
    from whisper.retrieve import retrieve

    hits = retrieve(queries=["q1", "q2"], scope=["articles"])

    slugs = [h.path for h in hits]
    assert slugs.count("concepts/foo") == 1


def test_retrieve_limits_to_top_n(mock_searches):
    ksearch, _, _ = mock_searches
    ksearch.return_value = [
        _article_hit(f"concepts/a{i}", f"A{i}") for i in range(30)
    ]
    from whisper.retrieve import retrieve

    hits = retrieve(queries=["q1"], scope=["articles"], top_n=12)

    assert len(hits) == 12


def test_retrieve_assigns_sequential_citation_ids(mock_searches):
    ksearch, _, _ = mock_searches
    ksearch.return_value = [_article_hit(f"concepts/a{i}", f"A{i}") for i in range(3)]
    from whisper.retrieve import retrieve

    hits = retrieve(queries=["q1"], scope=["articles"])

    assert [h.id for h in hits] == ["c1", "c2", "c3"]


def test_retrieve_empty_queries_returns_empty_list(mock_searches):
    from whisper.retrieve import retrieve

    assert retrieve(queries=[], scope=["articles"]) == []


def test_retrieve_scope_filtered_to_valid_channels(mock_searches):
    ksearch, csearch, _ = mock_searches
    from whisper.retrieve import retrieve

    # bogus channel ignored, valid ones fire
    retrieve(queries=["q1"], scope=["articles", "bogus", "code"])

    assert ksearch.called
    assert csearch.called


def test_rrf_prefers_items_ranked_high_in_multiple_queries():
    """Sanity-check the fusion math directly."""
    from whisper.retrieve import rrf_merge

    q1_results = [("A", 0.1), ("B", 0.2), ("C", 0.3)]
    q2_results = [("B", 0.1), ("A", 0.2), ("D", 0.3)]
    fused = rrf_merge([q1_results, q2_results], k=60)

    # A and B both appear in both lists at top ranks; they should win over C and D
    top_two_keys = [key for key, _score in fused[:2]]
    assert set(top_two_keys) == {"A", "B"}
