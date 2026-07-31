"""Tests for the reusable BM25 primitive extracted from bm25_store.

``TokenIndex`` is what lets the benchmark harness score *our* keyword
ranking — same tokenizer, same negative-IDF gate — over a foreign corpus
instead of a reimplementation that would measure nothing.
"""
from __future__ import annotations

import pytest

from bm25_store import TokenIndex


class TestRanking:
    def test_ranks_the_document_containing_the_query_term_first(self):
        index = TokenIndex([
            "postgres connection pooling",
            "symfony messenger retry policy",
            "tailwind utility classes",
        ])
        ranked = index.rank("messenger retry")
        assert ranked[0][0] == 1

    def test_returns_corpus_index_and_score_pairs(self):
        index = TokenIndex(["alpha beta", "gamma delta"])
        ranked = index.rank("alpha")
        assert len(ranked) == 1
        position, score = ranked[0]
        assert position == 0
        assert isinstance(score, float)

    def test_orders_by_descending_score(self):
        index = TokenIndex([
            "cache",
            "cache cache cache invalidation strategy",
            "unrelated text",
        ])
        ranked = index.rank("cache invalidation")
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)


class TestNegativeIdfGate:
    def test_keeps_documents_whose_score_went_negative(self):
        # A term present in every document drives BM25Okapi's IDF negative,
        # so "score > 0" would discard every true match. The gate is token
        # presence, not score — this is the behavior the gate exists for.
        index = TokenIndex(["python tooling", "python packaging", "python typing"])
        ranked = index.rank("python")
        assert len(ranked) == 3
        assert all(score < 0 for _, score in ranked)

    def test_excludes_documents_sharing_no_query_token(self):
        index = TokenIndex(["python tooling", "rust borrow checker"])
        ranked = index.rank("rust")
        assert [position for position, _ in ranked] == [1]


class TestTokenization:
    def test_uses_the_project_tokenizer_for_camel_case(self):
        index = TokenIndex(["HybridEmailValidationService handles blur events"])
        assert index.rank("email validation")
        assert index.rank("HybridEmailValidationService")

    def test_matches_across_snake_and_kebab_boundaries(self):
        index = TokenIndex(["the enhanced_email_validation_controller fires on blur"])
        assert index.rank("enhanced email validation")


class TestDegenerateInputs:
    def test_empty_corpus_ranks_nothing(self):
        assert TokenIndex([]).rank("anything") == []

    def test_query_with_no_usable_tokens_ranks_nothing(self):
        index = TokenIndex(["some real content"])
        assert index.rank("!!! ?") == []

    def test_corpus_of_untokenizable_text_ranks_nothing(self):
        index = TokenIndex(["!!!", "???"])
        assert index.rank("hello") == []

    def test_length_reports_the_corpus_size(self):
        assert len(TokenIndex(["a b", "c d", "e f"])) == 3


class TestStoreStillWorks:
    """The extraction must not change bm25_store's public search behavior."""

    def test_search_articles_is_still_exported(self):
        import bm25_store

        assert callable(bm25_store.search_articles)
        assert callable(bm25_store.tokenize)
