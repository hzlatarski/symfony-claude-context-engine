"""Tests for the ephemeral benchmark corpus.

The harness must rank a *foreign* corpus (LongMemEval sessions) through
the same BM25 + vector + RRF path the live store uses. These tests prove
each stream is actually wired, not stubbed — in particular that the vector
stream really embeds, since a silently-dead vector path would make every
hybrid benchmark number a BM25 number wearing a hat.
"""
from __future__ import annotations

import pytest

from eval_corpus import EphemeralCorpus


@pytest.fixture
def small_corpus():
    return [
        {"id": "s1", "text": "postgres connection pooling exhausted under load"},
        {"id": "s2", "text": "symfony messenger retry policy and failure transport"},
        {"id": "s3", "text": "tailwind utility classes are purged at build time"},
    ]


class TestBm25Mode:
    def test_ranks_the_lexically_matching_document_first(self, small_corpus):
        corpus = EphemeralCorpus(small_corpus)
        assert corpus.search("messenger retry transport", limit=3, mode="bm25")[0] == "s2"

    def test_returns_ids_only(self, small_corpus):
        corpus = EphemeralCorpus(small_corpus)
        results = corpus.search("postgres", limit=3, mode="bm25")
        assert all(isinstance(r, str) for r in results)

    def test_respects_the_limit(self, small_corpus):
        corpus = EphemeralCorpus(small_corpus)
        assert len(corpus.search("postgres symfony tailwind", limit=2, mode="bm25")) == 2

    def test_returns_nothing_for_a_query_sharing_no_tokens(self, small_corpus):
        corpus = EphemeralCorpus(small_corpus)
        assert corpus.search("kubernetes", limit=3, mode="bm25") == []


class TestVectorMode:
    def test_finds_a_semantic_match_with_no_shared_tokens(self):
        # No token overlap at all: if this passes, the embedding path is live.
        corpus = EphemeralCorpus([
            {"id": "feline", "text": "the feline was resting quietly upon the rug"},
            {"id": "finance", "text": "quarterly revenue guidance was revised downward"},
        ])
        assert corpus.search("cat sleeping on a carpet", limit=1, mode="vector") == ["feline"]

    def test_respects_the_limit(self, small_corpus):
        corpus = EphemeralCorpus(small_corpus)
        assert len(corpus.search("database", limit=2, mode="vector")) == 2


class TestHybridMode:
    def test_recovers_a_document_only_the_vector_stream_can_see(self):
        corpus = EphemeralCorpus([
            {"id": "feline", "text": "the feline was resting quietly upon the rug"},
            {"id": "finance", "text": "quarterly revenue guidance was revised downward"},
        ])
        assert "feline" in corpus.search("cat sleeping on a carpet", limit=2, mode="hybrid")

    def test_recovers_a_document_only_the_keyword_stream_can_see(self, small_corpus):
        corpus = EphemeralCorpus(small_corpus)
        # An exact identifier is BM25's strength and embeddings' weakness.
        assert "s2" in corpus.search("messenger", limit=2, mode="hybrid")

    def test_respects_the_limit(self, small_corpus):
        corpus = EphemeralCorpus(small_corpus)
        assert len(corpus.search("retry policy", limit=1, mode="hybrid")) == 1


class TestIsolationBetweenCorpora:
    """Each corpus must be a sealed universe.

    ``chromadb.EphemeralClient()`` does NOT give you a private store — it
    resolves through a shared in-process system client, so two clients
    asking for the same collection name get the *same* collection. In a
    benchmark that builds one index per question that silently merges every
    haystack into one, and recall collapses as the run proceeds. These
    tests exist because that bug shipped once already.
    """

    def test_a_second_corpus_cannot_see_the_first_corpus_documents(self):
        first = EphemeralCorpus([{"id": "leaked", "text": "postgres connection pooling"}])
        first.search("postgres", limit=5, mode="vector")  # force the index to build

        second = EphemeralCorpus([{"id": "own", "text": "tailwind utility classes"}])
        assert second.search("postgres connection pooling", limit=5, mode="vector") == ["own"]

    def test_hybrid_mode_is_isolated_too(self):
        first = EphemeralCorpus([{"id": "leaked", "text": "postgres connection pooling"}])
        first.search("postgres", limit=5, mode="hybrid")

        second = EphemeralCorpus([{"id": "own", "text": "tailwind utility classes"}])
        assert second.search("postgres connection pooling", limit=5, mode="hybrid") == ["own"]

    def test_many_corpora_never_accumulate(self):
        # Mimics the shape of a real run: build a lot of small indexes and
        # confirm the last one still only knows its own single document.
        for index in range(12):
            corpus = EphemeralCorpus([{"id": f"doc{index}", "text": f"topic number {index}"}])
            results = corpus.search("topic", limit=10, mode="vector")
            assert results == [f"doc{index}"], f"corpus {index} saw foreign documents: {results}"

    def test_closing_a_corpus_releases_its_collection(self):
        corpus = EphemeralCorpus([{"id": "a", "text": "alpha"}])
        corpus.search("alpha", limit=1, mode="vector")
        corpus.close()
        # Closing twice must not raise — evaluate() closes in a finally block.
        corpus.close()

    def test_works_as_a_context_manager(self):
        with EphemeralCorpus([{"id": "a", "text": "alpha beta"}]) as corpus:
            assert corpus.search("alpha", limit=1, mode="vector") == ["a"]


class TestValidation:
    def test_rejects_an_unknown_mode(self, small_corpus):
        corpus = EphemeralCorpus(small_corpus)
        with pytest.raises(ValueError, match="mode"):
            corpus.search("anything", limit=3, mode="telepathy")

    def test_rejects_duplicate_document_ids(self):
        # A duplicate id would double-count in recall and inflate the score.
        with pytest.raises(ValueError, match="duplicate"):
            EphemeralCorpus([
                {"id": "same", "text": "first"},
                {"id": "same", "text": "second"},
            ])

    def test_empty_corpus_searches_to_nothing(self):
        corpus = EphemeralCorpus([])
        assert corpus.search("anything", limit=5, mode="hybrid") == []

    def test_reports_its_size(self, small_corpus):
        assert len(EphemeralCorpus(small_corpus)) == 3

    def test_tolerates_a_document_with_empty_text(self):
        # Real session transcripts occasionally render to nothing; the run
        # must continue rather than abort 300 questions in.
        corpus = EphemeralCorpus([
            {"id": "blank", "text": ""},
            {"id": "real", "text": "postgres connection pooling"},
        ])
        assert corpus.search("postgres", limit=2, mode="hybrid")[0] == "real"


class TestBlankDocumentsAreNotEmbedded:
    """A blank document must never occupy a vector rank.

    An earlier version substituted a "∅" placeholder so Chroma would accept
    the row, with a comment claiming it "cannot match anything". That is
    false: nearest-neighbour search always returns *some* ranking, so the
    placeholder got a real embedding and could surface above genuine
    evidence. LongMemEval-S contains 1228 zero-turn sessions, so this was
    not hypothetical — roughly 5% of the indexed corpus was noise.
    """

    def test_blank_document_never_appears_in_vector_results(self):
        corpus = EphemeralCorpus([
            {"id": "blank", "text": "   "},
            {"id": "real", "text": "postgres connection pooling"},
        ])
        assert corpus.search("anything at all", limit=10, mode="vector") == ["real"]

    def test_blank_document_never_appears_in_hybrid_results(self):
        corpus = EphemeralCorpus([
            {"id": "blank", "text": ""},
            {"id": "real", "text": "postgres connection pooling"},
        ])
        assert "blank" not in corpus.search("unrelated query", limit=10, mode="hybrid")

    def test_the_number_of_skipped_blanks_is_reported(self):
        corpus = EphemeralCorpus([
            {"id": "b1", "text": ""},
            {"id": "b2", "text": "  \n "},
            {"id": "real", "text": "content"},
        ])
        assert corpus.blank_documents == 2

    def test_a_corpus_of_only_blanks_searches_to_nothing(self):
        corpus = EphemeralCorpus([{"id": "b1", "text": ""}, {"id": "b2", "text": ""}])
        assert corpus.search("query", limit=5, mode="vector") == []


class TestChunkText:
    """Windowing is the unit that decides what the embedder ever sees."""

    def test_short_text_is_a_single_chunk(self):
        from eval_corpus import chunk_text

        assert chunk_text("three short words", size=150, overlap=30) == ["three short words"]

    def test_long_text_is_split_into_windows(self):
        from eval_corpus import chunk_text

        words = " ".join(str(i) for i in range(500))
        chunks = chunk_text(words, size=100, overlap=20)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_windows_overlap_so_a_boundary_fact_survives(self):
        from eval_corpus import chunk_text

        words = " ".join(str(i) for i in range(200))
        chunks = chunk_text(words, size=100, overlap=20)
        # The last 20 words of chunk 0 must reappear at the head of chunk 1.
        assert chunks[0].split()[-20:] == chunks[1].split()[:20]

    def test_every_word_appears_somewhere(self):
        from eval_corpus import chunk_text

        words = [str(i) for i in range(250)]
        chunks = chunk_text(" ".join(words), size=100, overlap=20)
        covered = {w for c in chunks for w in c.split()}
        assert covered == set(words)

    def test_overlap_must_be_smaller_than_size(self):
        from eval_corpus import chunk_text

        with pytest.raises(ValueError, match="overlap"):
            chunk_text("a b c", size=10, overlap=10)

    def test_blank_text_yields_no_chunks(self):
        from eval_corpus import chunk_text

        assert chunk_text("   ", size=100, overlap=20) == []


class TestChunkedCorpus:
    """Chunking exists because the embedder only sees a ~190-word window.

    A fact buried 1000 words into a session is invisible to a whole-document
    embedding. Splitting the document and ranking it by its best chunk
    recovers it — measured at +12pp recall@5 on LongMemEval-S.
    """

    def _haystack(self):
        filler = " ".join(f"unrelated filler sentence {i}" for i in range(400))
        return [
            # The distinguishing fact sits far past any embedder's window.
            {"id": "buried", "text": f"{filler} the deployment key rotates every ninety days"},
            {"id": "other", "text": " ".join(f"quarterly revenue commentary {i}" for i in range(400))},
        ]

    def test_finds_a_fact_buried_past_the_embedder_window(self):
        corpus = EphemeralCorpus(self._haystack(), chunk_words=120, chunk_overlap=20)
        assert corpus.search("how often does the deployment key rotate", limit=1, mode="vector") == ["buried"]

    def test_results_are_parent_ids_not_chunk_ids(self):
        corpus = EphemeralCorpus(self._haystack(), chunk_words=120, chunk_overlap=20)
        results = corpus.search("deployment key rotation", limit=5, mode="vector")
        assert all("##" not in r for r in results)

    def test_a_parent_appears_at_most_once(self):
        corpus = EphemeralCorpus(self._haystack(), chunk_words=120, chunk_overlap=20)
        results = corpus.search("filler sentence", limit=5, mode="vector")
        assert len(results) == len(set(results))

    def test_limit_counts_parents_not_chunks(self):
        corpus = EphemeralCorpus(self._haystack(), chunk_words=120, chunk_overlap=20)
        assert len(corpus.search("sentence", limit=1, mode="vector")) == 1

    def test_chunking_is_off_by_default(self):
        corpus = EphemeralCorpus([{"id": "a", "text": "alpha beta"}])
        assert corpus.chunked is False

    def test_hybrid_still_works_with_chunking_enabled(self):
        corpus = EphemeralCorpus(self._haystack(), chunk_words=120, chunk_overlap=20)
        assert "buried" in corpus.search("deployment key rotates", limit=2, mode="hybrid")

    def test_blank_documents_are_still_excluded(self):
        corpus = EphemeralCorpus(
            [{"id": "blank", "text": ""}, {"id": "real", "text": "postgres pooling"}],
            chunk_words=120,
            chunk_overlap=20,
        )
        assert corpus.search("anything", limit=5, mode="vector") == ["real"]

    def test_chunked_corpora_are_isolated_from_each_other(self):
        first = EphemeralCorpus([{"id": "leaked", "text": "postgres pooling"}], chunk_words=50, chunk_overlap=10)
        first.search("postgres", limit=1, mode="vector")
        second = EphemeralCorpus([{"id": "own", "text": "tailwind classes"}], chunk_words=50, chunk_overlap=10)
        assert second.search("postgres pooling", limit=5, mode="vector") == ["own"]
