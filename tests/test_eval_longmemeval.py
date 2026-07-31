"""Tests for the LongMemEval harness: loading, flattening, and scoring.

Uses hand-built miniature datasets rather than the real 264 MB download,
so the suite stays offline and deterministic. What is being pinned here is
the dataset contract and the wiring — the ranking itself is covered by
test_eval_corpus.py and the metrics by test_eval_metrics.py.
"""
from __future__ import annotations

import json

import pytest

import eval_longmemeval as harness


def _session(*messages: str) -> list[dict]:
    """Alternate user/assistant turns from plain strings."""
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": text}
        for i, text in enumerate(messages)
    ]


def _instance(
    question_id: str,
    question: str,
    sessions: dict[str, list[dict]],
    gold: list[str],
) -> dict:
    return {
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": question,
        "answer": "irrelevant to retrieval scoring",
        "question_date": "2026/07/30 (Thu) 10:00",
        "haystack_session_ids": list(sessions),
        "haystack_dates": ["2026/07/01 (Wed) 09:00"] * len(sessions),
        "haystack_sessions": list(sessions.values()),
        "answer_session_ids": gold,
    }


@pytest.fixture
def dataset_path(tmp_path):
    """Two questions whose gold session is unambiguous under BM25."""
    instances = [
        _instance(
            "q1",
            "what did we decide about postgres connection pooling",
            {
                "s1": _session("postgres connection pooling was exhausting slots"),
                "s2": _session("tailwind purges unused utility classes"),
                "s3": _session("the standup is moving to tuesday"),
            },
            gold=["s1"],
        ),
        _instance(
            "q2",
            "which retry policy did we pick for symfony messenger",
            {
                "t1": _session("we chose an exponential symfony messenger retry policy"),
                "t2": _session("lunch options near the office"),
            },
            gold=["t1"],
        ),
    ]
    path = tmp_path / "longmemeval_s.json"
    path.write_text(json.dumps(instances), encoding="utf-8")
    return path


class TestFlattenSession:
    def test_includes_content_from_every_turn(self):
        text = harness.flatten_session(_session("first message", "second message"))
        assert "first message" in text
        assert "second message" in text

    def test_labels_turns_with_their_role(self):
        text = harness.flatten_session([{"role": "user", "content": "hello"}])
        assert "user" in text.lower()

    def test_empty_session_flattens_to_empty_string(self):
        assert harness.flatten_session([]) == ""

    def test_tolerates_a_turn_missing_its_content(self):
        assert harness.flatten_session([{"role": "user"}]) is not None


class TestLoadQuestions:
    def test_loads_every_instance(self, dataset_path):
        assert len(harness.load_questions(dataset_path)) == 2

    def test_carries_question_text_and_gold_ids(self, dataset_path):
        first = harness.load_questions(dataset_path)[0]
        assert first.question_id == "q1"
        assert "postgres" in first.question
        assert first.gold_ids == {"s1"}

    def test_pairs_each_session_id_with_its_flattened_text(self, dataset_path):
        first = harness.load_questions(dataset_path)[0]
        assert [d["id"] for d in first.documents] == ["s1", "s2", "s3"]
        assert "pooling" in first.documents[0]["text"]

    def test_limit_truncates_the_question_list(self, dataset_path):
        assert len(harness.load_questions(dataset_path, limit=1)) == 1

    def test_rejects_misaligned_session_ids_and_sessions(self, tmp_path):
        bad = _instance("q", "?", {"s1": _session("a")}, gold=["s1"])
        bad["haystack_session_ids"] = ["s1", "s2"]  # now longer than sessions
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([bad]), encoding="utf-8")
        with pytest.raises(ValueError, match="haystack"):
            harness.load_questions(path)

    def test_rejects_gold_ids_absent_from_the_haystack(self, tmp_path):
        # Unreachable gold scores 0.0 forever and looks like a retrieval bug.
        bad = _instance("q", "?", {"s1": _session("a")}, gold=["missing"])
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([bad]), encoding="utf-8")
        with pytest.raises(ValueError, match="gold"):
            harness.load_questions(path)


class TestDuplicateSessionIds:
    """LongMemEval-S really does repeat a session id — 15 of its 500 questions.

    Every observed case is byte-identical content and never a gold session,
    so collapsing is safe. Differing content under one id would instead be a
    genuine ambiguity and must not be silently resolved.
    """

    def _dataset_with_repeat(self, tmp_path, second_text: str) -> object:
        instance = _instance(
            "q",
            "postgres pooling",
            {"s1": _session("postgres connection pooling"), "s2": _session("unrelated")},
            gold=["s1"],
        )
        instance["haystack_session_ids"].append("s1")
        instance["haystack_sessions"].append(_session(second_text))
        instance["haystack_dates"].append("2026/07/02 (Thu) 09:00")
        path = tmp_path / "dup.json"
        path.write_text(json.dumps([instance]), encoding="utf-8")
        return path

    def test_collapses_a_repeated_id_with_identical_content(self, tmp_path):
        path = self._dataset_with_repeat(tmp_path, "postgres connection pooling")
        question = harness.load_questions(path)[0]
        assert [d["id"] for d in question.documents] == ["s1", "s2"]

    def test_rejects_a_repeated_id_carrying_different_content(self, tmp_path):
        path = self._dataset_with_repeat(tmp_path, "something else entirely")
        with pytest.raises(ValueError, match="conflicting"):
            harness.load_questions(path)

    def test_reports_the_collapse_rather_than_hiding_it(self, tmp_path, capsys):
        path = self._dataset_with_repeat(tmp_path, "postgres connection pooling")
        harness.load_questions(path)
        assert "duplicate" in capsys.readouterr().err.lower()

    def test_clean_dataset_reports_nothing(self, dataset_path, capsys):
        harness.load_questions(dataset_path)
        assert capsys.readouterr().err == ""


class TestEvaluate:
    def test_perfect_keyword_retrieval_scores_full_recall(self, dataset_path):
        report = harness.evaluate(harness.load_questions(dataset_path), mode="bm25")
        assert report["summary"]["recall@5"] == pytest.approx(1.0)
        assert report["summary"]["questions"] == 2

    def test_records_the_mode_it_ran(self, dataset_path):
        report = harness.evaluate(harness.load_questions(dataset_path), mode="bm25")
        assert report["mode"] == "bm25"

    def test_emits_one_per_question_record_with_its_rank_list(self, dataset_path):
        report = harness.evaluate(harness.load_questions(dataset_path), mode="bm25")
        assert len(report["per_question"]) == 2
        first = report["per_question"][0]
        assert first["question_id"] == "q1"
        assert first["retrieved"][0] == "s1"

    def test_unretrievable_gold_scores_zero_without_crashing(self, tmp_path):
        # Gold session shares no token with the question, and bm25 mode has
        # no embedding fallback — recall must be 0.0, not an exception.
        instance = _instance(
            "q",
            "zzzz qqqq",
            {"gold": _session("entirely unrelated wording"), "other": _session("also unrelated")},
            gold=["gold"],
        )
        path = tmp_path / "miss.json"
        path.write_text(json.dumps([instance]), encoding="utf-8")
        report = harness.evaluate(harness.load_questions(path), mode="bm25")
        assert report["summary"]["recall@5"] == 0.0

    def test_retrieval_depth_covers_the_widest_cutoff(self, dataset_path):
        report = harness.evaluate(
            harness.load_questions(dataset_path), mode="bm25", k_values=(1, 20)
        )
        assert "recall@1" in report["summary"]
        assert "recall@20" in report["summary"]
        assert report["depth"] >= 20


class TestHonestReporting:
    """The report must not claim more than it measured."""

    def test_reciprocal_rank_is_labelled_with_its_depth(self, dataset_path):
        # Results are only materialized to `depth`, so a gold session ranked
        # below it contributes 0 rather than its true reciprocal. Calling that
        # bare "rr" invites comparison against real full-ranking MRR.
        report = harness.evaluate(harness.load_questions(dataset_path), mode="bm25")
        assert f"rr@{report['depth']}" in report["summary"]
        assert "rr" not in report["summary"]

    def test_report_states_how_many_sessions_were_collapsed(self, tmp_path):
        instance = _instance(
            "q",
            "postgres pooling",
            {"s1": _session("postgres connection pooling"), "s2": _session("unrelated")},
            gold=["s1"],
        )
        instance["haystack_session_ids"].append("s1")
        instance["haystack_sessions"].append(_session("postgres connection pooling"))
        path = tmp_path / "dup.json"
        path.write_text(json.dumps([instance]), encoding="utf-8")

        report = harness.evaluate(harness.load_questions(path), mode="bm25")
        assert report["collapsed_duplicate_sessions"] == 1

    def test_report_states_how_many_blank_sessions_were_skipped(self, tmp_path):
        instance = _instance(
            "q",
            "postgres pooling",
            {"s1": _session("postgres connection pooling"), "empty": []},
            gold=["s1"],
        )
        path = tmp_path / "blank.json"
        path.write_text(json.dumps([instance]), encoding="utf-8")

        report = harness.evaluate(harness.load_questions(path), mode="bm25")
        assert report["blank_sessions"] == 1


class TestQuestionValidation:
    def test_an_empty_question_is_rejected(self, tmp_path):
        # An empty query returns nothing under BM25 but becomes an arbitrary
        # nearest-neighbour lookup under vector search — scoring either as a
        # retrieval outcome corrupts the mean.
        instance = _instance("q", "   ", {"s1": _session("content")}, gold=["s1"])
        path = tmp_path / "emptyq.json"
        path.write_text(json.dumps([instance]), encoding="utf-8")
        with pytest.raises(ValueError, match="question"):
            harness.load_questions(path)
