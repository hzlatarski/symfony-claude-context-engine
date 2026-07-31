"""LongMemEval-S retrieval benchmark for the memory compiler.

Answers the question our test suite could not: **is our retrieval actually
any good, and did that tuning change help or hurt?** Every knob we own —
the RRF constant, the pool multiplier, the tokenizer, the embedding model,
chunk sizing — has until now been set by argument rather than measurement.

Protocol (LongMemEval's, so the numbers are comparable to published ones):
each question ships its own haystack of ~48 chat sessions, of which one or
more are the gold evidence. We build a throwaway index over *that*
question's haystack, search it with the question text, and score whether
gold surfaced. No answer generation — this measures retrieval alone.

Usage
-----
The dataset is not vendored (264 MB). Fetch ``longmemeval_s.json`` from
the LongMemEval release on Hugging Face, then::

    uv run python scripts/eval_longmemeval.py --dataset path/to/longmemeval_s.json --mode bm25
    uv run python scripts/eval_longmemeval.py --dataset path/to/longmemeval_s.json --mode hybrid
    uv run python scripts/eval_longmemeval.py --dataset ... --mode hybrid --limit 50 --json out.json

``--mode bm25`` costs no embeddings and runs in seconds, which makes it the
right default for a quick regression check. ``--mode hybrid`` embeds every
session in every haystack and is the number that reflects live search.
Start with ``--limit`` to size the run before committing to all 500.

Caveat on reciprocal rank: results are only materialized to ``depth`` (the
widest cutoff), so a gold session ranked below ``depth`` contributes 0
rather than its true reciprocal. The metric is therefore reported as
``rr@<depth>`` and must not be compared against published full-ranking MRR.
Recall and NDCG at their cutoffs are exact.

The report also carries ``collapsed_duplicate_sessions`` and
``blank_sessions``: both are protocol deviations from the raw dataset, and a
run is only comparable to another run that reports the same numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import eval_metrics
from eval_corpus import MODES, EphemeralCorpus


@dataclass(frozen=True)
class Question:
    """One benchmark instance: a query, its haystack, and its gold ids."""

    question_id: str
    question: str
    question_type: str
    documents: list[dict[str, str]]
    gold_ids: set[str] = field(default_factory=set)


def flatten_session(turns: Sequence[dict[str, Any]]) -> str:
    """Render a session's turns into one searchable string.

    Roles are kept as labels: they cost almost nothing under BM25's IDF and
    give the embedding model the conversational shape it was trained on.
    """
    lines = []
    for turn in turns:
        role = str(turn.get("role") or "unknown")
        content = str(turn.get("content") or "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class QuestionList(list):
    """A list of ``Question`` that also carries dataset-level provenance.

    ``collapsed_duplicates`` records how many session entries the loader
    removed. The benchmark protocol is only comparable to another run if
    that number is visible, so it travels with the data rather than being
    printed once and forgotten.
    """

    collapsed_duplicates: int = 0


def _collapse_repeated_sessions(
    question_id: str,
    documents: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    """Drop verbatim repeats of a session id; refuse conflicting ones.

    LongMemEval-S lists the same session twice in 15 of its 500 questions,
    always with byte-identical content and never for a gold session. Those
    are safe to collapse — indexing identical text twice under one id just
    wastes an embedding and would break the corpus's uniqueness invariant.

    The same id carrying *different* content is a different animal: it
    would make "did we retrieve the gold session" ambiguous, so it raises
    rather than picking a winner.
    """
    kept: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    collapsed = 0

    for doc in documents:
        doc_id = doc["id"]
        if doc_id not in seen:
            seen[doc_id] = doc["text"]
            kept.append(doc)
            continue
        if seen[doc_id] != doc["text"]:
            raise ValueError(
                f"{question_id}: session id {doc_id!r} appears twice with "
                f"conflicting content — cannot score unambiguously"
            )
        collapsed += 1

    return kept, collapsed


def load_questions(path: Path | str, limit: int | None = None) -> list[Question]:
    """Parse a LongMemEval JSON file into ``Question`` records.

    Validates the two dataset invariants that would otherwise corrupt the
    scores silently: session ids must align 1:1 with sessions, and every
    gold id must exist in its haystack. Unreachable gold would score 0.0
    forever and read as a retrieval failure rather than a data problem.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if limit is not None:
        raw = raw[:limit]

    questions = QuestionList()
    collapsed_total = 0
    for instance in raw:
        question_id = str(instance.get("question_id", "<missing>"))
        session_ids = instance.get("haystack_session_ids") or []
        sessions = instance.get("haystack_sessions") or []
        if len(session_ids) != len(sessions):
            raise ValueError(
                f"{question_id}: haystack_session_ids ({len(session_ids)}) and "
                f"haystack_sessions ({len(sessions)}) must be the same length"
            )

        documents, collapsed = _collapse_repeated_sessions(
            question_id,
            [
                {"id": str(session_id), "text": flatten_session(session)}
                for session_id, session in zip(session_ids, sessions)
            ],
        )
        collapsed_total += collapsed
        available = {doc["id"] for doc in documents}
        gold = {str(g) for g in (instance.get("answer_session_ids") or [])}
        missing = gold - available
        if missing:
            raise ValueError(
                f"{question_id}: gold session ids absent from haystack: {sorted(missing)}"
            )
        if not gold:
            raise ValueError(f"{question_id}: no gold session ids — cannot be scored")

        question_text = str(instance.get("question") or "")
        if not question_text.strip():
            raise ValueError(
                f"{question_id}: empty question text — an empty query returns "
                f"nothing under BM25 but becomes an arbitrary nearest-neighbour "
                f"lookup under vector search, so it cannot be scored"
            )

        questions.append(
            Question(
                question_id=question_id,
                question=question_text,
                question_type=str(instance.get("question_type") or "unknown"),
                documents=documents,
                gold_ids=gold,
            )
        )

    if collapsed_total:
        # Surfaced, never silent: a reader comparing our numbers to a
        # published run needs to know the haystacks were touched at all.
        print(
            f"note: collapsed {collapsed_total} duplicate session "
            f"entries with identical content",
            file=sys.stderr,
            flush=True,
        )
    questions.collapsed_duplicates = collapsed_total
    return questions


def evaluate(
    questions: Sequence[Question],
    mode: str = "hybrid",
    k_values: Sequence[int] = eval_metrics.DEFAULT_K_VALUES,
    progress: bool = False,
    chunk_words: int | None = None,
    chunk_overlap: int = 0,
) -> dict[str, Any]:
    """Run ``mode`` retrieval over every question and score the results.

    ``chunk_words`` splits each session into overlapping windows before
    embedding, so the vector stream can see past the embedder's ~190-word
    limit. It affects the vector index only — BM25 always reads whole
    documents — and a chunked run is not comparable to an unchunked one, so
    both settings are recorded in the report.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    depth = max([*k_values, eval_metrics.NDCG_K])
    per_question: list[dict[str, Any]] = []
    blank_sessions = 0
    started = time.monotonic()

    for index, question in enumerate(questions, start=1):
        # Context-managed so each question's index is released as soon as it
        # is scored — see EphemeralCorpus.close for why that is not optional.
        with EphemeralCorpus(
            question.documents,
            chunk_words=chunk_words,
            chunk_overlap=chunk_overlap,
        ) as corpus:
            blank_sessions += corpus.blank_documents
            retrieved = corpus.search(question.question, limit=depth, mode=mode)
        scores = eval_metrics.score_question(
            retrieved, question.gold_ids, k_values=k_values, rr_key=f"rr@{depth}"
        )
        per_question.append({
            "question_id": question.question_id,
            "question_type": question.question_type,
            "gold": sorted(question.gold_ids),
            "retrieved": retrieved,
            "haystack_size": len(question.documents),
            **scores,
        })
        if progress and index % 10 == 0:
            elapsed = time.monotonic() - started
            print(
                f"  {index}/{len(questions)} questions  ({elapsed:.0f}s)",
                file=sys.stderr,
                flush=True,
            )

    metric_only = [
        {k: v for k, v in record.items() if isinstance(v, float)}
        for record in per_question
    ]
    return {
        "mode": mode,
        "depth": depth,
        "k_values": list(k_values),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        # Protocol deviations, surfaced so a run is comparable to another.
        "chunk_words": chunk_words,
        "chunk_overlap": chunk_overlap if chunk_words else 0,
        "collapsed_duplicate_sessions": getattr(questions, "collapsed_duplicates", 0),
        "blank_sessions": blank_sessions,
        "summary": eval_metrics.aggregate(metric_only),
        "by_question_type": _summarize_by_type(per_question),
        "per_question": per_question,
    }


def _summarize_by_type(per_question: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Break the summary out per question_type.

    LongMemEval's types differ wildly in difficulty; an aggregate that moves
    without any type moving means the mix changed, not the retrieval.
    """
    buckets: dict[str, list[dict[str, float]]] = {}
    for record in per_question:
        metrics = {k: v for k, v in record.items() if isinstance(v, float)}
        buckets.setdefault(record["question_type"], []).append(metrics)
    return {name: eval_metrics.aggregate(records) for name, records in sorted(buckets.items())}


def format_report(report: dict[str, Any]) -> str:
    """Human-readable summary — the thing you actually read in a terminal."""
    summary = report["summary"]
    lines = [
        "",
        f"LongMemEval-S retrieval — mode={report['mode']}  "
        f"questions={summary.get('questions', 0)}  "
        f"depth={report['depth']}  ({report['elapsed_seconds']}s)"
        + (f"  chunked={report['chunk_words']}w/{report['chunk_overlap']}o"
           if report.get("chunk_words") else ""),
        "=" * 72,
    ]
    for key in sorted(k for k in summary if k != "questions"):
        lines.append(f"  {key:<12} {summary[key]:.4f}")

    if len(report["by_question_type"]) > 1:
        lines.extend(["", "  by question type", "  " + "-" * 68])
        for name, stats in report["by_question_type"].items():
            headline = next((k for k in stats if k.startswith("recall@5")), None)
            value = f"{stats[headline]:.4f}" if headline else "n/a"
            lines.append(f"  {name:<34} n={stats['questions']:<5} recall@5={value}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score memory-compiler retrieval on LongMemEval-S.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to longmemeval_s.json (not vendored — see module docstring).",
    )
    parser.add_argument("--mode", default="hybrid", choices=MODES)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N questions. Use this to size a run first.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(eval_metrics.DEFAULT_K_VALUES),
        help="Recall cutoffs to report (default: 5 10 20).",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=None,
        help=(
            "Split sessions into windows of this many words before embedding. "
            "Lifts vector recall past the embedder's ~190-word limit; try 150."
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=30,
        help="Word overlap between chunks, so a boundary-straddling fact survives.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the full report, including per-question rankings, here.",
    )
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        parser.error(f"dataset not found: {args.dataset}")

    questions = load_questions(args.dataset, limit=args.limit)
    if not questions:
        parser.error("dataset contained no questions")

    print(
        f"Scoring {len(questions)} questions in {args.mode} mode…",
        file=sys.stderr,
        flush=True,
    )
    report = evaluate(
        questions,
        mode=args.mode,
        k_values=tuple(args.k),
        progress=True,
        chunk_words=args.chunk_words,
        chunk_overlap=args.chunk_overlap,
    )
    print(format_report(report))

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Full report written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
