"""Retrieval-quality metrics for the benchmark harness.

Pure functions over ranked id lists. Deliberately free of any index or
dataset coupling so the same scoring code can grade LongMemEval, an
in-corpus self-eval, or a hand-built regression fixture.

Metric choices follow the LongMemEval retrieval protocol:

* ``recall_any@k`` — does *any* gold session appear in the top ``k``?
  Binary per question, averaged across questions. This is the headline
  number: for question answering it does not matter whether we surfaced
  all five relevant sessions, only whether the answer was reachable.
* ``ndcg@k`` — rewards putting the gold sessions *high*, not merely
  inside the cutoff. Binary relevance (a session is gold or it isn't),
  so gain is 1 and the discount does all the work.
* ``rr`` / MRR — reciprocal of the first gold rank, computed over the
  **full** ranked list rather than a cutoff, per convention.

All three are cheap; the expensive half of a benchmark run is building
the index, so we compute every metric on every question rather than
making the caller choose.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

# Cutoffs reported by default. 5 is the headline (it is what fits in a
# context injection budget); 10 and 20 show whether misses are near-misses.
DEFAULT_K_VALUES: tuple[int, ...] = (5, 10, 20)

# NDCG is reported at a single cutoff — 10 is the LongMemEval convention.
NDCG_K = 10


def _validate_gold(gold_ids: Iterable[str]) -> set[str]:
    """Return ``gold_ids`` as a set, rejecting an empty one.

    A question with no gold session cannot be scored: returning 0.0 would
    silently depress the mean and look like a retrieval failure, which is
    exactly the kind of quiet corruption a benchmark must not have.
    """
    gold = set(gold_ids)
    if not gold:
        raise ValueError("gold_ids must not be empty — a question needs at least one gold id")
    return gold


def _validate_k(k: int) -> int:
    """Reject non-positive cutoffs.

    ``ranked_ids[:0]`` is empty and ``ranked_ids[:-3]`` quietly means "all but
    the last three" — both would score something plausible for a nonsense
    cutoff instead of failing.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return k


def recall_any_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """1.0 if any gold id appears in the first ``k`` results, else 0.0."""
    gold = _validate_gold(gold_ids)
    _validate_k(k)
    return 1.0 if any(rid in gold for rid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], gold_ids: Iterable[str]) -> float:
    """Reciprocal of the 1-based rank of the earliest gold id, else 0.0.

    Scans the whole ranking — MRR is not a cutoff metric.
    """
    gold = _validate_gold(gold_ids)
    for rank, rid in enumerate(ranked_ids, start=1):
        if rid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """Normalized discounted cumulative gain at ``k`` under binary relevance.

    The ideal ranking is capped at ``k`` gold documents, so a run that
    surfaces as many gold items as the cutoff allows scores 1.0 even when
    more gold exists than could possibly fit.
    """
    gold = _validate_gold(gold_ids)
    _validate_k(k)

    dcg = 0.0
    credited: set[str] = set()
    for rank, rid in enumerate(ranked_ids[:k], start=1):
        # Credit each gold id once. A malformed ranking that repeats a gold
        # document would otherwise accumulate gain per occurrence and score
        # above 1.0 — a "better than perfect" result that hides the defect.
        if rid in gold and rid not in credited:
            credited.add(rid)
            dcg += 1.0 / math.log2(rank + 1)

    reachable = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, reachable + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def score_question(
    ranked_ids: Sequence[str],
    gold_ids: Iterable[str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    ndcg_k: int = NDCG_K,
    rr_key: str = "rr",
) -> dict[str, float]:
    """Score one question across every configured cutoff.

    ``rr_key`` lets the caller label reciprocal rank with the depth its
    ranking was truncated to (``rr@20``). Reciprocal rank over a truncated
    list is not MRR, and naming it plainly ``rr`` invites comparison with
    published full-ranking MRR figures.
    """
    gold = _validate_gold(gold_ids)
    scores: dict[str, float] = {
        f"recall@{k}": recall_any_at_k(ranked_ids, gold, k) for k in k_values
    }
    scores[f"ndcg@{ndcg_k}"] = ndcg_at_k(ranked_ids, gold, ndcg_k)
    scores[rr_key] = reciprocal_rank(ranked_ids, gold)
    return scores


def aggregate(per_question: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean each metric across questions, plus the question count.

    Every question must carry the *same* metric keys. Averaging over
    whichever questions happen to have a key divides by a smaller
    denominator than the reported ``questions`` count, so a partial,
    resumed, or hand-assembled run would look complete — and better than it
    was. A ragged input is a bug in the caller, not something to average
    around.
    """
    if not per_question:
        return {"questions": 0}

    expected = set(per_question[0])
    for index, scores in enumerate(per_question[1:], start=1):
        if set(scores) != expected:
            missing = sorted(expected - set(scores))
            extra = sorted(set(scores) - expected)
            raise ValueError(
                f"every question must report the same metrics; question {index} "
                f"missing={missing} unexpected={extra}"
            )

    totals: dict[str, float] = {}
    for scores in per_question:
        for key, value in scores.items():
            totals[key] = totals.get(key, 0.0) + float(value)

    count = len(per_question)
    summary: dict[str, Any] = {key: total / count for key, total in totals.items()}
    summary["questions"] = count
    return summary
