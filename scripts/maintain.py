"""Safe, non-destructive knowledge-base maintenance — the "maintain" pass.

Mycelium-memory's ``/maintain`` idea, rebuilt as our own MIT code: run the
detectors we already have, then ACT on them in a way that can always be undone.
Nothing here deletes knowledge. Concretely, one dry-run-by-default pass that:

* **retires near-duplicates** — for each near-duplicate pair (``dedup``), marks
  the *weaker* twin **cold** (``cold.py`` — reversible), never the stronger one,
  and never deletes either. A human later merges and removes by hand.
* **prunes ghost vectors** — embeddings whose article file is gone
  (``dedup.prune_stale_vectors``); reversible by re-indexing.
* **prunes faded co-access edges** — usage links decayed below the floor
  (``coaccess.prune``); purely derived data, safe to drop.
* **rescues orphans** (opt-in, ``--crosslink``) — backfills ``[[wikilinks]]``
  for unlinked title mentions (``crosslink``) so orphaned articles rejoin the
  graph. OFF by default: it is the one step that REWRITES article files, so it
  is kept out of the safe core — the default pass (cold + prune) never edits an
  article body, which is what lets us promise the default ``--apply`` loses no
  content.

Dry-run is the default (matches ``crosslink``/``dedup``): it prints the plan and
changes nothing. ``--apply`` executes it. Every cold mark is audit-logged by
``cold.py``; the whole run is also summarized to ``reports/``.

The weaker-twin choice is the one real judgement, so ``pick_weaker`` is a pure,
injectable, unit-tested function: a pinned article always outranks an unpinned
one (and two pinned twins are left alone for a human); otherwise lower
retrieval-feedback score loses, then older ``updated`` date, then shorter body,
then a deterministic slug tiebreak. It errs toward retiring the article the
system has least evidence for.
"""
from __future__ import annotations

import argparse
import functools
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts import config
except ImportError:  # pragma: no cover - CLI entrypoint
    import config


@dataclass(frozen=True)
class RetireAction:
    """A proposed cold-retirement of the weaker twin of a near-duplicate pair."""

    weaker: str
    stronger: str
    similarity: float
    reason: str


def _article_file(slug: str, knowledge_dir: Path) -> Path:
    return knowledge_dir / f"{slug}.md"


def _frontmatter(slug: str, knowledge_dir: Path) -> dict:
    """Parse one article's YAML frontmatter; empty dict if missing/unreadable."""
    path = _article_file(slug, knowledge_dir)
    if not path.exists():
        return {}
    try:
        from compile_truth import parse_frontmatter
    except ImportError:  # pragma: no cover
        from scripts.compile_truth import parse_frontmatter
    try:
        return parse_frontmatter(path.read_text(encoding="utf-8")) or {}
    except OSError:  # pragma: no cover
        return {}


def _body_len(slug: str, knowledge_dir: Path) -> int:
    path = _article_file(slug, knowledge_dir)
    try:
        return len(path.read_text(encoding="utf-8"))
    except OSError:  # pragma: no cover
        return 0


def pick_weaker(
    slug_a: str,
    slug_b: str,
    *,
    knowledge_dir: Path,
    feedback_scores: dict[str, float],
) -> tuple[str | None, str | None]:
    """Choose which twin to retire. Returns ``(weaker, stronger)``.

    Returns ``(None, None)`` when the choice must be left to a human — i.e.
    BOTH articles are pinned (retiring a pinned article silently is never
    acceptable). The ordering of tiebreakers, strongest signal first:

    1. **Pinned beats unpinned.** A pinned twin is always the stronger.
    2. **Retrieval-feedback score** — the twin the agent has found *less*
       useful (lower score, neutral 0.5 when unrated) is weaker.
    3. **Age** — the older ``updated``/``created`` twin is weaker.
    4. **Body length** — the shorter twin is weaker (less content to lose).
    5. **Slug** — deterministic final tiebreak (retire the lexicographically
       later slug) so the pass is reproducible.
    """
    fa = _frontmatter(slug_a, knowledge_dir)
    fb = _frontmatter(slug_b, knowledge_dir)
    pinned_a = bool(fa.get("pinned"))
    pinned_b = bool(fb.get("pinned"))
    if pinned_a and pinned_b:
        return None, None
    if pinned_a != pinned_b:
        return (slug_b, slug_a) if pinned_a else (slug_a, slug_b)

    score_a = feedback_scores.get(slug_a, 0.5)
    score_b = feedback_scores.get(slug_b, 0.5)
    if score_a != score_b:
        return (slug_a, slug_b) if score_a < score_b else (slug_b, slug_a)

    date_a = str(fa.get("updated") or fa.get("created") or "")
    date_b = str(fb.get("updated") or fb.get("created") or "")
    if date_a != date_b:
        return (slug_a, slug_b) if date_a < date_b else (slug_b, slug_a)

    len_a = _body_len(slug_a, knowledge_dir)
    len_b = _body_len(slug_b, knowledge_dir)
    if len_a != len_b:
        return (slug_a, slug_b) if len_a < len_b else (slug_b, slug_a)

    # Fully tied — retire the lexicographically later slug, deterministically.
    hi, lo = max(slug_a, slug_b), min(slug_a, slug_b)
    return hi, lo


def _is_pinned(slug: str, knowledge_dir: Path) -> bool:
    return bool(_frontmatter(slug, knowledge_dir).get("pinned"))


def _components(pairs: list) -> list[set[str]]:
    """Connected components of the near-duplicate graph (union-find).

    Near-duplicate is transitive-ish: A~B and B~C put A, B and C in one
    cluster even if A~C was below the reporting floor. Retiring per-pair would
    mark an article both 'keep' (vs a weaker twin) and 'cold' (vs a stronger
    one); grouping into clusters lets us keep exactly ONE survivor per cluster.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path-compress
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in pairs:
        union(p.slug_a, p.slug_b)

    comps: dict[str, set[str]] = {}
    for slug in list(parent):
        comps.setdefault(find(slug), set()).add(slug)
    return list(comps.values())


def _cmp_strength(x: str, y: str, knowledge_dir: Path, feedback_scores: dict[str, float]) -> int:
    """Comparator making the STRONGER article sort first (ascending).

    Delegates to ``pick_weaker`` so the cluster winner and the pairwise choice
    share one tested definition of strength. Both-pinned (``None``) can only
    occur inside a cluster already excluded from retirement, so a stable
    slug order there is harmless.
    """
    if x == y:
        return 0  # reflexive: a consistent comparator must call equals equal
    weaker, stronger = pick_weaker(
        x, y, knowledge_dir=knowledge_dir, feedback_scores=feedback_scores
    )
    if weaker is None:
        return (x > y) - (x < y)
    return -1 if stronger == x else 1


def plan_retirements(
    pairs: list,
    *,
    knowledge_dir: Path,
    feedback_scores: dict[str, float],
    already_cold: set[str],
) -> tuple[list[RetireAction], list[tuple[str, str, float]]]:
    """Turn near-duplicate pairs into retire actions + a skipped list.

    Groups the pairs into near-duplicate CLUSTERS and keeps exactly one
    survivor per cluster (the strongest by ``pick_weaker``); every other
    cluster member is retired to cold, each referencing that single survivor —
    so no article is ever both kept and retired. A cluster with two or more
    pinned members is left entirely to a human. An already-cold loser, or a
    pinned loser, is reported in ``skipped`` rather than acted on.

    ``pairs`` are ``dedup.DuplicatePair`` (``slug_a``, ``slug_b``,
    ``similarity``). Returns ``(actions, skipped)``; each skipped entry is
    ``(slug, other, similarity)``.
    """
    # Max similarity incident to each slug, for human-readable reporting.
    max_sim: dict[str, float] = {}
    for p in pairs:
        max_sim[p.slug_a] = max(max_sim.get(p.slug_a, 0.0), p.similarity)
        max_sim[p.slug_b] = max(max_sim.get(p.slug_b, 0.0), p.similarity)

    actions: list[RetireAction] = []
    skipped: list[tuple[str, str, float]] = []
    key = functools.cmp_to_key(
        lambda a, b: _cmp_strength(a, b, knowledge_dir, feedback_scores)
    )

    for comp in _components(pairs):
        pinned = [s for s in comp if _is_pinned(s, knowledge_dir)]
        if len(pinned) >= 2:
            # Ambiguous: two protected twins. Report every in-cluster pair.
            for p in pairs:
                if p.slug_a in comp and p.slug_b in comp:
                    skipped.append((p.slug_a, p.slug_b, p.similarity))
            continue

        ordered = sorted(comp, key=key)  # strongest first
        # The survivor must be an EXISTING, HOT article. Choosing the strongest
        # member blindly can leave a cluster whose only survivor is already cold
        # — retiring the last hot copy. Pick the strongest member that is not
        # already cold and still on disk; if none qualifies, the cluster is
        # already fully retired and there is nothing safe to do.
        winner = next(
            (
                s for s in ordered
                if s not in already_cold and _article_file(s, knowledge_dir).exists()
            ),
            None,
        )
        if winner is None:
            continue
        for loser in ordered:
            if loser == winner:
                continue
            sim = max_sim.get(loser, 0.0)
            if loser in already_cold or _is_pinned(loser, knowledge_dir):
                skipped.append((loser, winner, sim))
                continue
            actions.append(RetireAction(
                weaker=loser,
                stronger=winner,
                similarity=sim,
                reason=f"near-duplicate of {winner} (cosine {sim:.3f})",
            ))
    return actions, skipped


def run(apply: bool = False, *, with_crosslink: bool = False) -> int:
    """Run the maintenance pass. Dry-run unless ``apply`` is True."""
    import dedup
    import cold
    import coaccess
    import retrieval_feedback

    knowledge_dir = config.KNOWLEDGE_DIR
    already_cold = cold.load_cold()
    feedback_scores = retrieval_feedback.load_scores()

    stale = dedup.find_stale_vectors()
    pairs = dedup.find_near_duplicates()
    actions, skipped = plan_retirements(
        pairs,
        knowledge_dir=knowledge_dir,
        feedback_scores=feedback_scores,
        already_cold=already_cold,
    )

    mode = "APPLY" if apply else "DRY-RUN (use --apply to act)"
    print(f"=== maintain [{mode}] ===\n")

    print(f"Near-duplicate retirements (mark weaker twin cold): {len(actions)}")
    for a in actions:
        print(f"  cold <- {a.weaker}   (keep {a.stronger}, cosine {a.similarity:.3f})")
    if skipped:
        print(f"\nSkipped for human review (both pinned, or weaker already cold): {len(skipped)}")
        for sa, sb, sim in skipped:
            print(f"  {sa}  <->  {sb}  (cosine {sim:.3f})")

    print(f"\nStale ghost vectors to prune: {len(stale)}")
    for slug in stale:
        print(f"  prune vector <- {slug}")

    if apply:
        retired = 0
        for a in actions:
            if cold.mark_cold(a.weaker, reason=a.reason, by="maintain"):
                retired += 1
        pruned = dedup.prune_stale_vectors()
        faded = coaccess.prune()
        crosslinked = 0
        if with_crosslink:
            import crosslink
            crosslinked = crosslink.run(knowledge_dir, apply=True)
        print(
            f"\nAPPLIED: retired {retired} article(s) to cold, pruned "
            f"{len(pruned)} ghost vector(s), dropped {faded} faded co-access "
            f"edge(s), crosslink rc={crosslinked}."
        )
    else:
        faded = len(coaccess.load_edges(min_weight=0.0)) - len(
            coaccess.load_edges()
        )
        print(
            f"\nNothing changed (dry-run). Re-run with --apply to retire "
            f"{len(actions)} duplicate(s), prune {len(stale)} ghost vector(s), "
            f"and drop ~{max(faded, 0)} faded co-access edge(s)."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safe, reversible knowledge-base maintenance (dry-run by default).",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Execute the plan (default: dry-run preview only).",
    )
    parser.add_argument(
        "--crosslink", action="store_true",
        help="Also run the orphan-rescue crosslink backfill. OFF by default — "
             "it is the only step that rewrites article files.",
    )
    args = parser.parse_args(argv)
    return run(apply=args.apply, with_crosslink=args.crosslink)


if __name__ == "__main__":
    sys.exit(main())
