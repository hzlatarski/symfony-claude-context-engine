"""Aggregate retrieval-outcome feedback into knowledge/LESSONS.md.

The companion to ``retrieval_feedback.record`` (graphify's ``reflect``
step). Reads the recency-weighted feedback store and writes a
human-readable, git-diffable digest of which articles have earned trust,
which are dead-ends, and which keep getting *corrected* (the ones that
most need a human to fix or retire).

Zero LLM cost — pure aggregation. Safe to run on every compile or on a
schedule. Idempotent: overwrites LESSONS.md from the current store each
run, so a resolved article silently drops off once its recent outcomes
turn positive (older negatives decay via the 45-day half-life).

Usage:
    uv run python scripts/reflect.py            # write knowledge/LESSONS.md
    uv run python scripts/reflect.py --dry-run  # print, don't write
"""
from __future__ import annotations

import argparse
from datetime import date

import retrieval_feedback
from config import KNOWLEDGE_DIR

LESSONS_FILE = KNOWLEDGE_DIR / "LESSONS.md"

# Score thresholds for bucketing. Neutral is 0.5; feedback pushes away
# from it in either direction.
TRUSTED_THRESHOLD = 0.65
CONTESTED_THRESHOLD = 0.40


def build_lessons(*, today: date | None = None, path=None) -> str:
    """Render LESSONS.md content from the current feedback store."""
    today = today or date.today()
    agg = retrieval_feedback.aggregate(today=today, path=path)

    trusted = [r for r in agg if r["score"] >= TRUSTED_THRESHOLD]
    contested = [r for r in agg if r["score"] < CONTESTED_THRESHOLD]
    # aggregate() sorts ascending by score — trusted needs descending.
    trusted.sort(key=lambda r: (-r["score"], -r["total"], r["slug"]))

    lines = [
        "# Lessons — retrieval-outcome feedback digest",
        "",
        f"> Generated {today.isoformat()} from `retrieval-feedback.json` · "
        f"{len(agg)} rated article(s). Pure aggregation, zero LLM cost.",
        "",
        "Outcomes are recency-weighted (45-day half-life): a fix silently "
        "promotes an article out of *Contested* as its recent ratings turn "
        "positive. This file is regenerated on every run — edit the articles, "
        "not this digest.",
        "",
    ]

    lines.append(f"## ⚠ Contested — review or retire ({len(contested)})")
    lines.append("")
    if contested:
        lines.append("Repeatedly unhelpful or *corrected* when retrieved. "
                     "The `corrected` count is the strongest signal — those "
                     "articles were wrong, not just unhelpful.")
        lines.append("")
        lines.append("| Article | Score | corrected | dead_end | useful | Last note |")
        lines.append("|---|---|---|---|---|---|")
        for r in contested:
            note = (r["last_note"] or "").replace("|", "\\|")[:80]
            lines.append(
                f"| [[{r['slug']}]] | {r['score']:.2f} | {r['corrected']} | "
                f"{r['dead_end']} | {r['useful']} | {note} |"
            )
    else:
        lines.append("_None — no article is currently below the contested threshold._")
    lines.append("")

    lines.append(f"## ✓ Trusted — proven useful when retrieved ({len(trusted)})")
    lines.append("")
    if trusted:
        lines.append("| Article | Score | useful | dead_end | corrected |")
        lines.append("|---|---|---|---|---|")
        for r in trusted:
            lines.append(
                f"| [[{r['slug']}]] | {r['score']:.2f} | {r['useful']} | "
                f"{r['dead_end']} | {r['corrected']} |"
            )
    else:
        lines.append("_None yet — record `useful` outcomes to populate this list._")
    lines.append("")

    return "\n".join(lines)


def write_lessons(*, today: date | None = None) -> tuple[int, int]:
    """Write LESSONS.md. Returns (trusted_count, contested_count)."""
    content = build_lessons(today=today)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    LESSONS_FILE.write_text(content, encoding="utf-8")

    agg = retrieval_feedback.aggregate(today=today)
    trusted = sum(1 for r in agg if r["score"] >= TRUSTED_THRESHOLD)
    contested = sum(1 for r in agg if r["score"] < CONTESTED_THRESHOLD)
    return trusted, contested


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate retrieval-outcome feedback into knowledge/LESSONS.md"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the digest instead of writing it."
    )
    args = parser.parse_args()

    if args.dry_run:
        print(build_lessons())
        return

    trusted, contested = write_lessons()
    print(f"Wrote {LESSONS_FILE}")
    print(f"  Trusted: {trusted} | Contested: {contested}")


if __name__ == "__main__":
    main()
