"""
Compile daily conversation logs into structured knowledge articles.

This is the "LLM compiler" - it reads daily logs (source code) and produces
organized knowledge articles (the executable).

Usage:
    uv run python compile.py                    # compile new/changed logs only
    uv run python compile.py --all              # force recompile everything
    uv run python compile.py --file daily/2026-04-01.md  # compile a specific log
    uv run python compile.py --dry-run          # show what would be compiled
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from filelock import FileLock

from config import AGENTS_FILE, CLAUDE_BIN, CONCEPTS_DIR, CONNECTIONS_DIR, DAILY_DIR, KNOWLEDGE_DIR, MODEL_COMPILE, NO_WINDOW_CREATIONFLAGS, now_iso
from utils import (
    file_hash,
    list_raw_files,
    list_wiki_articles,
    migrate_state_schema,
    read_wiki_index,
    update_state,
)
from compile_truth import compile_truth as regenerate_truth, COMPILED_TRUTH_FILE
import dedup


# ── Paths for the LLM to use ──────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent


async def compile_daily_log(log_path: Path, state: dict) -> bool:
    """Compile a single daily log into knowledge articles.

    Returns whether the external compile completed and produced credited output.
    """
    log_content = log_path.read_text(encoding="utf-8")
    schema = AGENTS_FILE.read_text(encoding="utf-8")
    wiki_index = read_wiki_index(compact=True)

    # Read compiled truth — dense summary of all current knowledge (zero-cost artifact)
    compiled_truth = ""
    if COMPILED_TRUTH_FILE.exists():
        compiled_truth = COMPILED_TRUTH_FILE.read_text(encoding="utf-8")

    # Duplicate prevention. Telling the model to "check the index for
    # near-duplicates" against a 678-line index does not work — the KB has
    # byte-identical articles under two slugs to prove it. Naming the specific
    # articles this source is semantically closest to gives it something it can
    # actually act on. Zero cost: local embeddings, no LLM, no network.
    preflight = ""
    try:
        preflight = dedup.format_preflight_block(
            dedup.similar_to_text(log_content, limit=5)
        )
    except Exception as exc:  # never let dedup block a compile
        print(f"  Duplicate pre-flight failed (continuing): {exc}", file=sys.stderr)

    timestamp = now_iso()
    before = _articles_snapshot()

    prompt = f"""You are a knowledge compiler. Your job is to read a daily conversation log
and extract knowledge into structured wiki articles.

## Schema (AGENTS.md)

{schema}

## Current Wiki Index

{wiki_index}

## Current Knowledge (compiled truth)

{compiled_truth if compiled_truth else "(No compiled truth yet — run compile_truth.py first)"}

{preflight}

## Daily Log to Compile

**File:** {log_path.name}

{log_content}

## Your Task

Read the daily log above and compile it into wiki articles following the schema exactly.

### Rules:

1. **Extract key concepts** - Identify 3-7 distinct concepts worth their own article
2. **Create concept articles** in `knowledge/concepts/` - One .md file per concept
   - Use the Truth + Timeline format from AGENTS.md:
     * `## Truth` section with TWO subsections:
       - `### Observed` — facts extracted DIRECTLY from the source. Every bullet
         MUST end with at least one `[src:daily/YYYY-MM-DD.md]` anchor pointing
         to the daily log the fact came from. The anchor is mandatory, not
         optional — without it the claim cannot be re-verified by lint.
       - `### Synthesized` — compiler inferences drawn from combining multiple
         Observed facts. Use sparingly and only when the inference is
         non-obvious. Also anchor where possible.
     * `### Related Concepts` subsection (under Truth): [[wikilinks]] with
       one-line descriptions
     * `---` horizontal rule separator
     * `## Timeline` section: per-source entries documenting what was learned and when
   - Include `sources:` in frontmatter pointing to the daily log file
   - Include `confidence:` in frontmatter (0.0-1.0) based on how validated the info is:
     * 0.8-1.0: Firm decisions, verified facts, implemented and tested
     * 0.6-0.8: Patterns observed multiple times, strong consensus
     * 0.4-0.6: Single observation, reasonable but unverified
     * 0.2-0.4: Speculation, tentative plans, may change
   - Include `type:` in frontmatter — exactly one of the eight canonical values:
     * `fact` — static knowledge not bound to a moment ("Stimulus uses kebab-case")
     * `event` — something that happened on a date ("launched onboarding v2")
     * `discovery` — finding from debugging/investigation ("root cause: missing index on sessions.user_id")
     * `preference` — user preference or stylistic rule ("don't mock the database in tests")
     * `advice` — actionable guidance for future sessions ("run Tailwind rebuild after editing app.css")
     * `decision` — locked-in architectural choice ("using framework X, not framework Y")
     * `tension` — known unresolved architectural conflict actively being worked through
       (e.g. "should we keep both Stripe and PayPal or consolidate?"). Distinct from
       contradictions, which are accidentally inconsistent claims auto-detected by lint.
     * `hypothesis` — unvalidated theory awaiting corroboration ("we suspect the slow
       query comes from missing index on user_id"). Promote to `discovery` or `fact`
       in a later compile when corroborated.
     Pick the MOST specific type that fits. When genuinely uncertain, default to `fact`.
     An unknown value here fails lint — use exactly one of the eight above.
   - Write Truth in encyclopedia style — dense, factual, no "we discovered"
3. **Create connection articles** in `knowledge/connections/` if this log reveals non-obvious
   relationships between 2+ existing concepts
4. **Update existing articles — with skeptical verification** if this log adds
   new information to concepts already in the wiki:
   - Read the existing article with the Read tool
   - Compare the new info against the article's current Truth section
   - **If the new info CORROBORATES existing Truth:** merge it, bump confidence
     by 0.1 (capped at 1.0), add the daily log to frontmatter sources
   - **If the new info CONTRADICTS existing Truth:** do NOT silently overwrite.
     Instead:
       (a) Keep the existing Truth unchanged
       (b) Append a `### Conflict <YYYY-MM-DD>` subsection to Timeline documenting
           both the old and new claim with evidence from each source
       (c) Emit a line in your final response starting with `CONTRADICTION: `
           followed by `[concepts/<slug>]` or `[connections/<slug>]` or `[qa/<slug>]`
           and a one-line description, using the exact bracket format so that
           lint.py can parse it
       (d) LOWER confidence by 0.1 (floor 0.1) to reflect the unresolved conflict
   - **If the new info EXTENDS existing Truth** (adds non-conflicting detail):
     merge it into Key Points without changing confidence
   - Never overwrite a factual claim in Truth without running the above check
5. **Update knowledge/index.md** - Add new entries to the table
   - Each entry: `| [[path/slug]] | One-line summary | source-file | {timestamp[:10]} |`
6. **Append to knowledge/log.md** - Add a timestamped entry:
   ```
   ## [{timestamp}] compile | {log_path.name}
   - Source: daily/{log_path.name}
   - Articles created: [[concepts/x]], [[concepts/y]]
   - Articles updated: [[concepts/z]] (if any)
   ```
7. **Use tools to check existing articles** — The wiki index above lists every article
   with its path and one-line summary. Before creating a new article, check the index
   for duplicates or near-duplicates. To update an existing article, use the Read tool
   to fetch it first, then Edit to modify it. Use Grep to search for related concepts
   when adding [[wikilinks]].

### File paths:
- Write concept articles to: {CONCEPTS_DIR}
- Write connection articles to: {CONNECTIONS_DIR}
- Update index at: {KNOWLEDGE_DIR / 'index.md'}
- Append log at: {KNOWLEDGE_DIR / 'log.md'}

### Quality standards:
- Every article must have complete YAML frontmatter
- Every article must link to at least 2 other articles via [[wikilinks]]
- Key Points section should have 3-5 bullet points
- Details section should have 2+ paragraphs
- Related Concepts section should have 2+ entries
- Sources section should cite the daily log with specific claims extracted
"""

    project_root = str(Path(__file__).resolve().parent.parent.parent.parent)

    # Strip ANTHROPIC_API_KEY so claude uses subscription auth, not API credits
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    cmd = [
        CLAUDE_BIN, "-p",
        "--model", MODEL_COMPILE,
        "--no-session-persistence",
        "--output-format", "text",
        "--max-turns", "30",
        "--dangerously-skip-permissions",
    ]

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
            cwd=project_root,
            env=env,
            creationflags=NO_WINDOW_CREATIONFLAGS,
        )

    try:
        result = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        print(f"  Error: claude CLI timed out after 600s")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

    if result.returncode != 0:
        detail = (result.stderr.strip() or result.stdout.strip())[:300]
        print(
            f"  Error: claude CLI exited {result.returncode} — {detail}",
            file=sys.stderr,
        )
        return False

    rel_source = f"daily/{log_path.name}"
    if not _articles_written_for(before, rel_source):
        print(
            f"  Error: claude CLI exited 0 but wrote no article credited to "
            f"{rel_source}",
            file=sys.stderr,
        )
        return False

    # Update state (cost is 0.0 — subscription billing, not API credits)
    rel_path = log_path.name
    entry = {
        "hash": file_hash(log_path),
        "compiled_at": now_iso(),
        "cost_usd": 0.0,
    }
    state.setdefault("ingested_daily", {})[rel_path] = entry
    update_state(
        lambda current: current.setdefault("ingested_daily", {}).__setitem__(
            rel_path, entry
        )
    )

    return True


def _articles_snapshot() -> dict[str, str]:
    """Return content hashes for compiler-managed knowledge articles."""
    snapshot: dict[str, str] = {}
    for directory in (CONCEPTS_DIR, CONNECTIONS_DIR):
        if not directory.exists():
            continue
        for path in directory.rglob("*.md"):
            try:
                snapshot[str(path.resolve())] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            except OSError:
                continue
    return snapshot


def _articles_written_for(before: dict[str, str], rel_source: str) -> bool:
    """Return whether a changed article explicitly credits this daily source."""
    for directory in (CONCEPTS_DIR, CONNECTIONS_DIR):
        if not directory.exists():
            continue
        for path in directory.rglob("*.md"):
            try:
                content = path.read_text(encoding="utf-8")
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            except OSError:
                continue
            if before.get(str(path.resolve())) == digest:
                continue
            if f"[src:{rel_source}]" in content:
                return True
    return False


async def compile_logs(logs: list[Path], state: dict) -> list[bool]:
    """Compile logs oldest-first with exactly one knowledge writer."""
    results: list[bool] = []
    for log in sorted(logs, key=lambda path: path.name):
        results.append(await compile_daily_log(log, state))
    return results


def _main_unlocked(args: argparse.Namespace) -> int:
    def migrate(current: dict) -> None:
        migrated = migrate_state_schema(current)
        current.clear()
        current.update(migrated)

    state = update_state(migrate)

    # Determine which files to compile
    if args.file:
        target = Path(args.file)
        if not target.is_absolute():
            target = DAILY_DIR / target.name
        if not target.exists():
            # Try resolving relative to project root
            target = ROOT_DIR / args.file
        if not target.exists():
            print(f"Error: {args.file} not found")
            return 2
        to_compile = [target]
    else:
        all_logs = list_raw_files()
        if args.all:
            to_compile = all_logs
        else:
            to_compile = []
            for log_path in all_logs:
                rel = log_path.name
                prev = state.get("ingested_daily", {}).get(rel, {})
                if not prev or prev.get("hash") != file_hash(log_path):
                    to_compile.append(log_path)

    if not to_compile:
        print("Nothing to compile - all daily logs are up to date.")
        return 0

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Files to compile ({len(to_compile)}):")
    for f in to_compile:
        print(f"  - {f.name}")

    if args.dry_run:
        return 0

    print(f"\nCompiling {len(to_compile)} logs oldest-first...")
    results = asyncio.run(compile_logs(to_compile, state))
    failed_count = results.count(False)
    succeeded_count = results.count(True)
    print(f"  Completed {succeeded_count}/{len(results)} daily logs.")
    
    articles = list_wiki_articles()

    # Incremental vector re-embed of anything the LLM just touched.
    # Hash-based — skips articles whose content didn't change.
    # Failure here is non-fatal: it logs and moves on so a transient
    # Chroma hiccup doesn't mask a successful compile.
    try:
        from reindex import reindex_articles
        embedded, skipped = reindex_articles(force=False)
        if embedded:
            print(f"Vector index: embedded {embedded} changed articles (skipped {skipped})")
    except Exception as exc:
        print(f"  Vector embed skipped: {exc}", file=sys.stderr)

    regenerate_truth()
    print("\nCompilation pass complete. Total cost: $0.00")
    print(f"Knowledge base: {len(articles)} articles")
    if failed_count:
        print(
            f"{failed_count} daily log(s) failed and remain eligible for retry.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile daily logs into knowledge articles")
    parser.add_argument("--all", action="store_true", help="Force recompile all logs")
    parser.add_argument("--file", type=str, help="Compile a specific daily log file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be compiled")
    args = parser.parse_args()

    lock_path = ROOT_DIR / "scripts" / "compile.lock"
    with FileLock(str(lock_path)):
        return _main_unlocked(args)



if __name__ == "__main__":
    raise SystemExit(main())
