"""
Ingest static source files into the knowledge base.

Reads sources.yaml, resolves file globs, dispatches to type-specific handlers,
and feeds each source file to the Claude Agent SDK for extraction into
knowledge/concepts/ and knowledge/connections/ articles.

Usage:
    uv run python ingest.py                     # incremental (new/changed only)
    uv run python ingest.py --all               # force re-ingest everything
    uv run python ingest.py --source design-specs  # only one source group
    uv run python ingest.py --dry-run           # show what would happen
    uv run python ingest.py --verbose           # print per-file decisions
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import time

import ingest_state
from config import AGENTS_FILE, CLAUDE_BIN, CONCEPTS_DIR, CONNECTIONS_DIR, KNOWLEDGE_DIR, MODEL_INGEST, NO_WINDOW_CREATIONFLAGS, now_iso
from source_handlers import get_handler
from utils import (
    SourceGroup,
    file_hash,
    list_wiki_articles,
    load_sources_config,
    migrate_state_mutator,
    read_wiki_index,
    resolve_source_files,
    update_state,
)
from compile_truth import compile_truth as regenerate_truth, COMPILED_TRUTH_FILE
import dedup

ROOT_DIR = Path(__file__).resolve().parent.parent


# A file that can never succeed (permanently oversize, unsupported, corrupt)
# would otherwise be retried on every run — and each attempt costs a full CLI
# invocation, up to the 600s timeout. Give up after this many consecutive
# failures; --all still forces a retry.
MAX_INGEST_ATTEMPTS = 3


def source_state_key(group: SourceGroup, file_path: Path) -> str:
    """Build the state.json key for a source file: '{source_id}/{filename}'."""
    return f"{group.id}/{file_path.name}"


def record_failure(state: dict, key: str, current_hash: str) -> None:
    """Count a failed attempt against ``key``.

    Tracked separately from ``ingested_sources`` on purpose: recording a
    success hash is what makes a file invisible, so failures must never be
    written there. The attempt counter resets whenever the source file itself
    changes — editing a document that failed is an implicit request to retry.
    """
    def apply_failure(current: dict) -> None:
        failures = current.setdefault("failed_sources", {})
        prev = failures.get(key, {})
        attempts = (
            prev.get("attempts", 0) + 1
            if prev.get("hash") == current_hash
            else 1
        )
        failures[key] = {
            "hash": current_hash,
            "attempts": attempts,
            "last_failed_at": now_iso(),
        }

    apply_failure(state)
    update_state(apply_failure)


def should_skip_after_failures(state: dict, key: str, current_hash: str) -> bool:
    """True once ``key`` has failed MAX_INGEST_ATTEMPTS times unchanged."""
    entry = state.get("failed_sources", {}).get(key, {})
    return (
        entry.get("hash") == current_hash
        and entry.get("attempts", 0) >= MAX_INGEST_ATTEMPTS
    )


def collect_files_to_ingest(
    groups: list[SourceGroup],
    state: dict,
    force_all: bool = False,
    only_source: str | None = None,
    verbose: bool = False,
) -> list[tuple[SourceGroup, Path]]:
    """Resolve all source files, filter to new/changed ones."""
    ingested_sources = state.get("ingested_sources", {})
    to_ingest: list[tuple[SourceGroup, Path]] = []

    for group in groups:
        if only_source and group.id != only_source:
            continue

        files = resolve_source_files(group)
        if verbose:
            print(f"  {group.id}: {len(files)} files resolved")

        for fpath in files:
            key = source_state_key(group, fpath)
            current_hash = file_hash(fpath)

            if not force_all:
                prev = ingested_sources.get(key, {})
                if prev.get("hash") == current_hash:
                    if verbose:
                        print(f"    SKIP (unchanged): {fpath.name}")
                    continue
                if should_skip_after_failures(state, key, current_hash):
                    print(f"    SKIP ({MAX_INGEST_ATTEMPTS} failed attempts, "
                          f"unchanged since): {fpath.name} — use --all to retry")
                    continue

            to_ingest.append((group, fpath))
            if verbose:
                print(f"    QUEUE: {fpath.name}")

    return to_ingest


def _articles_snapshot() -> dict[str, str]:
    """sha256 of every article's content, keyed by path.

    Content hashes rather than mtimes: an mtime comparison accepts a bare
    ``touch`` or an identical rewrite as proof that knowledge was produced, and
    conversely misses a real same-tick rewrite on a coarse-resolution
    filesystem. Both directions matter here because this snapshot decides
    whether a source file's hash gets recorded — and recording it is what makes
    the file permanently invisible to future runs.
    """
    snapshot: dict[str, str] = {}
    for directory in (CONCEPTS_DIR, CONNECTIONS_DIR):
        if not directory.exists():
            continue
        for path in directory.glob("*.md"):
            try:
                snapshot[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                pass
    return snapshot


def _articles_written_for(before: dict[str, str], rel_source: str) -> bool:
    """True if an article changed *and* credits ``rel_source``.

    "Some article changed" is not enough. The snapshot spans the whole
    knowledge base, so a concurrent ``compile.py``, a second ingest, or the
    user editing an article in an editor would all satisfy it — and this run
    would then record its hash despite having produced nothing, recreating the
    silent permanent data loss this check exists to prevent.

    So we additionally require a changed article to carry this source's
    ``[src:...]`` anchor, which the ingest prompt mandates on every extracted
    claim. That ties the evidence to the file actually being ingested.
    """
    after = _articles_snapshot()
    changed = [p for p, digest in after.items() if before.get(p) != digest]
    if not changed:
        return False

    anchor = f"[src:{rel_source}]"
    for path in changed:
        try:
            if anchor in Path(path).read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False


async def ingest_source_file(
    group: SourceGroup,
    file_path: Path,
    state: dict,
) -> tuple[float, bool]:
    """Ingest a single source file into the knowledge base.

    Returns ``(api_cost, succeeded)``. ``succeeded`` is False whenever no
    article was produced, so the caller can report the failure instead of
    printing "Done." over the top of an error.
    """
    before_snapshot = _articles_snapshot()

    handler = get_handler(group.type)
    doc = handler(file_path)

    schema = AGENTS_FILE.read_text(encoding="utf-8")
    # Compact: the full index is ~530 bytes/article and blew the context window
    # at 651 articles. The dedup pre-flight below carries the detail that
    # matters. See utils.read_wiki_index.
    wiki_index = read_wiki_index(compact=True)

    compiled_truth = ""
    if COMPILED_TRUTH_FILE.exists():
        compiled_truth = COMPILED_TRUTH_FILE.read_text(encoding="utf-8")

    # Duplicate prevention — name the existing articles this source is closest
    # to, so "prefer UPDATE over CREATE" has concrete targets instead of being
    # an instruction to search a 678-line index. See dedup.py.
    preflight = ""
    try:
        preflight = dedup.format_preflight_block(
            dedup.similar_to_text(doc.content, limit=5)
        )
    except Exception as exc:  # never let dedup block an ingest
        print(f"  Duplicate pre-flight failed (continuing): {exc}", file=sys.stderr)

    timestamp = now_iso()
    rel_source = f"sources/{group.id}/{file_path.name}"

    prompt = f"""You are a knowledge compiler. Your job is to read a source document and
extract knowledge into structured wiki articles.

## Schema (AGENTS.md)

{schema}

## Current Wiki Index

{wiki_index}

## Current Knowledge (compiled truth)

{compiled_truth if compiled_truth else "(No compiled truth yet — run compile_truth.py first)"}

{preflight}

## Source Document to Ingest

**File:** {file_path.name}
**Source ID:** {group.id}
**Category:** {group.category}
**Type:** {group.type}

{doc.content}

## Your Task

Read the source document above and compile it into wiki articles following the schema.

This is a **{group.category}** source ({group.description}). Unlike daily conversation
logs, this is a structured document — extract its key concepts, decisions, and
architectural patterns.

### Rules:

1. **Extract key concepts** — Identify 2-5 distinct concepts worth their own article.
   A large design spec may warrant more; a small governance doc may warrant 1-2.
2. **Create concept articles** in `knowledge/concepts/` — One .md file per concept
   - Use the Truth + Timeline format from AGENTS.md:
     * `## Truth` section with TWO subsections:
       - `### Observed` — facts extracted DIRECTLY from the source. Every bullet
         MUST end with at least one `[src:{rel_source}]` anchor pointing to this
         source file. The anchor is mandatory, not optional — without it the
         claim cannot be re-verified by lint.
       - `### Synthesized` — compiler inferences drawn from combining multiple
         Observed facts. Use sparingly and only when the inference is
         non-obvious. Also anchor where possible.
     * `### Related Concepts` subsection (under Truth): [[wikilinks]] with
       one-line descriptions
     * `---` horizontal rule separator
     * `## Timeline` section: per-source entries documenting what was learned and when
   - Include `sources:` in frontmatter pointing to `{rel_source}`
   - Add `source_type: {group.type}` and `source_category: {group.category}` to frontmatter
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
     * `hypothesis` — unvalidated theory awaiting corroboration. Promote to `discovery`
       or `fact` in a later compile when corroborated by additional sources.
     Pick the MOST specific type that fits. When genuinely uncertain, default to `fact`.
     An unknown value here fails lint — use exactly one of the eight above.
   - Write Truth in encyclopedia style — dense, factual, no "we discovered"
3. **Create connection articles** in `knowledge/connections/` if this source reveals
   non-obvious relationships between 2+ existing concepts in the wiki
4. **Update existing articles — with skeptical verification** if this source adds
   new information to concepts already in the wiki:
   - Read the existing article with the Read tool
   - Compare the new info against the article's current Truth section
   - **If the new info CORROBORATES existing Truth:** merge it, bump confidence
     by 0.1 (capped at 1.0), add this source to frontmatter sources
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
5. **Update knowledge/index.md** — Add new entries to the table
   - Each entry: `| [[path/slug]] | One-line summary | {rel_source} | {timestamp[:10]} |`
6. **Append to knowledge/log.md** — Add a timestamped entry:
   ```
   ## [{timestamp}] ingest | {file_path.name}
   - Source: {rel_source}
   - Category: {group.category}
   - Articles created: [[concepts/x]], [[concepts/y]]
   - Articles updated: [[concepts/z]] (if any)
   ```
7. **Use tools to check existing articles** — The wiki index above lists every article
   with its path and one-line summary. Before creating a new article, check the index
   for duplicates or near-duplicates. To update an existing article, use the Read tool
   to fetch it first, then Edit to modify it. Use Grep to search for related concepts
   when adding [[wikilinks]]. Prefer UPDATING an existing article over creating a
   near-duplicate.

### Quality standards:
- Every article must have complete YAML frontmatter (title, aliases, tags, sources,
  source_type, source_category, confidence, created, updated)
- Every article must link to at least 2 other articles via [[wikilinks]]
- Key Points section should have 3-5 bullet points
- Details section should have 2+ paragraphs
- Prefer UPDATING an existing article over creating a near-duplicate
- When multiple source files cover the same topic, the resulting article should
  SYNTHESIZE all of them, not just reflect the latest one

### File paths (use these EXACT paths):
- Write concept articles to: {CONCEPTS_DIR}
- Write connection articles to: {CONNECTIONS_DIR}
- Update index at: {KNOWLEDGE_DIR / 'index.md'}
- Append log at: {KNOWLEDGE_DIR / 'log.md'}
"""

    from config import PROJECT_ROOT

    # Strip ANTHROPIC_API_KEY so claude uses subscription auth, not API credits
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    cmd = [
        CLAUDE_BIN, "-p",
        "--model", MODEL_INGEST,
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
            cwd=str(PROJECT_ROOT),
            env=env,
            creationflags=NO_WINDOW_CREATIONFLAGS,
        )

    try:
        result = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        print(f"  Error: claude CLI timed out after 600s")
        return 0.0, False
    except Exception as e:
        print(f"  Error: {e}")
        return 0.0, False

    # A non-zero exit is a failure even when the CLI explains itself on stdout.
    # This was previously `returncode != 0 AND not stdout.strip()`, so
    # "Prompt is too long" — printed to stdout with exit 1 — fell through to the
    # success path below: the hash was recorded, "Done." was printed, and no
    # article was ever written. Every subsequent run then SKIPPED the file as
    # already-ingested. Silent data loss; found 2026-07-26.
    if result.returncode != 0:
        detail = (result.stderr.strip() or result.stdout.strip())[:300]
        print(f"  Error: claude CLI exited {result.returncode} — {detail}",
              file=sys.stderr)
        return 0.0, False

    # Exit 0 still isn't proof of work. The contract of this function is that
    # the source was turned into at least one article, so verify that actually
    # happened before recording the hash — recording it is what makes the file
    # invisible to future runs.
    if not _articles_written_for(before_snapshot, rel_source):
        print(f"  Error: claude CLI exited 0 but no article credits "
              f"{rel_source} — not marking {file_path.name} as ingested",
              file=sys.stderr)
        return 0.0, False

    key = source_state_key(group, file_path)
    entry = {
        "hash": file_hash(file_path),
        "ingested_at": now_iso(),
        "cost_usd": 0.0,
        "source_id": group.id,
    }
    state["ingested_sources"][key] = entry
    update_state(
        lambda current: current.setdefault("ingested_sources", {}).__setitem__(
            key, entry
        )
    )

    return 0.0, True


def main():
    parser = argparse.ArgumentParser(description="Ingest source files into knowledge base")
    parser.add_argument("--all", action="store_true", help="Force re-ingest all sources")
    parser.add_argument("--source", type=str, help="Only ingest a specific source group by id")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")
    parser.add_argument("--verbose", action="store_true", help="Print per-file decisions")
    args = parser.parse_args()

    state = update_state(migrate_state_mutator)

    groups = load_sources_config()
    if not groups:
        print("No sources.yaml found or no source groups defined.")
        print("Copy sources.yaml.example to sources.yaml and customize it.")
        return

    if args.source:
        valid_ids = [g.id for g in groups]
        if args.source not in valid_ids:
            print(f"Error: source group '{args.source}' not found.")
            print(f"Available: {', '.join(valid_ids)}")
            sys.exit(1)

    to_ingest = collect_files_to_ingest(
        groups, state,
        force_all=args.all,
        only_source=args.source,
        verbose=args.verbose,
    )

    if not to_ingest:
        print("Nothing to ingest — all source files are up to date.")
        return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Files to ingest ({len(to_ingest)}):")
    for group, fpath in to_ingest:
        print(f"  [{group.id}] {fpath.name}")

    if args.dry_run:
        return

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    CONCEPTS_DIR.mkdir(parents=True, exist_ok=True)
    CONNECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Clear any prior stop signal — a fresh run shouldn't honor a stale flag.
    ingest_state.clear_stop()
    started = time.time()
    total = len(to_ingest)
    ingest_state.write_status(
        phase="starting", processed=0, total=total, started_at=started,
    )

    total_cost = 0.0
    stopped = False
    failed: list[str] = []
    for i, (group, fpath) in enumerate(to_ingest, 1):
        if ingest_state.should_stop():
            print(f"\nStop requested — halting after {i - 1}/{total} files.")
            ingest_state.clear_stop()
            stopped = True
            break

        print(f"\n[{i}/{total}] Ingesting [{group.id}] {fpath.name}...")
        ingest_state.write_status(
            phase="running",
            current_file=f"{group.id}/{fpath.name}",
            processed=i - 1,
            total=total,
            total_cost=total_cost,
            started_at=started,
        )
        cost, ok = asyncio.run(ingest_source_file(group, fpath, state))
        total_cost += cost
        key = source_state_key(group, fpath)
        if ok:
            state.get("failed_sources", {}).pop(key, None)
            update_state(
                lambda current: current.setdefault(
                    "failed_sources", {}
                ).pop(key, None)
            )
        else:
            failed.append(key)
            record_failure(state, key, file_hash(fpath))
        ingest_state.write_status(
            phase="running",
            current_file=f"{group.id}/{fpath.name}",
            processed=i,
            total=total,
            total_cost=total_cost,
            started_at=started,
        )
        # Never print "Done." over the top of an error — that is what made the
        # 2026-07-26 failures look like successes in the run output.
        print("  Done." if ok else "  NOT ingested — will retry on the next run.")

    final_phase = "stopped" if stopped else "finished"
    # When stopped at iteration i, files 1..i-1 actually completed; iteration
    # i was aborted before its Sonnet call. When finished cleanly, all
    # `total` files completed.
    processed_count = (i - 1) if stopped else total
    ingest_state.write_status(
        phase=final_phase,
        processed=processed_count,
        total=total,
        total_cost=total_cost,
        started_at=started,
    )

    articles = list_wiki_articles()

    # Incremental vector re-embed of anything the LLM just touched.
    # Hash-based — skips articles whose content didn't change.
    # Failure here is non-fatal: it logs and moves on so a transient
    # Chroma hiccup doesn't mask a successful ingest.
    try:
        from reindex import reindex_articles
        embedded, skipped = reindex_articles(force=False)
        if embedded:
            print(f"Vector index: embedded {embedded} changed articles (skipped {skipped})")
    except Exception as exc:
        print(f"  Vector embed skipped: {exc}", file=sys.stderr)

    regenerate_truth()
    print(f"\nIngestion complete. Total cost: ${total_cost:.2f}")
    if failed:
        # Loud and last: an ingest that produced nothing used to be
        # indistinguishable from a clean run in this summary.
        print(f"\n{len(failed)} file(s) FAILED and were not ingested:")
        for name in failed:
            print(f"  - {name}")
        print("Their hashes were not recorded, so the next run retries them "
              f"(up to {MAX_INGEST_ATTEMPTS} attempts, then --all to force).")
        sys.exit(1)
    print(f"Knowledge base: {len(articles)} articles")


if __name__ == "__main__":
    main()
