"""Self-healing fixers for the lint checks.

Each fix is a pure text transform run locally on one article at a time.
No LLM calls on the fix path — that rule is load-bearing. LLM fixes in
a lint pass would introduce unbounded cost, non-determinism, and a
review nightmare; deterministic transforms are reviewable, reversible,
and testable, so we stay within them.

Five fixers are registered:

- ``missing_backlink`` — append ``[[source]]`` to the target's Related
  Concepts section, creating that section if absent.
- ``broken_link`` — fuzzy-match the broken slug against existing slugs
  via ``difflib.get_close_matches`` with a conservative ratio floor of
  ``BROKEN_LINK_FUZZY_THRESHOLD``. Successful matches rewrite the link;
  failed matches are commented out so lint stops re-flagging them.
- ``stale_article`` — add ``stale: true`` to the source daily file's
  frontmatter sentinel (actually a *pointer* — we mark the concept
  articles that sourced it). Currently implemented as a state.json
  nudge: we simply clear the hash for that daily log so the next
  ``compile.py`` run recompiles it. Safer than silently editing
  articles and matches user intent better.
- ``broken_source_anchor`` — fuzzy-match the anchor path under the
  knowledge tree; else comment out the anchor.
- ``missing_memory_type`` — infer ``type:`` from filename prefix
  (``feedback_*`` → preference, ``project_*`` → project, ``decision_*``
  → decision, ``event_*`` → event, ``discovery_*`` → discovery). Skip
  the article if no confident inference is possible.

Every successful fix is logged through ``_append_audit`` to
``knowledge/log.md`` so weekly reviews can catch mistakes. Fixes that
decline to act (insufficient confidence, missing data) return ``False``
so the caller counts them as unfixed.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Callable

from config import (
    CONCEPTS_DIR,
    CONNECTIONS_DIR,
    DAILY_DIR,
    KNOWLEDGE_DIR,
    LOG_FILE,
    QA_DIR,
    now_iso,
)
from utils import (
    file_hash,
    list_wiki_articles,
    load_state,
    save_state,
)

# Fuzzy-match threshold for broken-link repair. difflib's SequenceMatcher
# ratio ranges 0.0–1.0; 0.9 is empirically strict enough to require
# "same slug minus a typo" rather than "vaguely similar words". Do not
# lower without a trip through the test suite — wrong rewrites corrupt
# the wiki graph silently.
BROKEN_LINK_FUZZY_THRESHOLD = 0.9

# Filename-prefix heuristics for ``missing_memory_type``. The mapping is
# intentionally small — adding entries is cheap, but wrong inferences
# silently misclassify articles into the wrong search bucket. When
# adding a prefix, make sure it's distinctive enough to match exactly
# what the compile prompt produces.
#
# IMPORTANT: ``project_/project-`` is deliberately NOT mapped — those
# articles describe the project (its stack, its overview) and are
# usually ``fact``, not ``event``. Without explicit evidence from the
# compile prompt, the safer move is to leave them alone and let the
# missing_memory_type warning persist as a suggestion.
_MEMORY_TYPE_PREFIXES: dict[str, str] = {
    "feedback_": "preference",
    "feedback-": "preference",
    "decision_": "decision",
    "decision-": "decision",
    "discovery_": "discovery",
    "discovery-": "discovery",
    "event_": "event",
    "event-": "event",
}


# -----------------------------------------------------------------------------
# Audit trail
# -----------------------------------------------------------------------------


def _append_audit(check: str, file: str, change: str) -> None:
    """Append a fix entry to knowledge/log.md.

    The timestamp format matches the ``compile`` and ``ingest`` entries
    so a single grep against the log yields a continuous operation
    history across all writers.
    """
    entry = (
        f"\n## [{now_iso()}] lint-fix | {file}\n"
        f"- Check: {check}\n"
        f"- Change: {change}\n"
    )
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(entry)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _article_path_for_slug(slug: str) -> Path:
    """Resolve a slug like ``concepts/foo`` to its on-disk path."""
    return KNOWLEDGE_DIR / f"{slug}.md"


def _known_article_slugs() -> list[str]:
    """Return every article slug currently on disk (``concepts/...`` etc.)."""
    slugs: list[str] = []
    for article in list_wiki_articles():
        rel = article.relative_to(KNOWLEDGE_DIR)
        slugs.append(str(rel).replace("\\", "/").removesuffix(".md"))
    return slugs


def _fuzzy_match_slug(target: str, candidates: list[str]) -> str | None:
    """Return the best fuzzy match for ``target`` if ratio ≥ threshold."""
    matches = difflib.get_close_matches(
        target, candidates, n=1, cutoff=BROKEN_LINK_FUZZY_THRESHOLD,
    )
    return matches[0] if matches else None


# -----------------------------------------------------------------------------
# Fixers
# -----------------------------------------------------------------------------


def fix_missing_backlink(issue: dict) -> bool:
    """Append the missing backlink to the target article's Related Concepts.

    If the article already has a ``### Related Concepts`` subsection we
    insert the new bullet directly beneath its heading. If it doesn't,
    we create one at the end of the Truth section (before the ``---``
    horizontal rule separator or, lacking one, at end-of-file). The
    inserted bullet uses a neutral description so a human can refine it
    later without the fix clobbering their wording on the next run.
    """
    source_slug = issue.get("source_slug")
    target_slug = issue.get("target_slug")
    if not source_slug or not target_slug:
        return False

    target_path = _article_path_for_slug(target_slug)
    if not target_path.exists():
        return False

    content = target_path.read_text(encoding="utf-8")
    bullet = f"- [[{source_slug}]] - Auto-added backlink"

    # Already present? No-op — idempotent.
    if f"[[{source_slug}]]" in content:
        return False

    if "### Related Concepts" in content:
        # Insert the bullet on the line after the heading's blank line,
        # preserving any existing bullets.
        updated = re.sub(
            r"(### Related Concepts\s*\n\s*\n)",
            rf"\1{bullet}\n",
            content,
            count=1,
        )
    else:
        # No Related Concepts section — create one at the end of Truth.
        # The Truth section ends at a ``---`` horizontal rule (which
        # also separates Truth from Timeline), or at ``## Timeline``,
        # whichever comes first. Critically, we must skip past the
        # YAML frontmatter first — its closing ``---`` would otherwise
        # match our boundary regex and we'd inject the section ABOVE
        # the frontmatter, corrupting the file.
        section = f"\n### Related Concepts\n\n{bullet}\n"
        body_offset = 0
        if content.startswith("---"):
            fm_end = content.find("---", 3)
            if fm_end != -1:
                body_offset = fm_end + 3

        body = content[body_offset:]
        timeline = re.search(r"^## Timeline\s*$", body, re.MULTILINE)
        rule = re.search(r"^---\s*$", body, re.MULTILINE)
        candidates = [m.start() for m in (timeline, rule) if m]
        if candidates:
            idx = body_offset + min(candidates)
            updated = content[:idx] + section + "\n" + content[idx:]
        else:
            updated = content.rstrip() + "\n" + section

    if updated == content:
        return False
    target_path.write_text(updated, encoding="utf-8")
    _append_audit(
        "missing_backlink",
        target_slug,
        f"Added backlink [[{source_slug}]] to Related Concepts",
    )
    return True


def fix_broken_link(issue: dict) -> bool:
    """Rewrite or comment out a broken wikilink.

    Tries to fuzzy-match the broken target against known slugs first.
    On a successful match (ratio ≥ ``BROKEN_LINK_FUZZY_THRESHOLD``) the
    wikilink in the source article is rewritten in place. On failure
    the wikilink is wrapped in an HTML comment so lint stops re-
    flagging the same issue, the original text is preserved verbatim
    for human review, and the audit entry records that no auto-fix was
    possible.

    Re-entrancy guard: if the broken target is already wrapped in a
    ``<!-- BROKEN-LINK: ... -->`` comment from a prior run, the fixer
    refuses to touch it. Without this guard, ``replace`` would match
    the ``[[...]]`` text *inside* the existing comment and wrap it
    again, producing nested comments that grow on every re-run.
    """
    source_file = issue.get("file")
    broken_target = issue.get("broken_target")
    if not source_file or not broken_target:
        return False

    source_path = KNOWLEDGE_DIR / source_file
    if not source_path.exists():
        return False

    content = source_path.read_text(encoding="utf-8")
    old_link = f"[[{broken_target}]]"
    if old_link not in content:
        return False

    # Re-entrancy guard. If every occurrence of the bare link is
    # already inside a BROKEN-LINK wrapper, there is nothing left to
    # fix. We check by counting wrapped vs. total occurrences — if
    # they're equal, all instances are wrapped.
    wrapped_count = content.count(f"<!-- BROKEN-LINK: {old_link}")
    bare_count = content.count(old_link)
    if wrapped_count >= bare_count:
        return False

    candidates = [s for s in _known_article_slugs() if s != broken_target]
    match = _fuzzy_match_slug(broken_target, candidates)

    if match:
        updated = content.replace(old_link, f"[[{match}]]")
        change = f"Rewrote [[{broken_target}]] → [[{match}]] (fuzzy match)"
    else:
        comment = f"<!-- BROKEN-LINK: [[{broken_target}]] -->"
        # Only wrap occurrences that are NOT already wrapped. Replace
        # the whole-content string in one pass would re-wrap; do it
        # surgically by walking and skipping wrapped positions.
        updated = _wrap_only_unwrapped(content, old_link, comment)
        change = f"Commented out broken [[{broken_target}]] (no confident match)"

    if updated == content:
        return False
    source_path.write_text(updated, encoding="utf-8")
    _append_audit("broken_link", source_file, change)
    return True


def _wrap_only_unwrapped(content: str, target: str, wrapped: str) -> str:
    """Replace bare occurrences of ``target`` with ``wrapped``.

    Skips occurrences whose immediately-preceding text is the prefix
    of an HTML wrapper (``<!-- BROKEN-LINK: `` or ``<!-- BROKEN-SRC: ``).
    Without this guard, a previously-wrapped ``[[...]]`` would match
    again on a re-run and produce ``<!-- BROKEN-LINK: <!-- BROKEN-LINK:
    [[...]] --> -->`` nesting.
    """
    out_parts: list[str] = []
    i = 0
    target_len = len(target)
    wrap_prefixes = ("<!-- BROKEN-LINK: ", "<!-- BROKEN-SRC: ")
    while i < len(content):
        idx = content.find(target, i)
        if idx == -1:
            out_parts.append(content[i:])
            break
        # Look back to see if any wrapper prefix immediately precedes.
        already_wrapped = any(
            idx >= len(p) and content[idx - len(p):idx] == p
            for p in wrap_prefixes
        )
        out_parts.append(content[i:idx])
        if already_wrapped:
            out_parts.append(target)  # leave it alone
        else:
            out_parts.append(wrapped)
        i = idx + target_len
    return "".join(out_parts)


def fix_stale_article(issue: dict) -> bool:
    """Clear the cached hash for a stale daily log so compile re-runs it.

    The cleanest way to "fix" a stale article is to re-compile the
    daily log, but compile costs an LLM call which is explicitly
    forbidden on the fix path. Instead we clear the stored hash in
    state.json so the next ``compile.py`` invocation treats the daily
    log as new and picks it up automatically, deferring the actual
    recompile to the correct tool without silently running it here.
    """
    daily_name = issue.get("daily_name")
    if not daily_name:
        return False

    state = load_state()
    ingested = state.get("ingested_daily") or state.get("ingested") or {}
    if daily_name not in ingested:
        return False

    # Clear the hash so next compile detects the file as changed.
    ingested[daily_name]["hash"] = ""
    state["ingested_daily"] = ingested
    save_state(state)

    _append_audit(
        "stale_article",
        f"daily/{daily_name}",
        "Cleared cached hash — next compile.py will re-ingest this daily log",
    )
    return True


def fix_broken_source_anchor(issue: dict) -> bool:
    """Fuzzy-match a broken ``[src:...]`` anchor or comment it out.

    Same fuzzy/fallback strategy as ``fix_broken_link`` but scoped to
    files under the knowledge tree. Candidate pool is the set of all
    files under ``daily/``, ``sources/``, and ``concepts/`` so both
    daily log paths and renamed research files can be rescued.
    """
    source_file = issue.get("file")
    broken_anchor = issue.get("broken_anchor")
    if not source_file or not broken_anchor:
        return False

    source_path = KNOWLEDGE_DIR / source_file
    if not source_path.exists():
        return False

    content = source_path.read_text(encoding="utf-8")
    old_anchor = f"[src:{broken_anchor}]"
    if old_anchor not in content:
        return False

    # Re-entrancy guard — same logic as fix_broken_link. If every
    # occurrence is already wrapped, there is nothing to do.
    wrapped_count = content.count(f"<!-- BROKEN-SRC: {old_anchor}")
    bare_count = content.count(old_anchor)
    if wrapped_count >= bare_count:
        return False

    # Build the candidate pool from existing files on disk.
    candidates: list[str] = []
    for subdir in (DAILY_DIR, CONCEPTS_DIR, CONNECTIONS_DIR, QA_DIR):
        if subdir.exists():
            for f in subdir.rglob("*.md"):
                rel = f.relative_to(KNOWLEDGE_DIR).as_posix()
                candidates.append(rel)

    match = _fuzzy_match_slug(broken_anchor, candidates)
    if match:
        updated = content.replace(old_anchor, f"[src:{match}]")
        change = f"Rewrote [src:{broken_anchor}] → [src:{match}] (fuzzy match)"
    else:
        comment = f"<!-- BROKEN-SRC: [src:{broken_anchor}] -->"
        updated = _wrap_only_unwrapped(content, old_anchor, comment)
        change = f"Commented out broken [src:{broken_anchor}] (no confident match)"

    if updated == content:
        return False
    source_path.write_text(updated, encoding="utf-8")
    _append_audit("broken_source_anchor", source_file, change)
    return True


def fix_missing_memory_type(issue: dict) -> bool:
    """Infer and insert a ``type:`` field from filename prefix heuristics.

    Only fires when the inference is confident (filename matches one of
    ``_MEMORY_TYPE_PREFIXES``). Articles without a matching prefix are
    left alone — missing type is still a suggestion, not an error, so
    declining to act preserves correctness.
    """
    source_file = issue.get("file")
    if not source_file:
        return False

    source_path = KNOWLEDGE_DIR / source_file
    if not source_path.exists():
        return False

    filename = source_path.stem
    inferred: str | None = None
    for prefix, memory_type in _MEMORY_TYPE_PREFIXES.items():
        if filename.startswith(prefix):
            inferred = memory_type
            break
    if inferred is None:
        return False

    content = source_path.read_text(encoding="utf-8")
    # Already has a type field? Leave it alone — the check only fires
    # when type is missing, but double-check to be safe.
    if re.search(r"^type\s*:", content, re.MULTILINE):
        return False

    # Insert after the frontmatter opener.
    fm_end = content.find("---", 3)
    if not content.startswith("---") or fm_end == -1:
        return False

    updated = content[:fm_end] + f"type: {inferred}\n" + content[fm_end:]
    source_path.write_text(updated, encoding="utf-8")
    _append_audit(
        "missing_memory_type",
        source_file,
        f"Inferred type: {inferred} from filename prefix",
    )
    return True


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------


FIX_REGISTRY: dict[str, Callable[[dict], bool]] = {
    "missing_backlink": fix_missing_backlink,
    "broken_link": fix_broken_link,
    "stale_article": fix_stale_article,
    "broken_source_anchor": fix_broken_source_anchor,
    "missing_memory_type": fix_missing_memory_type,
}


def apply_fixes(
    issues: list[dict],
    only_checks: set[str] | None = None,
) -> tuple[int, int]:
    """Run registered fixers against the issue list.

    Returns ``(fixed, attempted)`` so the caller can report both
    numbers. Unknown check names are skipped silently — the registry
    is the source of truth for which checks are fixable at all.

    Pass ``only_checks`` to canary-run a subset of fixers. Useful for
    incremental rollout on large knowledge bases where the operator
    wants to eyeball one category's output before unleashing the rest.
    ``None`` (the default) runs every fixer in the registry.
    """
    fixed = 0
    attempted = 0
    for issue in issues:
        check = issue.get("check", "")
        if only_checks is not None and check not in only_checks:
            continue
        fixer = FIX_REGISTRY.get(check)
        if fixer is None:
            continue
        attempted += 1
        if fixer(issue):
            fixed += 1
    return fixed, attempted
