"""Shared utilities for the personal knowledge base."""

import hashlib
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable

from filelock import FileLock

from config import (
    CONCEPTS_DIR,
    CONNECTIONS_DIR,
    DAILY_DIR,
    INDEX_FILE,
    KNOWLEDGE_DIR,
    LOG_FILE,
    QA_DIR,
    SOURCES_FILE,
    STATE_FILE,
)


# ── State management ──────────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "ingested_daily": {},
        "ingested_sources": {},
        "access_counts": {},
        "query_count": 0,
        "last_lint": None,
        "total_cost": 0.0,
    }


def load_state(state_file: Path | None = None) -> dict:
    """Load persistent state from state.json, normalized against the defaults.

    Every default key is guaranteed present. Without this, a state.json that
    has lost a key — which happened on 2026-07-30, when the live file came
    back missing ``ingested_sources`` — hands writers a dict they index
    directly, and ``state["ingested_sources"][key] = entry`` raises KeyError
    only *after* a full ingest has re-processed every source file. Existing
    values and unknown keys are left untouched.
    """
    path = state_file or STATE_FILE
    if not path.exists():
        return _default_state()

    state = json.loads(path.read_text(encoding="utf-8"))
    backfilled = []
    for key, default in _default_state().items():
        if key not in state:
            state[key] = default
            backfilled.append(key)

    if backfilled:
        # Warn rather than fail. Failing is what cost a completed ingest run;
        # staying silent is what let 496 records disappear unnoticed for days.
        print(
            f"warning: {path.name} was missing {', '.join(sorted(backfilled))} — "
            f"backfilled with empty defaults. If this is not a fresh install, "
            f"the file lost data and a backup should be reconciled.",
            file=sys.stderr,
            flush=True,
        )
    return state


def _merge_state(current: dict, incoming: dict) -> dict:
    """Recursively merge a stale snapshot without discarding current mappings."""
    merged = dict(current)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_state(merged[key], value)
        else:
            merged[key] = value
    return merged


def _atomic_write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(state, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def update_state(
    mutator: Callable[[dict], object],
    state_file: Path | None = None,
) -> dict:
    """Atomically mutate current state under a cross-process lock."""
    path = state_file or STATE_FILE
    lock = FileLock(str(path.with_suffix(f"{path.suffix}.lock")))
    with lock:
        state = load_state(path)
        mutator(state)
        _atomic_write_state(path, state)
        return state


def save_state(state: dict, state_file: Path | None = None) -> None:
    """Merge and atomically save a possibly stale state snapshot."""
    path = state_file or STATE_FILE

    def merge(current: dict) -> None:
        merged = _merge_state(current, state)
        current.clear()
        current.update(merged)

    update_state(merge, path)


# ── Contradiction quarantine ──────────────────────────────────────────

CONTRADICTIONS_FILE = KNOWLEDGE_DIR / "contradictions.json"


def load_contradictions(path: Path | None = None) -> set[str]:
    """Load the quarantined article slug set. Missing file = empty set.

    On corruption or schema violation, raises RuntimeError rather than
    silently returning an empty set. This is load-bearing: a quarantine
    that silently disappears on I/O trouble defeats the whole feature.
    """
    target = path or CONTRADICTIONS_FILE
    if not target.exists():
        return set()

    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(
            f"Failed to read quarantine file at {target}: {e}. "
            f"Fix the I/O issue or delete the file to reset."
        ) from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Corrupted quarantine file at {target}: {e}. "
            f"Fix the JSON manually or delete the file to reset."
        ) from e

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Quarantine file {target} has wrong schema: expected dict, got {type(data).__name__}."
        )
    quarantined = data.get("quarantined", [])
    if not isinstance(quarantined, list):
        raise RuntimeError(
            f"Quarantine file {target} has wrong schema: "
            f"'quarantined' must be a list, got {type(quarantined).__name__}."
        )
    if not all(isinstance(s, str) for s in quarantined):
        raise RuntimeError(
            f"Quarantine file {target} has wrong schema: "
            f"all entries in 'quarantined' must be strings."
        )

    return set(quarantined)


def save_contradictions(slugs: set[str], path: Path | None = None) -> None:
    """Persist the quarantined slug set as sorted JSON with timestamp.

    Writes atomically via temp-file rename so a crash mid-write can't leave
    a corrupted file that load_contradictions would reject on next read.

    Note on concurrent writers: this function assumes single-writer semantics
    (one lint invocation at a time). Two concurrent lint runs doing
    read-modify-write can still lose flag unions; the atomic write only
    prevents partial-file corruption.
    """
    target = path or CONTRADICTIONS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "quarantined": sorted(slugs),
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(target)


# ── File hashing ──────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    """SHA-256 hash of a file (first 16 hex chars)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ── Slug / naming ─────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def slugify_chunk_id(source_rel: str, section_title: str) -> str:
    """Stable id for a daily-log chunk.

    Format: ``{source_rel}#{kebab-section}`` — e.g.
    ``daily/2026-04-12.md#morning-standup``. Used by vector_store to key
    verbatim drawer chunks and by reindex to delete+re-upsert on changes.

    Empty or all-punctuation section titles fall back to ``"section"``
    so the id is always stable and non-empty.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", section_title.lower()).strip("-") or "section"
    return f"{source_rel}#{slug}"


ARTICLE_ROOTS = frozenset({"concepts", "connections", "qa"})


def resolve_article_path(
    slug: str,
    knowledge_dir: Path | None = None,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a canonical article slug without allowing path traversal."""
    root = (knowledge_dir or KNOWLEDGE_DIR).resolve()
    if not isinstance(slug, str) or not slug or "\\" in slug or "\x00" in slug:
        raise ValueError(f"Invalid article slug: {slug!r}")

    normalized = slug.removesuffix(".md")
    pure = PurePosixPath(normalized)
    parts = pure.parts
    if (
        pure.is_absolute()
        or len(parts) < 2
        or parts[0] not in ARTICLE_ROOTS
        or any(part in {"", ".", ".."} for part in parts)
        or normalized != normalized.strip()
    ):
        raise ValueError(f"Invalid article slug: {slug!r}")

    candidate = (root / f"{normalized}.md").resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Invalid article slug: {slug!r}")
    if must_exist and not candidate.is_file():
        raise FileNotFoundError(f"No article at slug {normalized!r}")
    return candidate


# ── Source config ─────────────────────────────────────────────────────

@dataclass
class SourceGroup:
    """One entry from sources.yaml."""

    id: str
    type: str
    include: list[str]
    exclude: list[str] = field(default_factory=list)
    category: str = ""
    description: str = ""


def load_sources_config() -> list[SourceGroup]:
    """Read and validate sources.yaml. Returns empty list if file missing."""
    if not SOURCES_FILE.exists():
        return []

    import yaml

    raw = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8"))
    if not raw or not isinstance(raw, dict):
        return []

    version = raw.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported sources.yaml version: {version}")

    groups = []
    for entry in raw.get("sources", []):
        if not entry.get("id") or not entry.get("type") or not entry.get("include"):
            continue
        groups.append(SourceGroup(
            id=entry["id"],
            type=entry["type"],
            include=entry["include"],
            exclude=entry.get("exclude", []),
            category=entry.get("category", ""),
            description=entry.get("description", ""),
        ))
    return groups


def resolve_source_files(group: SourceGroup, root: Path | None = None) -> list[Path]:
    """Expand include globs and subtract exclude globs. Returns sorted unique paths."""
    import fnmatch
    from config import ROOT_DIR

    base = root or ROOT_DIR
    included: set[Path] = set()
    for pattern in group.include:
        for match in base.glob(pattern):
            if match.is_file():
                included.add(match.resolve())

    # Exclude patterns are matched against filenames (not re-globbed from root)
    # because include paths often escape the base dir via ../../ and re-globbing
    # from root wouldn't find those files.
    if group.exclude:
        filtered: set[Path] = set()
        for fpath in included:
            skip = False
            for pattern in group.exclude:
                # Strip leading **/ for fnmatch against filename
                pat = pattern.lstrip("*").lstrip("/")
                if fnmatch.fnmatch(fpath.name, pat):
                    skip = True
                    break
            if not skip:
                filtered.add(fpath)
        included = filtered

    return sorted(included)


def migrate_state_schema(state: dict) -> dict:
    """Migrate state.json from old schema (flat 'ingested') to new split schema.

    Old: {"ingested": {"2026-04-09.md": {...}}, ...}
    New: {"ingested_daily": {"2026-04-09.md": {...}}, "ingested_sources": {}, ...}

    Idempotent: safe to call multiple times.

    **Pure** — returns a new dict and never mutates its argument. It used to
    mutate in place *and* return the same object, which quietly turned the
    obvious caller idiom into data loss::

        migrated = migrate_state_schema(current)
        current.clear()          # also empties `migrated` — same object
        current.update(migrated) # copies nothing back

    Both ``compile.py`` and ``ingest.py`` independently wrote exactly that,
    and both destroyed state.json on every run. Purity is what makes the
    idiom safe for every caller, including ones not written yet.
    """
    state = dict(state)

    if "ingested_daily" in state:
        state.setdefault("ingested_sources", {})
        return state

    if "ingested" in state:
        state["ingested_daily"] = state.pop("ingested")
    else:
        state["ingested_daily"] = {}

    state.setdefault("ingested_sources", {})
    return state


def migrate_state_mutator(current: dict) -> None:
    """In-place schema migration, for use as an ``update_state`` callback.

    The single shared implementation. Two hand-rolled copies drifted apart
    once already — one was fixed and the other kept destroying state.json on
    the scheduler's timetable.
    """
    migrated = migrate_state_schema(current)
    current.clear()
    current.update(migrated)


# ── Wikilink helpers ──────────────────────────────────────────────────

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[wikilinks]] from markdown content.

    Wikilinks inside HTML comments are skipped — they're explicitly
    disabled markers (typically used by the lint --fix broken_link
    fallback to deactivate a link without losing the original text).
    Counting them as live links would re-flag previously-fixed errors
    on every subsequent lint run.

    Strips any trailing ``{relation}`` annotation so callers see the bare
    target (e.g. ``concepts/foo``) regardless of whether the source wrote
    ``[[concepts/foo]]`` or ``[[concepts/foo]]{depends_on}``. The relation
    metadata is exposed separately via ``extract_typed_wikilinks``.
    """
    stripped = _HTML_COMMENT_RE.sub("", content)
    return re.findall(r"\[\[([^\]]+)\]\]", stripped)


# Pattern captures (target, optional_relation). The relation is the
# trailing ``{...}`` token immediately following the closing ``]]``,
# limited to lowercase letters, digits, and underscores.
_TYPED_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\](?:\{([a-z0-9_]+)\})?")


def extract_typed_wikilinks(content: str) -> list[tuple[str, str | None]]:
    """Extract wikilinks with their optional ``{relation}`` annotation.

    Returns a list of ``(target, relation)`` tuples where ``relation`` is
    ``None`` for plain ``[[target]]`` and a string for ``[[target]]{relation}``.
    HTML comments are skipped, matching ``extract_wikilinks``. Used by
    ``lint.check_wikilink_relations`` to validate against the
    ``config.WIKILINK_RELATIONS`` allowed set without disturbing the
    untyped retrieval path.
    """
    stripped = _HTML_COMMENT_RE.sub("", content)
    return [(m.group(1), m.group(2)) for m in _TYPED_WIKILINK_RE.finditer(stripped)]


def wiki_article_exists(link: str) -> bool:
    """Check if a wikilinked article exists on disk."""
    path = KNOWLEDGE_DIR / f"{link}.md"
    return path.exists()


def extract_source_anchors(content: str) -> list[str]:
    """Extract all [src:path/to/file.md] source anchors from article text.

    Anchors are used by lint.py to verify that cited sources still exist.
    Returns the raw path (relative to the knowledge root or project root)
    for each anchor occurrence. Returns an empty list if no anchors are found.
    """
    return re.findall(r"\[src:([^\]]+)\]", content)


def verify_source_anchor(anchor: str) -> bool:
    """Check whether a [src:...] anchor points to an existing file.

    Anchors may be relative to KNOWLEDGE_DIR (daily/ logs, sources/ drops),
    the outer project root (sources/... referenced in sources.yaml), or the
    memory-compiler directory itself (local docs). Checks all three candidate
    roots for v1.
    """
    from config import PROJECT_ROOT

    candidates = [
        KNOWLEDGE_DIR / anchor,
        PROJECT_ROOT / anchor,
        PROJECT_ROOT / ".claude" / "memory-compiler" / anchor,
    ]
    return any(c.exists() for c in candidates)


# ── Wiki content helpers ──────────────────────────────────────────────

_EMPTY_INDEX = (
    "# Knowledge Base Index\n\n| Article | Summary | Compiled From | Updated |\n"
    "|---------|---------|---------------|---------|"
)


def read_wiki_index(compact: bool = False) -> str:
    """Read the knowledge base index file.

    ``compact=True`` drops the Summary and Compiled-From columns, keeping only
    the article link and its updated date.

    Why that matters: the full index grows ~530 bytes per article, so at 651
    articles it reached 338 KB (~86k tokens) — 84% of the ingest prompt, which
    pushed the whole thing past the model's context window. The CLI then
    returned "Prompt is too long" and every ingest silently produced nothing
    (see ingest.ingest_source_file). Callers that only need the *names* of
    existing articles — to pick wikilink targets and to prefer UPDATE over
    CREATE — should pass ``compact=True``; the dedup pre-flight already
    supplies full detail for the handful of genuinely similar articles.
    """
    if not INDEX_FILE.exists():
        return _EMPTY_INDEX

    text = INDEX_FILE.read_text(encoding="utf-8")
    if not compact:
        return text

    out: list[str] = []
    for line in text.splitlines():
        stripped = line.removeprefix("﻿").strip()
        # Pull the article link out with a regex rather than splitting the row
        # on "|": an aliased wikilink ([[slug|Label]]) or a summary containing a
        # pipe would shift the columns and silently truncate the link.
        link = re.search(r"\[\[[^\]]+\]\]", stripped)
        if stripped.startswith("|") and stripped.endswith("|") and link:
            updated = stripped.rstrip("|").rsplit("|", 1)[-1].strip()
            out.append(f"| {link.group(0)} | {updated} |")
            continue
        out.append(line)
    return "\n".join(out)


def read_all_wiki_content() -> str:
    """Read index + all wiki articles into a single string for context."""
    parts = [f"## INDEX\n\n{read_wiki_index()}"]

    for subdir in [CONCEPTS_DIR, CONNECTIONS_DIR, QA_DIR]:
        if not subdir.exists():
            continue
        for md_file in sorted(subdir.glob("*.md")):
            rel = md_file.relative_to(KNOWLEDGE_DIR)
            content = md_file.read_text(encoding="utf-8")
            parts.append(f"## {rel}\n\n{content}")

    return "\n\n---\n\n".join(parts)


def list_wiki_articles() -> list[Path]:
    """List all wiki article files."""
    articles = []
    for subdir in [CONCEPTS_DIR, CONNECTIONS_DIR, QA_DIR]:
        if subdir.exists():
            articles.extend(sorted(subdir.glob("*.md")))
    return articles


def list_raw_files() -> list[Path]:
    """List all daily log files."""
    if not DAILY_DIR.exists():
        return []
    return sorted(DAILY_DIR.glob("*.md"))


def list_transcript_files() -> list[Path]:
    """List byte-for-byte raw session transcripts retained for retrieval."""
    transcript_dir = DAILY_DIR / "transcripts"
    if not transcript_dir.exists():
        return []
    return sorted(transcript_dir.glob("*.jsonl"))


def daily_file_lock_path(daily_path: Path) -> Path:
    """Return the shared lock for one daily file's append/index transaction."""
    return daily_path.parent / f".{daily_path.name}.write.lock"


# ── Vector-store embedding helpers ────────────────────────────────────

def embed_article_file(
    article_path: Path,
    quarantined: set[str] | None = None,
) -> bool:
    """Embed (or re-embed) one article file into the vector store.

    Reads the article, parses frontmatter, splits into Observed/Synthesized
    zones, and upserts each non-empty zone as a separate Chroma document.
    Legacy articles without ``## Truth`` fall back to ``extract_fallback_truth``
    and are stored entirely in the Observed zone.

    Safely no-ops when the article is gone. Returns True if at least one
    zone was embedded, False otherwise.

    Pass ``quarantined`` to skip the per-call ``load_contradictions()``
    read when bulk-embedding — otherwise reindex.py would re-parse
    contradictions.json once per article. Single-file callers can leave
    it as None and pay one read.

    Shared between ``reindex.py`` (bulk backfill) and ``compile.py`` /
    ``ingest.py`` (per-file updates) so both paths normalize the same way.
    """
    if not article_path.exists():
        return False

    from compile_truth import (
        TruthZones,
        extract_fallback_truth,
        extract_zones,
        parse_frontmatter,
    )
    from vector_store import replace_article

    if quarantined is None:
        quarantined = load_contradictions()

    rel = str(article_path.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")
    slug = rel.removesuffix(".md")
    content = article_path.read_text(encoding="utf-8")
    fm = parse_frontmatter(content)
    zones = extract_zones(content)

    # Legacy articles (no ## Truth header) need fallback extraction so
    # the vector store doesn't silently drop them.
    if not zones.observed and not zones.synthesized:
        legacy = extract_fallback_truth(content)
        if legacy:
            zones = TruthZones(observed=legacy, synthesized="")

    # type: defaults to None for legacy articles that haven't been tagged
    # yet (Task 7 migrates them). Using None rather than "fact" keeps
    # search_knowledge type_filter queries honest — a legacy article
    # isn't a confirmed fact, it's untyped.
    metadata = {
        "type": fm.get("type"),
        "confidence": float(fm.get("confidence", 0.5)),
        "quarantined": slug in quarantined,
        "updated": fm.get("updated") or fm.get("created") or "unknown",
        "pinned": bool(fm.get("pinned", False)),
    }
    title = fm.get("title") or slug.split("/")[-1]

    zone_documents = {
        "observed": zones.observed,
        "synthesized": zones.synthesized,
    }
    replace_article(slug, title, zone_documents, metadata)
    return any(text.strip() for text in zone_documents.values())


def embed_daily_file(daily_path: Path) -> int:
    """Re-chunk and re-embed one daily log. Returns number of chunks embedded.

    Requires ``scripts/chunk_daily.py`` (Task 5 of the steal-list plan).
    Raises ``ModuleNotFoundError(name='chunk_daily')`` if called before
    Task 5 lands — callers that need to tolerate the missing module
    should catch that specific exception. ``reindex.main()`` already
    handles it gracefully with a stderr hint.

    Kept here alongside ``embed_article_file`` so both embedding paths
    share one home.
    """
    if not daily_path.exists():
        return 0

    from chunk_daily import chunk_daily_log
    from vector_store import replace_chunks_for_source

    rel = str(daily_path.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")
    date_str = daily_path.stem  # YYYY-MM-DD

    content = daily_path.read_text(encoding="utf-8")
    chunks: list[dict] = []
    for chunk in chunk_daily_log(content, source_rel=rel):
        chunks.append({
            "chunk_id": chunk.id,
            "source_file": rel,
            "text": chunk.text,
            "metadata": {"section": chunk.section, "date": date_str},
        })
    replace_chunks_for_source(rel, chunks)
    return len(chunks)


# ── Index helpers ─────────────────────────────────────────────────────

def count_inbound_links(target: str, exclude_file: Path | None = None) -> int:
    """Count how many wiki articles link to a given target."""
    count = 0
    for article in list_wiki_articles():
        if article == exclude_file:
            continue
        content = article.read_text(encoding="utf-8")
        if f"[[{target}]]" in content:
            count += 1
    return count


def get_article_word_count(path: Path) -> int:
    """Count words in an article, excluding YAML frontmatter."""
    content = path.read_text(encoding="utf-8")
    # Strip frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:]
    return len(content.split())


def build_index_entry(rel_path: str, summary: str, sources: str, updated: str) -> str:
    """Build a single index table row."""
    link = rel_path.replace(".md", "")
    return f"| [[{link}]] | {summary} | {sources} | {updated} |"
