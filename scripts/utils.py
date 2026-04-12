"""Shared utilities for the personal knowledge base."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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

def load_state() -> dict:
    """Load persistent state from state.json."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "ingested_daily": {},
        "ingested_sources": {},
        "access_counts": {},
        "query_count": 0,
        "last_lint": None,
        "total_cost": 0.0,
    }


def save_state(state: dict) -> None:
    """Save state to state.json."""
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


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
    """
    if "ingested_daily" in state:
        state.setdefault("ingested_sources", {})
        return state

    if "ingested" in state:
        state["ingested_daily"] = state.pop("ingested")
    else:
        state["ingested_daily"] = {}

    state.setdefault("ingested_sources", {})
    return state


# ── Wikilink helpers ──────────────────────────────────────────────────

def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[wikilinks]] from markdown content."""
    return re.findall(r"\[\[([^\]]+)\]\]", content)


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

def read_wiki_index() -> str:
    """Read the knowledge base index file."""
    if INDEX_FILE.exists():
        return INDEX_FILE.read_text(encoding="utf-8")
    return "# Knowledge Base Index\n\n| Article | Summary | Compiled From | Updated |\n|---------|---------|---------------|---------|"


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
    from vector_store import delete_article, upsert_article

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

    # Wipe both zones before upsert so stale zones from an older version
    # of the article don't linger after it shrinks.
    delete_article(slug)

    embedded_any = False
    if zones.observed:
        upsert_article(slug, title, "observed", zones.observed, metadata)
        embedded_any = True
    if zones.synthesized:
        upsert_article(slug, title, "synthesized", zones.synthesized, metadata)
        embedded_any = True

    return embedded_any


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
    from vector_store import delete_chunks_for_daily, upsert_chunk

    rel = str(daily_path.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")
    date_str = daily_path.stem  # YYYY-MM-DD

    delete_chunks_for_daily(rel)
    content = daily_path.read_text(encoding="utf-8")
    count = 0
    for chunk in chunk_daily_log(content, source_rel=rel):
        upsert_chunk(
            chunk_id=chunk.id,
            source_file=rel,
            text=chunk.text,
            metadata={"section": chunk.section, "date": date_str},
        )
        count += 1
    return count


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
