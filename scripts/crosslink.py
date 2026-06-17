"""Cross-linker / wikilink backfill pass — pure Python, zero LLM cost.

Scans every knowledge article body for UNLINKED prose mentions of other
articles' titles or aliases and weaves in ``[[wikilinks]]`` to densify the
knowledge graph. This repo's wikilink regex (see ``unified_graph``) is
PATH-based (``[[concepts/foo]]``) and does NOT support ``[[slug|display]]``
pipe aliases — a pipe would corrupt the graph node id. So instead of
inline-rewriting prose into piped links, when article A mentions article B
we append ``[[<B-slug>]]`` to A's ``### Related Concepts`` section (which
lives under ``## Truth``), creating the section if absent.

Matching is deliberately conservative — false links pollute the graph:

* case-insensitive, whole-word (regex word boundaries)
* titles/aliases longer than 3 chars only; longest first
* never inside frontmatter, fenced code, inline code, existing
  ``[[wikilinks]]``, or markdown ``[..](..)`` links
* never link an article to itself; never duplicate an existing A→B link

CLI: ``uv run python scripts/crosslink.py`` (dry-run) / ``--apply``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

# The ``config`` module resolves under two import regimes depending on how
# the calling entrypoint set up sys.path — mirror unified_graph's pattern.
try:
    from scripts import config
except ImportError:  # pragma: no cover - exercised only via CLI entrypoint
    import config


# ── Regexes ────────────────────────────────────────────────────────────

# Path-based wikilink, optional {relation} suffix (matches unified_graph).
_WIKILINK_RE = re.compile(r"\[\[([^\]]+?)\]\](?:\{[a-z0-9_]+\})?")
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.DOTALL)

_RELATED_HEADING = "### Related Concepts"
_LINK_SUFFIX = " — auto-linked mention"

MIN_TITLE_LEN = 4  # match strings strictly longer than 3 chars


# ── Pure functions ─────────────────────────────────────────────────────

def build_title_index(articles: list[dict]) -> dict:
    """Map lowercased title/alias → slug.

    ``articles`` is a list of ``{slug, title, aliases, body}`` dicts. Strings
    of length <= 3 are skipped (too ambiguous). If two articles share a
    title/alias the first wins (stable input order) — collisions are rare and
    a deterministic choice keeps the pass reproducible.
    """
    index: dict[str, str] = {}
    for art in articles:
        slug = art["slug"]
        names = [art.get("title") or ""]
        names.extend(art.get("aliases") or [])
        for name in names:
            name = (name or "").strip()
            if len(name) < MIN_TITLE_LEN:
                continue
            key = name.lower()
            index.setdefault(key, slug)
    return index


def existing_links(body: str) -> set[str]:
    """Return the set of slugs ``body`` already links to via ``[[...]]``."""
    return {m.group(1).strip() for m in _WIKILINK_RE.finditer(body)}


def mask_non_prose(body: str) -> str:
    """Blank out regions where a title match must NOT count.

    Masks (replaces with spaces, preserving length so offsets stay sane):
    frontmatter, fenced code blocks, inline code, existing wikilinks, and
    markdown links. Returns text safe for whole-word title matching.
    """
    def _blank(match: re.Match) -> str:
        # Preserve newlines so line structure (and thus length) is stable.
        return re.sub(r"[^\n]", " ", match.group(0))

    masked = _FRONTMATTER_RE.sub(_blank, body)
    masked = _FENCED_CODE_RE.sub(_blank, masked)
    masked = _WIKILINK_RE.sub(_blank, masked)
    masked = _MD_LINK_RE.sub(_blank, masked)
    masked = _INLINE_CODE_RE.sub(_blank, masked)
    return masked


def find_missing_links(body: str, self_slug: str, index: dict) -> list[str]:
    """Ordered, de-duplicated list of slugs to add to ``body``.

    A slug is included when one of its titles/aliases appears as a whole word
    (case-insensitive) in the prose of ``body`` AND ``body`` does not already
    link to it AND it is not ``self_slug``. Results are ordered by first
    appearance of any matching name in the (masked) body. Longer names are
    tried first so a long title isn't shadowed by a shorter substring.
    """
    masked = mask_non_prose(body)
    already = existing_links(body)
    lowered = masked.lower()

    # name -> slug, longest names first (longest-match preference).
    names_sorted = sorted(index.keys(), key=len, reverse=True)

    first_pos: dict[str, int] = {}
    for name in names_sorted:
        slug = index[name]
        if slug == self_slug or slug in already:
            continue
        if slug in first_pos:
            continue  # already located this target via a longer name
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        m = pattern.search(lowered)
        if m:
            first_pos[slug] = m.start()

    return sorted(first_pos, key=lambda s: first_pos[s])


def add_related_links(text: str, slugs: list[str]) -> str:
    """Append ``[[slug]]`` bullets to the ``### Related Concepts`` section.

    Idempotent: slugs already present (anywhere in the article as a wikilink)
    are skipped. Creates the section at the end of the ``## Truth`` block (just
    before the ``---``/``## Timeline`` separator) if it is absent. Returns the
    (possibly unchanged) full article text.
    """
    if not slugs:
        return text

    present = existing_links(text)
    to_add = [s for s in slugs if s not in present]
    if not to_add:
        return text

    bullets = "\n".join(f"- [[{s}]]{_LINK_SUFFIX}" for s in to_add)

    if _RELATED_HEADING in text:
        return _append_to_existing_section(text, bullets)
    return _insert_new_section(text, bullets)


def _append_to_existing_section(text: str, bullets: str) -> str:
    """Insert ``bullets`` after the last bullet of the existing section."""
    heading_idx = text.index(_RELATED_HEADING)
    after_heading = heading_idx + len(_RELATED_HEADING)

    # Find the end of the section: next heading or the timeline separator.
    rest = text[after_heading:]
    end_match = re.search(r"\n(?=(#{1,3} |---\s*\n))", rest)
    section_end = after_heading + (end_match.start() if end_match else len(rest))

    body_block = text[:section_end].rstrip("\n")
    tail = text[section_end:]
    return f"{body_block}\n{bullets}\n{tail}" if tail else f"{body_block}\n{bullets}\n"


def _insert_new_section(text: str, bullets: str) -> str:
    """Create the section at the end of ## Truth, before the separator."""
    section = f"{_RELATED_HEADING}\n\n{bullets}\n"

    # Skip past frontmatter so its closing ``---`` is not mistaken for the
    # Truth/Timeline separator.
    fm = _FRONTMATTER_RE.match(text)
    scan_from = fm.end() if fm else 0

    # Prefer inserting right before the timeline separator (a ``---`` line or a
    # ``## Timeline`` heading) that closes the Truth block.
    sep_match = re.search(r"\n---\s*\n", text[scan_from:])
    timeline_match = re.search(r"\n## Timeline\b", text[scan_from:])

    insert_at = None
    if sep_match:
        insert_at = scan_from + sep_match.start() + 1  # keep the leading newline
    elif timeline_match:
        insert_at = scan_from + timeline_match.start() + 1

    if insert_at is not None:
        before = text[:insert_at].rstrip("\n")
        after = text[insert_at:]
        return f"{before}\n\n{section}\n{after}"

    # No separator at all — append to the very end.
    return f"{text.rstrip(chr(10))}\n\n{section}"


# ── Article loading ────────────────────────────────────────────────────

def _parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter robustly with pyyaml; ``{}`` on absence/error."""
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return {}
    raw = content[: m.end()]
    inner = raw.split("---", 2)[1] if raw.count("---") >= 2 else ""
    try:
        data = yaml.safe_load(inner) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def load_articles(knowledge_root: Path) -> list[dict]:
    """Load all articles as ``{slug, title, aliases, body, path}`` dicts."""
    articles: list[dict] = []
    for subdir in ("concepts", "connections", "qa"):
        root = knowledge_root / subdir
        if not root.exists():
            continue
        for md in sorted(root.glob("*.md")):
            content = md.read_text(encoding="utf-8")
            fm = _parse_frontmatter(content)
            title = fm.get("title") or md.stem
            aliases = fm.get("aliases") or []
            if not isinstance(aliases, list):
                aliases = [aliases]
            aliases = [str(a) for a in aliases if a]
            articles.append(
                {
                    "slug": f"{subdir}/{md.stem}",
                    "title": str(title),
                    "aliases": aliases,
                    "body": content,
                    "path": md,
                }
            )
    return articles


# ── CLI ────────────────────────────────────────────────────────────────

def run(knowledge_root: Path, apply: bool = False, verbose: bool = False) -> int:
    """Scan articles, report (and optionally apply) missing wikilinks."""
    articles = load_articles(knowledge_root)
    index = build_title_index(articles)

    total_links = 0
    changed_articles = 0

    for art in articles:
        missing = find_missing_links(art["body"], art["slug"], index)
        if not missing:
            if verbose:
                print(f"  (no new links) {art['slug']}")
            continue

        changed_articles += 1
        total_links += len(missing)
        print(f"{art['slug']}: +{len(missing)} link(s)")
        for slug in missing:
            print(f"    -> [[{slug}]]")

        if apply:
            new_text = add_related_links(art["body"], missing)
            if new_text != art["body"]:
                art["path"].write_text(new_text, encoding="utf-8")

    mode = "APPLIED" if apply else "DRY-RUN (use --apply to write)"
    print(
        f"\n{mode}: {total_links} link(s) across {changed_articles} "
        f"article(s) of {len(articles)} scanned."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill [[wikilinks]] for unlinked article-title mentions."
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write changes to disk (default: dry-run preview only).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Also list articles with no new links.",
    )
    args = parser.parse_args(argv)
    return run(config.KNOWLEDGE_DIR, apply=args.apply, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
