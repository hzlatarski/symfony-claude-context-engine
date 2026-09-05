"""Entity extraction pass — pure Python, zero LLM cost. (Slice 1 of the
entity cross-referencing feature — see ``ENTITY-EXTRACTION-PLAN.md``.)

Lifts recurring NAMED THINGS out of article prose — servers, project
commands, services, env vars, URLs, source paths — so the unified graph can
later cross-reference every article that mentions the *same* one. Today two
articles that both name prod host ``65.21.4.203`` share no graph edge unless
one is an article title; entities close that gap.

**This module is EXTRACTION ONLY.** It finds entities and reports them. It
does NOT touch the graph and it NEVER writes to an article file. Wiring the
hits into ``unified_graph.build()`` as ``entity:`` nodes + ``mentions`` edges
is Slice 2 — and it will follow the ``note:`` rationale-node precedent
(materialized at graph-build time, source files untouched).

Design mirrors ``crosslink.py``: a small, closed, high-precision vocabulary
(false entities pollute the graph exactly like false wikilinks), whole-word
matching, and matching only in real prose — never inside frontmatter, code
fences, inline code, existing ``[[wikilinks]]``, markdown links, or
``[src:...]`` anchors (reuses ``crosslink.mask_non_prose`` plus a local
src-anchor mask).

CLI (dry-run report over the live KB — for eyeballing the false-positive rate
before Slice 2 wires anything into the graph):

    uv run python scripts/entities.py             # every entity, per type
    uv run python scripts/entities.py --min 2     # only entities in >=2 articles
    uv run python scripts/entities.py --type host # one type only
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# ``crosslink`` (for mask_non_prose + load_articles) and ``config`` resolve
# under two import regimes depending on how the entrypoint set up sys.path —
# mirror crosslink's own pattern.
try:
    from scripts import config, crosslink
except ImportError:  # pragma: no cover - exercised only via CLI entrypoint
    import config
    import crosslink


# ── Data model ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Entity:
    """One entity mention found in a body.

    ``canonical_id`` is the graph node id a Slice-2 wiring would mint,
    ``entity:<type>/<canonical-value>``. Two different surface forms that
    normalize to the same value (``65.21.4.203`` and ``prod (65.21.4.203)``)
    produce the SAME ``canonical_id`` — that is where cross-referencing
    happens.
    """
    type: str
    canonical_id: str
    surface: str
    pos: int


@dataclass(frozen=True)
class Recognizer:
    """A named entity recognizer: a regex plus a normalizer.

    ``normalize`` returns the canonical value, or ``None`` to REJECT the match
    (e.g. an IPv4-shaped token whose octets are out of range). Rejection keeps
    the regexes loose and readable while precision lives in one place.
    """
    name: str
    pattern: re.Pattern
    normalize: Callable[[str], str | None]


# ── Recognizers (conservative, high-precision) ─────────────────────────

_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


def _norm_host(surface: str) -> str | None:
    """Accept only a genuine IPv4 (every octet 0-255) — rejects version
    strings like ``1.2.3.400`` while keeping ``65.21.4.203``."""
    octets = surface.split(".")
    try:
        if all(0 <= int(o) <= 255 for o in octets):
            return surface
    except ValueError:
        return None
    return None


# Project console namespace: ``app:content:narrate``, ``app:images:push-to-cdn``.
# Requires at least one ``:segment`` after ``app:`` so bare ``app:`` never matches.
_CMD_RE = re.compile(r"\bapp:[a-z0-9]+(?::[a-z0-9][a-z0-9-]*)+\b")


def _norm_cmd(surface: str) -> str | None:
    return surface.lower()


# Symfony security roles: ``ROLE_ADMIN``, ``ROLE_CORPORATE_MANAGER``. Their own
# high-precision type — split out of ``envvar`` so neither pollutes the other.
_ROLE_RE = re.compile(r"\bROLE_[A-Z][A-Z0-9_]*\b")


def _norm_role(surface: str) -> str | None:
    return surface


# Real env vars, not just any SCREAMING_SNAKE constant. The broad shape catches
# every all-caps constant (``ROLE_ADMIN``, ``STATUS_DISABLED``, ``DEFAULT_RATES``
# — none of them env vars), so precision lives in a suffix allowlist: the token's
# LAST segment must name an env-var-ish thing. Precision over recall — a suffixless
# real var (``DEPLOY_AGENT``) is deliberately missed rather than let the junk in.
_ENV_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_ENV_SUFFIXES = frozenset({
    "KEY", "SECRET", "TOKEN", "DSN", "URL", "URI", "PATH", "HOST",
    "PORT", "ENDPOINT", "REGION", "BUCKET", "PASSWORD", "PWD",
})


def _norm_env(surface: str) -> str | None:
    return surface if surface.rsplit("_", 1)[-1] in _ENV_SUFFIXES else None


_URL_RE = re.compile(r"\b(?:https?|wss?)://[^\s)\]}>\"'`]+")


def _norm_url(surface: str) -> str | None:
    stripped = re.sub(r"^(?:https?|wss?)://", "", surface).rstrip("/.,;:)]}")
    return stripped.lower() or None


# Repo-relative source paths in the known top dirs.
_PATH_RE = re.compile(r"\b(?:src|templates|assets|config)/[A-Za-z0-9_][A-Za-z0-9_./-]*")


def _norm_path(surface: str) -> str | None:
    stripped = surface.rstrip("/.,;:)]}")
    # Require a real path: a second segment beyond the top dir, or a file
    # extension on the tail — rejects a bare ``src/`` fragment.
    last = stripped.split("/")[-1]
    if stripped.count("/") >= 2 or "." in last:
        return stripped
    return None


# PascalCase class names that are load-bearing by convention. ``Command`` is
# deliberately excluded (the ``app:`` recognizer already covers commands and a
# bare ``...Command`` PascalCase token is lower-precision).
_SVC_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]*"
    r"(?:Service|Repository|Controller|Handler|Subscriber|Listener|Factory)\b"
)


def _norm_service(surface: str) -> str | None:
    return surface


RECOGNIZERS: list[Recognizer] = [
    Recognizer("host", _IPV4_RE, _norm_host),
    Recognizer("command", _CMD_RE, _norm_cmd),
    Recognizer("role", _ROLE_RE, _norm_role),
    Recognizer("envvar", _ENV_RE, _norm_env),
    Recognizer("url", _URL_RE, _norm_url),
    Recognizer("path", _PATH_RE, _norm_path),
    Recognizer("service", _SVC_RE, _norm_service),
]


# ── Masking ────────────────────────────────────────────────────────────

_SRC_ANCHOR_RE = re.compile(r"\[src:[^\]]*\]")


def _blank(match: re.Match) -> str:
    """Replace a region with spaces, preserving newlines (offset-stable)."""
    return re.sub(r"[^\n]", " ", match.group(0))


def mask(body: str) -> str:
    """Blank every non-prose region for entity matching.

    Starts from ``crosslink.mask_non_prose`` (frontmatter, fenced/inline code,
    existing wikilinks, markdown links) and additionally blanks ``[src:...]``
    anchors — those already become ``file:`` nodes via ``cites`` edges, so
    counting them as ``path`` entities would double-represent one thing.
    """
    masked = crosslink.mask_non_prose(body)
    return _SRC_ANCHOR_RE.sub(_blank, masked)


# ── Extraction ─────────────────────────────────────────────────────────

def extract(body: str) -> list[Entity]:
    """Return the de-duplicated entities mentioned in ``body``'s prose.

    De-duplication is per ``canonical_id`` (first occurrence wins its
    position), so an entity named five times yields one ``Entity``. Results
    are ordered by first appearance in the masked body.
    """
    masked = mask(body)
    found: dict[str, Entity] = {}
    for rec in RECOGNIZERS:
        for m in rec.pattern.finditer(masked):
            surface = m.group(0)
            value = rec.normalize(surface)
            if value is None:
                continue
            canonical_id = f"entity:{rec.name}/{value}"
            if canonical_id in found:
                continue
            found[canonical_id] = Entity(rec.name, canonical_id, surface, m.start())
    return sorted(found.values(), key=lambda e: e.pos)


# ── Dry-run report ─────────────────────────────────────────────────────

def collect(knowledge_root: Path, entity_type: str | None = None) -> dict:
    """Aggregate entities across the whole KB.

    Returns ``{canonical_id: {type, surface, slugs:set[str]}}`` — ``slugs`` is
    every article that mentions the entity, which is exactly the
    cross-reference set a Slice-2 ``mentions`` edge would materialize.
    """
    articles = crosslink.load_articles(knowledge_root)
    by_entity: dict[str, dict] = {}
    for art in articles:
        for e in extract(art["body"]):
            if entity_type and e.type != entity_type:
                continue
            rec = by_entity.setdefault(
                e.canonical_id, {"type": e.type, "surface": e.surface, "slugs": set()}
            )
            rec["slugs"].add(art["slug"])
    return by_entity


def run(
    knowledge_root: Path,
    min_articles: int = 1,
    entity_type: str | None = None,
    verbose: bool = False,
) -> int:
    by_entity = collect(knowledge_root, entity_type=entity_type)

    # Per-type totals (over ALL entities, before the --min filter).
    type_counts: dict[str, int] = {}
    for rec in by_entity.values():
        type_counts[rec["type"]] = type_counts.get(rec["type"], 0) + 1

    rows = [
        (cid, rec) for cid, rec in by_entity.items()
        if len(rec["slugs"]) >= min_articles
    ]
    # Most cross-referenced first — that is the whole point of the feature.
    rows.sort(key=lambda kv: (-len(kv[1]["slugs"]), kv[0]))

    print("=== Entity extraction (dry-run) ===")
    print(f"knowledge root: {knowledge_root}")
    print(f"distinct entities: {len(by_entity)}   shown (>= {min_articles} article(s)): {len(rows)}\n")

    print("by type:")
    for t in sorted(type_counts):
        print(f"  {t:<9} {type_counts[t]}")
    print()

    for cid, rec in rows:
        n = len(rec["slugs"])
        print(f"[{n:>3}] {cid}")
        if verbose:
            for slug in sorted(rec["slugs"]):
                print(f"         {slug}")

    cross = sum(1 for _, rec in by_entity.items() if len(rec["slugs"]) >= 2)
    print(f"\ncross-referenced (in >= 2 articles): {cross}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract named entities from KB article prose (Slice 1 — report only)."
    )
    parser.add_argument(
        "--min", type=int, default=1, metavar="N",
        help="Only show entities mentioned in at least N articles (default: 1).",
    )
    parser.add_argument(
        "--type", dest="entity_type", default=None,
        choices=[r.name for r in RECOGNIZERS],
        help="Restrict to a single entity type.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="List the articles each entity appears in.",
    )
    args = parser.parse_args(argv)
    return run(
        config.KNOWLEDGE_DIR,
        min_articles=args.min,
        entity_type=args.entity_type,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
