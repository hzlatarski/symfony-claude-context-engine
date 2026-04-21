"""Direct-ingest canonicalized scenarios into the ChromaDB articles collection.

The regular `ingest.py` pipeline runs every source file through the Claude
Agent SDK, which *distills* each file into wiki-style concept articles —
losing the structured taxonomy frontmatter (primary_technique,
persona_archetype, counter_patterns, etc.) that the dedup adjudicator
depends on. So the scenarios need their own ingest path that preserves
frontmatter as searchable metadata and indexes the structured axes
directly into the embedded document text.

Input:  knowledge/scenarios/<track-slug>/<scenario-slug>.md files with
        YAML frontmatter produced by `ScenarioKnowledgeExporterService`.
Output: One entry per scenario in the `articles` ChromaDB collection,
        keyed as `scenarios/<track-slug>/<scenario-slug>`, zone="observed".

Run with:
    uv run python scripts/ingest_scenarios.py
    uv run python scripts/ingest_scenarios.py --only tactical-negotiation

Called by scripts/refresh-scenario-kb.sh after the export command updates
the markdown files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import vector_store  # noqa: E402
from config import PROJECT_ROOT  # noqa: E402


SCENARIOS_DIR = PROJECT_ROOT / "knowledge" / "scenarios"

# How much of the article body to include in the embedded document. The
# taxonomy block at the top already gives the embedder the structured
# axes; the body adds enough prose for semantic hits on unusual queries
# but stays short enough to keep embedding cost down.
BODY_CHARS = 1200


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Split a YAML-frontmatter markdown file into (frontmatter, body)."""
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    fm_text = raw[3:end].strip()
    body = raw[end + 4:].lstrip("\n")
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}
    if not isinstance(fm, dict):
        fm = {}
    return fm, body


def _build_embedding_text(fm: dict[str, Any], body: str) -> str:
    """Construct the text we hand to the embedder.

    Structured taxonomy axes go first (matches how the adjudicator queries:
    `technique:X persona:Y stakes:Z counter:A,B`), then a truncated body
    tail gives semantic context for fuzzy queries.
    """
    def _join_list(v: Any) -> str:
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return str(v or "")

    header_lines = [
        f"Title: {fm.get('title') or ''}",
        f"Track: {fm.get('track') or ''}",
        f"Phase: {fm.get('phase') or ''}",
        f"Primary technique: {fm.get('primary_technique') or ''}",
        f"Secondary techniques: {_join_list(fm.get('secondary_techniques'))}",
        f"Counter patterns: {_join_list(fm.get('counter_patterns'))}",
        f"Persona archetype: {fm.get('persona_archetype') or ''}",
        f"Stakes level: {fm.get('stakes_level') or ''}",
        f"Difficulty: {fm.get('difficulty_signal') or fm.get('difficulty') or ''}",
        f"Target skills: {_join_list(fm.get('target_skills'))}",
        f"Teaching angle: {fm.get('teaching_angle') or ''}",
    ]
    header = "\n".join(line for line in header_lines if line.split(": ", 1)[1].strip())

    trimmed_body = body.strip()
    if len(trimmed_body) > BODY_CHARS:
        trimmed_body = trimmed_body[:BODY_CHARS].rstrip() + "…"

    return header + "\n\n" + trimmed_body


def _metadata_for(fm: dict[str, Any]) -> dict[str, Any]:
    """Pick taxonomy fields the adjudicator will read back on search hits.

    Lists are passed through — vector_store._flatten_metadata joins them
    with commas AND emits `<key>_<value>: True` boolean columns for
    exact-match filtering (e.g. `counter_patterns_consensus_seeking_loop`).
    """
    return {
        "source_category": "scenarios",
        "track": fm.get("track"),
        "track_slug": fm.get("track_slug"),
        "phase": fm.get("phase"),
        "scenario_uuid": fm.get("scenario_uuid"),
        "primary_technique": fm.get("primary_technique"),
        "secondary_techniques": fm.get("secondary_techniques") or [],
        "counter_patterns": fm.get("counter_patterns") or [],
        "persona_archetype": fm.get("persona_archetype"),
        "stakes_level": fm.get("stakes_level"),
        "difficulty_signal": fm.get("difficulty_signal"),
        "target_skills": fm.get("target_skills") or [],
        "teaching_angle": fm.get("teaching_angle"),
        "type": "scenario",
        "confidence": 1.0,
        "quarantined": False,
        "pinned": False,
        "updated": None,
    }


def _slug_for(path: Path) -> str:
    """knowledge/scenarios/<track>/<file>.md -> scenarios/<track>/<file>."""
    rel = path.relative_to(SCENARIOS_DIR)
    return "scenarios/" + rel.with_suffix("").as_posix()


def ingest_one(path: Path, *, verbose: bool) -> bool:
    raw = path.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(raw)
    if not fm.get("slug") or not fm.get("title"):
        if verbose:
            print(f"  SKIP {path.name} — missing slug/title in frontmatter", file=sys.stderr)
        return False

    slug = _slug_for(path)
    text = _build_embedding_text(fm, body)
    metadata = _metadata_for(fm)

    vector_store.upsert_article(
        slug=slug,
        title=str(fm.get("title") or path.stem),
        zone="observed",
        text=text,
        metadata=metadata,
    )
    if verbose:
        print(f"  OK   {slug}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct-ingest scenarios into ChromaDB.")
    parser.add_argument("--only", help="Substring filter on relative path (e.g. track slug)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file output")
    args = parser.parse_args()

    if not SCENARIOS_DIR.is_dir():
        print(f"ingest_scenarios: {SCENARIOS_DIR} does not exist — nothing to ingest.", file=sys.stderr)
        return 0

    files = sorted(SCENARIOS_DIR.rglob("*.md"))
    if args.only:
        files = [f for f in files if args.only in str(f.relative_to(SCENARIOS_DIR))]

    if not files:
        print("ingest_scenarios: no scenario files matched.", file=sys.stderr)
        return 0

    print(f"ingest_scenarios: upserting {len(files)} scenario(s)…")
    ok = 0
    for path in files:
        if ingest_one(path, verbose=not args.quiet):
            ok += 1
    print(f"ingest_scenarios: done — {ok}/{len(files)} scenarios indexed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
