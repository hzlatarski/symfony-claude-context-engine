"""Query expansion via Haiku: transcript → queries + intent + scope.

This is the first LLM step in the enhance pipeline. A small/fast model
reads the rough voice transcript and emits a JSON object that drives
the retrieval fan-out and the rewrite prompt's metadata.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import config
from whisper.prompts import QUERY_EXPANSION_SYSTEM_PROMPT


VALID_INTENTS = {
    "implement", "refactor", "audit", "debug", "explain", "document",
    "plan", "design", "brainstorm", "write", "copy", "marketing",
    "reflect", "decide", "discuss", "generic",
}

VALID_SCOPES = {"articles", "code", "daily"}

MAX_QUERIES = 10            # Haiku prompted for 3-5; cap provides safety margin
MAX_QUERY_LENGTH = 500      # per-query char cap defends retrieval embedding
EXPAND_TIMEOUT_SECONDS = 15.0   # tight budget for voice UX; Haiku is normally sub-second


class ExpansionError(Exception):
    """Raised when the Haiku call or its output is unusable."""


@dataclass
class Expansion:
    queries: list[str]
    intent: str
    scope: list[str]


@lru_cache(maxsize=1)
def _get_client():
    """Lazy Anthropic client, reused across calls within one process."""
    from anthropic import Anthropic
    return Anthropic()


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that some Haiku outputs wrap JSON in."""
    return _FENCE_RE.sub("", text).strip()


def _parse_json(text: str) -> dict[str, Any]:
    stripped = _strip_fences(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ExpansionError(f"Haiku returned non-JSON output: {e}") from e


def _validate_intent(raw: Any) -> str:
    if not isinstance(raw, str):
        return "generic"
    return raw if raw in VALID_INTENTS else "generic"


def _validate_scope(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return ["articles"]
    kept = [s for s in raw if isinstance(s, str) and s in VALID_SCOPES]
    return kept if kept else ["articles"]


def expand(transcript: str) -> Expansion:
    """Expand a rough transcript into retrieval queries + intent + scope.

    Args:
        transcript: the user's voice utterance, transcribed to text.

    Returns:
        An Expansion with validated fields.

    Raises:
        ExpansionError: if Haiku returns non-JSON or no queries.
    """
    client = _get_client()
    resp = client.messages.create(
        model=config.MODEL_EXPAND,
        max_tokens=512,
        system=QUERY_EXPANSION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript}],
        timeout=EXPAND_TIMEOUT_SECONDS,
    )

    # Anthropic Messages API returns content as a list of blocks
    text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    if not text_blocks:
        raise ExpansionError("Haiku returned no text blocks")
    raw = "\n".join(text_blocks)

    data = _parse_json(raw)

    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ExpansionError("Haiku response missing 'queries' list")
    queries = [
        q.strip()[:MAX_QUERY_LENGTH]
        for q in queries
        if isinstance(q, str) and q.strip()
    ]
    if not queries:
        raise ExpansionError("Haiku response 'queries' contained no valid strings")
    queries = queries[:MAX_QUERIES]

    return Expansion(
        queries=queries,
        intent=_validate_intent(data.get("intent")),
        scope=_validate_scope(data.get("scope")),
    )
