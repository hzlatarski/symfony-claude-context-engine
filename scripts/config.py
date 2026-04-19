"""Path constants and configuration for the personal knowledge base."""

from pathlib import Path
from datetime import datetime, timezone

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT_DIR.parent.parent  # Outer Symfony project root (outside .claude/)
# Knowledge base lives at project root so the Agent SDK can write to it
# (Claude Code blocks writes inside .claude/ even with bypassPermissions)
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
DAILY_DIR = KNOWLEDGE_DIR / "daily"
CONCEPTS_DIR = KNOWLEDGE_DIR / "concepts"
CONNECTIONS_DIR = KNOWLEDGE_DIR / "connections"
QA_DIR = KNOWLEDGE_DIR / "qa"
REPORTS_DIR = ROOT_DIR / "reports"
SCRIPTS_DIR = ROOT_DIR / "scripts"
HOOKS_DIR = ROOT_DIR / "hooks"
AGENTS_FILE = ROOT_DIR / "AGENTS.md"

INDEX_FILE = KNOWLEDGE_DIR / "index.md"
LOG_FILE = KNOWLEDGE_DIR / "log.md"
STATE_FILE = SCRIPTS_DIR / "state.json"
SOURCES_FILE = ROOT_DIR / "sources.yaml"

# ── Vector store ─────────────────────────────────────────────────────
CHROMA_DB_DIR = KNOWLEDGE_DIR / "chroma"
CHROMA_COLLECTION_ARTICLES = "articles"
CHROMA_COLLECTION_DAILY = "daily_chunks"

# Memory type taxonomy (Phase 3). Centralized here so vector_store,
# lint, and the knowledge MCP server share one source of truth.
MEMORY_TYPES = {
    "fact",          # static knowledge ("Stimulus controllers use kebab-case")
    "event",         # something that happened on a date ("launched feature X")
    "discovery",     # a finding from debugging/investigation
    "preference",    # user preference ("don't mock the DB in tests")
    "advice",        # actionable guidance ("always run Tailwind rebuild after CSS")
    "decision",      # locked-in architectural choice
}

# ── Timezone ───────────────────────────────────────────────────────────
TIMEZONE = "America/Chicago"

# ── Model selection ───────────────────────────────────────────────────
# Route cheap tasks to cheaper models. Override via env vars if needed.
# Flush is simple extraction → Haiku (cheapest, ~60% savings vs Sonnet)
# Compile/Ingest/Query are synthesis → Sonnet (good quality/cost balance)
import os
MODEL_FLUSH = os.environ.get("MEMORY_COMPILER_MODEL_FLUSH", "claude-haiku-4-5-20251001")
MODEL_COMPILE = os.environ.get("MEMORY_COMPILER_MODEL_COMPILE", "claude-sonnet-4-6")
MODEL_INGEST = os.environ.get("MEMORY_COMPILER_MODEL_INGEST", "claude-sonnet-4-6")
MODEL_QUERY = os.environ.get("MEMORY_COMPILER_MODEL_QUERY", "claude-sonnet-4-6")
MODEL_CANARY = os.environ.get("MEMORY_COMPILER_MODEL_CANARY", "claude-haiku-4-5-20251001")
MODEL_REWRITE = os.environ.get("MEMORY_COMPILER_MODEL_REWRITE", "claude-sonnet-4-6")
MODEL_EXPAND = os.environ.get("MEMORY_COMPILER_MODEL_EXPAND", "claude-haiku-4-5-20251001")
MODEL_CLEAN = os.environ.get("MEMORY_COMPILER_MODEL_CLEAN", "claude-haiku-4-5-20251001")

# ── Whisper (voice transcription) ─────────────────────────────────────
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")


def now_iso() -> str:
    """Current time in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def today_iso() -> str:
    """Current date in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
