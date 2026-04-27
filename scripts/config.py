"""Path constants and configuration for the personal knowledge base."""

from pathlib import Path
from datetime import datetime, timezone

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT_DIR.parent.parent  # Outer Symfony project root (outside .claude/)

# Load .env.local from the Symfony project root so ANTHROPIC_API_KEY etc. are
# available when running standalone scripts (viewer.py, canary.py, query.py).
try:
    from dotenv import load_dotenv
    _env_local = PROJECT_ROOT / ".env.local"
    if _env_local.exists():
        load_dotenv(_env_local, override=False)
except ImportError:
    pass
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
CHROMA_COLLECTION_CODEBASE = "codebase"

# Cross-process file locks for Chroma writes. One lock file per collection
# under knowledge/chroma/.locks/. The directory is created on first use.
CHROMA_LOCKS_DIR = CHROMA_DB_DIR / ".locks"
CHROMA_LOCK_TIMEOUT_SECONDS = 60.0  # Wait this long for a busy lock before failing.

# ── Ingest pipeline state ────────────────────────────────────────────
# Per-file checkpoints make ingest.py crash-safe: on rerun, files whose
# content hash matches the checkpoint are skipped without re-billing
# Sonnet. Status / stop coordinate live progress reporting and cooperative
# cancellation from MCP tools.
INGEST_CHECKPOINT_FILE = KNOWLEDGE_DIR / ".ingest-checkpoint.json"
INGEST_STATUS_FILE = KNOWLEDGE_DIR / ".ingest-status.json"
INGEST_STOP_FILE = KNOWLEDGE_DIR / ".ingest-stop"

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

# ── Cross-project linked search ───────────────────────────────────────
# Comma-separated absolute paths to other Symfony project roots whose
# `.claude/memory-compiler/` directories should be searchable from this
# project. Used by hybrid_search.search_articles_linked() to fan out a
# single search_knowledge call across multiple project knowledge bases.
# Non-existent paths are silently skipped at search time.
def _parse_linked_projects() -> list[Path]:
    raw = os.environ.get("MEMORY_COMPILER_LINKED_PROJECTS", "").strip()
    if not raw:
        return []
    out: list[Path] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(Path(part).expanduser().resolve())
    return out


LINKED_PROJECTS: list[Path] = _parse_linked_projects()


# ── Custom file extensions ────────────────────────────────────────────
# Comma-separated list of additional file extensions to scan during
# codebase indexing. Useful for projects with non-standard extensions
# (e.g. `.dist`, `.neon`, `.tpl`). Each extension must include the
# leading dot. Files indexed via this hook fall through to line-based
# chunking — AST chunking is reserved for known languages.
def _parse_extra_extensions() -> tuple[str, ...]:
    raw = os.environ.get("MEMORY_COMPILER_EXTRA_EXTENSIONS", "").strip()
    if not raw:
        return ()
    out: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.startswith("."):
            part = "." + part
        out.append(part.lower())
    return tuple(out)


EXTRA_EXTENSIONS: tuple[str, ...] = _parse_extra_extensions()


def now_iso() -> str:
    """Current time in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def today_iso() -> str:
    """Current date in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
