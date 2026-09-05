"""Tests for the entity extraction pass (Slice 1).

The extractor lifts recurring NAMED THINGS (hosts, project commands,
services, env vars, URLs, source paths) out of article prose. It is a
pure-Python, zero-LLM, report-only pass — it never touches the graph or any
article file. We test the pure functions against synthetic strings and
tmp_path; no dependence on live project state, network, or Chroma.
"""
from scripts import entities


def _ids(body: str) -> set[str]:
    return {e.canonical_id for e in entities.extract(body)}


def _by_type(body: str, t: str) -> list[str]:
    return [e.canonical_id for e in entities.extract(body) if e.type == t]


# ── host (IPv4) ─────────────────────────────────────────────────────────

def test_host_extracts_valid_ipv4():
    assert _by_type("Prod is at 65.21.4.203 over SSH.", "host") == [
        "entity:host/65.21.4.203"
    ]


def test_host_rejects_out_of_range_octet():
    # 400 > 255 → not a host. (Also guards against version-like tokens.)
    assert _by_type("build 1.2.3.400 failed", "host") == []


def test_host_ignores_three_segment_version():
    assert _by_type("MariaDB 11.7.2 shipped", "host") == []


# ── command (app: namespace) ────────────────────────────────────────────

def test_command_extracts_app_namespace():
    assert _by_type("Run app:content:narrate to voice it.", "command") == [
        "entity:command/app:content:narrate"
    ]


def test_command_matches_hyphenated_segment():
    assert _by_type("Then app:images:push-to-cdn publishes.", "command") == [
        "entity:command/app:images:push-to-cdn"
    ]


def test_command_ignores_bare_app_prefix():
    assert _by_type("the app: prefix is reserved", "command") == []


# ── envvar (SCREAMING_SNAKE) ────────────────────────────────────────────

def test_envvar_extracts_multi_segment():
    assert _by_type("Never set ANTHROPIC_API_KEY here.", "envvar") == [
        "entity:envvar/ANTHROPIC_API_KEY"
    ]


def test_envvar_ignores_single_word_allcaps():
    # No underscore → not an env var (avoids TODO / WHY / HACK tags).
    assert _by_type("This is a TODO and a HACK.", "envvar") == []


def test_envvar_requires_known_suffix():
    # SCREAMING_SNAKE constants that are not env vars are rejected by the
    # suffix allowlist — this was the noisy case the dry-run surfaced.
    assert _by_type("STATUS_DISABLED and DEFAULT_RATES are constants.", "envvar") == []


def test_envvar_accepts_various_env_suffixes():
    ids = _by_type("Set MAILER_DSN and MERCURE_PUBLIC_URL and CLAUDE_BINARY_PATH.", "envvar")
    assert ids == [
        "entity:envvar/MAILER_DSN",
        "entity:envvar/MERCURE_PUBLIC_URL",
        "entity:envvar/CLAUDE_BINARY_PATH",
    ]


# ── role (Symfony security roles) ───────────────────────────────────────

def test_role_extracts_role_constants():
    ids = _by_type("Requires ROLE_ADMIN, not ROLE_CORPORATE_MANAGER.", "role")
    assert ids == ["entity:role/ROLE_ADMIN", "entity:role/ROLE_CORPORATE_MANAGER"]


def test_role_is_not_also_an_envvar():
    # ROLE_ADMIN must be a role only — never double-counted as an env var.
    assert _by_type("Grant ROLE_ADMIN here.", "envvar") == []
    assert _by_type("Grant ROLE_ADMIN here.", "role") == ["entity:role/ROLE_ADMIN"]


# ── url ─────────────────────────────────────────────────────────────────

def test_url_normalizes_scheme_and_trailing_punct():
    assert _by_type("See https://127.0.0.1:8001/arena.", "url") == [
        "entity:url/127.0.0.1:8001/arena"
    ]


def test_url_matches_websocket_scheme():
    assert _by_type("Bridge speaks wss://voice.local/ws here.", "url") == [
        "entity:url/voice.local/ws"
    ]


# ── path ────────────────────────────────────────────────────────────────

def test_path_extracts_src_file():
    assert _by_type("Edit src/Service/KokoroTtsService.php now.", "path") == [
        "entity:path/src/Service/KokoroTtsService.php"
    ]


def test_path_rejects_bare_top_dir():
    # "src/" alone (no file, no second segment) is not a path entity.
    assert _by_type("everything under src/ is domain code", "path") == []


# ── service (PascalCase class-name convention) ──────────────────────────

def test_service_extracts_suffix_classes():
    ids = _by_type("HybridEmailValidationService calls the UserRepository.", "service")
    assert "entity:service/HybridEmailValidationService" in ids
    assert "entity:service/UserRepository" in ids


# ── masking: no entity inside code / anchors / links ────────────────────

def test_extract_ignores_fenced_code():
    body = "before\n```\nrun app:content:narrate at 65.21.4.203\n```\nafter"
    assert entities.extract(body) == []


def test_extract_ignores_inline_code():
    assert _ids("call `app:content:narrate` in a snippet") == set()


def test_extract_ignores_frontmatter():
    body = "---\nhost: 65.21.4.203\n---\nplain prose with no entities"
    assert entities.extract(body) == []


def test_extract_ignores_src_anchor_paths():
    # [src:...] anchors already become file: nodes; don't double-count them.
    body = "Fact about grading [src:src/Service/GradeSessionHandler.php]."
    assert _by_type(body, "path") == []


def test_extract_ignores_markdown_link_target():
    assert _ids("see [the docs](https://example.com/guide) for more") == set()


# ── canonicalization + de-duplication ───────────────────────────────────

def test_two_surfaces_collapse_to_one_canonical_id():
    # The cross-referencing property: different surface forms → one node id.
    body = "Deploy to 65.21.4.203, then verify prod (65.21.4.203) is live."
    hosts = _by_type(body, "host")
    assert hosts == ["entity:host/65.21.4.203"]


def test_repeated_mention_yields_single_entity():
    body = "app:content:set here, app:content:set there, app:content:set again."
    assert _by_type(body, "command") == ["entity:command/app:content:set"]


def test_extract_orders_by_first_appearance():
    body = "First app:content:narrate then later ANTHROPIC_API_KEY."
    ordered = [e.canonical_id for e in entities.extract(body)]
    assert ordered.index("entity:command/app:content:narrate") < ordered.index(
        "entity:envvar/ANTHROPIC_API_KEY"
    )


# ── cross-KB aggregation ────────────────────────────────────────────────

def test_collect_groups_articles_by_shared_entity(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    (concepts / "a.md").write_text(
        "---\ntitle: A\n---\n## Truth\n\nProd host 65.21.4.203 runs it.",
        encoding="utf-8",
    )
    (concepts / "b.md").write_text(
        "---\ntitle: B\n---\n## Truth\n\nSSH into 65.21.4.203 for logs.",
        encoding="utf-8",
    )
    by_entity = entities.collect(tmp_path)
    host = by_entity["entity:host/65.21.4.203"]
    assert host["slugs"] == {"concepts/a", "concepts/b"}
