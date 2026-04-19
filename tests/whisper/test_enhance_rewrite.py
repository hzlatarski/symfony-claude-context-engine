"""Unit tests for whisper.enhance.enhance_rewrite and anchor verification."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from whisper.types import Hit


def _mock_response(text: str):
    return SimpleNamespace(content=[SimpleNamespace(text=text, type="text")])


def _hit(cid: str, source: str, path: str, category=None):
    return Hit(
        id=cid, source=source, category=category,
        path=path, title=path, snippet="snippet",
        full_body=f"full body for {path}",
        score=1.0, symbols=[], metadata={},
    )


@pytest.fixture
def mock_client(monkeypatch):
    client = MagicMock()
    import whisper.enhance as e
    e._get_client.cache_clear()
    monkeypatch.setattr(e, "_get_client", lambda: client)
    return client


def test_rewrite_passes_transcript_and_context_in_user_message(mock_client):
    mock_client.messages.create.return_value = _mock_response("REWRITTEN")
    from whisper.enhance import enhance_rewrite

    hits = [
        _hit("c1", "article", "concepts/s3-migration", category="captured-memory"),
        _hit("c2", "code", "src/Service/Foo.php:1-20"),
    ]

    enhance_rewrite("audit the S3 migration", hits)

    _args, kwargs = mock_client.messages.create.call_args
    user_msg = kwargs["messages"][0]["content"]
    assert "audit the S3 migration" in user_msg
    assert "concepts/s3-migration" in user_msg
    assert "src/Service/Foo.php:1-20" in user_msg


def test_rewrite_uses_sonnet_model_from_config(mock_client):
    mock_client.messages.create.return_value = _mock_response("out")
    import config
    from whisper.enhance import enhance_rewrite

    enhance_rewrite("x", [_hit("c1", "article", "slug")])

    _args, kwargs = mock_client.messages.create.call_args
    assert kwargs["model"] == config.MODEL_REWRITE


def test_verify_anchors_strips_hallucinated_paths():
    from whisper.enhance import verify_anchors

    hits = [_hit("c1", "article", "concepts/real-slug")]
    rewritten = (
        "Task 1: update [src:concepts/real-slug] "
        "Task 2: also touch [src:src/Invented/Fake.php]"
    )

    cleaned, warnings = verify_anchors(rewritten, hits)

    assert "[src:concepts/real-slug]" in cleaned
    assert "[src:src/Invented/Fake.php]" not in cleaned
    assert any("src/Invented/Fake.php" in w for w in warnings)


def test_verify_anchors_preserves_line_ranges_on_code_paths():
    from whisper.enhance import verify_anchors

    hits = [_hit("c1", "code", "src/Service/Foo.php:1-20")]
    # LLM might write the anchor with or without the line range — both should be accepted
    rewritten = "See [src:src/Service/Foo.php] and [src:src/Service/Foo.php:1-20]"

    cleaned, warnings = verify_anchors(rewritten, hits)

    assert "[src:src/Service/Foo.php]" in cleaned
    assert "[src:src/Service/Foo.php:1-20]" in cleaned
    assert warnings == []


def test_verify_anchors_no_anchors_produces_no_warnings():
    from whisper.enhance import verify_anchors

    hits = [_hit("c1", "article", "concepts/foo")]
    rewritten = "plain text with no anchors"

    cleaned, warnings = verify_anchors(rewritten, hits)

    assert cleaned == rewritten
    assert warnings == []


def test_rewrite_with_zero_hits_raises(mock_client):
    from whisper.enhance import enhance_rewrite, EnhanceError

    with pytest.raises(EnhanceError):
        enhance_rewrite("transcript", [])


def test_rewrite_integrates_verification_and_reports_warnings(mock_client):
    mock_client.messages.create.return_value = _mock_response(
        "Do [src:concepts/real] but also [src:src/Fake.php]"
    )
    from whisper.enhance import enhance_rewrite

    hits = [_hit("c1", "article", "concepts/real")]
    result = enhance_rewrite("do stuff", hits)

    assert "[src:src/Fake.php]" not in result.prompt
    assert "[src:concepts/real]" in result.prompt
    assert any("src/Fake.php" in w for w in result.warnings)
