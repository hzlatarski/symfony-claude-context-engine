"""Tests for transcript path resolution.

The distinction that matters: a *missing* transcript is an anomaly worth an
error, while a *sidechain* session — a subagent, which Claude Code gives a
session id and a directory but no standalone transcript — is normal and must
not be logged as a failure.
"""
import sys

import pytest

from scripts.transcript import (
    TRANSCRIPT_MISSING,
    TRANSCRIPT_OK,
    TRANSCRIPT_SIDECHAIN,
    resolve_transcript,
)


def test_existing_transcript_resolves_ok(tmp_path):
    path = tmp_path / "sess.jsonl"
    path.write_text("{}", encoding="utf-8")
    resolved, status = resolve_transcript(str(path))
    assert status == TRANSCRIPT_OK
    assert resolved == path


def test_session_directory_without_transcript_is_sidechain(tmp_path):
    (tmp_path / "sess").mkdir()
    resolved, status = resolve_transcript(str(tmp_path / "sess.jsonl"))
    assert status == TRANSCRIPT_SIDECHAIN
    assert resolved is None


def test_neither_file_nor_directory_is_missing(tmp_path):
    resolved, status = resolve_transcript(str(tmp_path / "nope.jsonl"))
    assert status == TRANSCRIPT_MISSING
    assert resolved is None


def test_a_real_transcript_wins_over_the_sibling_directory(tmp_path):
    """Real sessions have *both*. That must resolve to the transcript."""
    (tmp_path / "sess").mkdir()
    path = tmp_path / "sess.jsonl"
    path.write_text("{}", encoding="utf-8")
    resolved, status = resolve_transcript(str(path))
    assert status == TRANSCRIPT_OK
    assert resolved == path


@pytest.mark.parametrize("value", ["", "   ", None, 123, []])
def test_empty_or_non_string_input_is_missing(value):
    resolved, status = resolve_transcript(value)
    assert (resolved, status) == (None, TRANSCRIPT_MISSING)


@pytest.mark.skipif(sys.platform != "win32", reason="drive-letter casing is Windows-only")
def test_drive_letter_case_is_normalized(tmp_path):
    """Claude Code sometimes emits C:\\... where the path is c:\\..."""
    path = tmp_path / "sess.jsonl"
    path.write_text("{}", encoding="utf-8")
    swapped = str(path)[0].swapcase() + str(path)[1:]
    resolved, status = resolve_transcript(swapped)
    assert status == TRANSCRIPT_OK
    assert resolved is not None


@pytest.mark.skipif(sys.platform != "win32", reason="drive-letter casing is Windows-only")
def test_sidechain_is_detected_through_a_swapped_drive_letter(tmp_path):
    (tmp_path / "sess").mkdir()
    original = str(tmp_path / "sess.jsonl")
    swapped = original[0].swapcase() + original[1:]
    _resolved, status = resolve_transcript(swapped)
    assert status == TRANSCRIPT_SIDECHAIN
