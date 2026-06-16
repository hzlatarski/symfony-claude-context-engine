"""Tests for the release helper's pure functions.

We exercise the version-bump and CHANGELOG-insertion logic directly — the git
plumbing in main() is thin subprocess orchestration and is covered by the
manual --dry-run smoke test, not here.
"""
import pytest

from scripts import release


@pytest.mark.parametrize("current,level,expected", [
    ("0.1.0", "patch", "0.1.1"),
    ("0.1.0", "minor", "0.2.0"),
    ("0.1.0", "major", "1.0.0"),
    ("1.2.3", "patch", "1.2.4"),
    ("1.2.3", "minor", "1.3.0"),
    ("1.2.3", "major", "2.0.0"),
    ("0.1", "minor", "0.2.0"),       # short version is normalised to 3 segments
    ("2", "patch", "2.0.1"),
])
def test_bump_version(current, level, expected):
    assert release.bump_version(current, level) == expected


def test_bump_version_rejects_non_numeric():
    with pytest.raises(ValueError):
        release.bump_version("1.x.0", "patch")


def test_bump_version_rejects_unknown_level():
    with pytest.raises(ValueError):
        release.bump_version("1.0.0", "mega")


def test_is_valid_version():
    assert release.is_valid_version("1.0.0")
    assert release.is_valid_version("0.2")
    assert not release.is_valid_version("1.0.0-rc1")
    assert not release.is_valid_version("v1.0.0")


def test_build_changelog_section_with_bullets():
    section = release.build_changelog_section(
        "0.2.0", "2026-06-16", ["feat: a thing", "fix: a bug"]
    )
    assert section.startswith("## [0.2.0] — 2026-06-16")
    assert "### Changed" in section
    assert "- feat: a thing" in section
    assert "- fix: a bug" in section


def test_build_changelog_section_without_bullets_is_stub():
    section = release.build_changelog_section("0.2.0", "2026-06-16", [])
    assert "## [0.2.0]" in section
    assert "### Changed" not in section
    assert "Describe this release" in section


def test_insert_changelog_section_goes_above_latest_entry():
    changelog = (
        "# Changelog\n\nPreamble line.\n\n"
        "## [0.1.0] — 2026-04-27\n\nFirst release.\n"
    )
    section = release.build_changelog_section("0.2.0", "2026-06-16", ["feat: x"])
    result = release.insert_changelog_section(changelog, section)
    # New section appears, and appears BEFORE the old one.
    assert "## [0.2.0]" in result
    assert result.index("## [0.2.0]") < result.index("## [0.1.0]")
    # Preamble is preserved ahead of both.
    assert result.index("Preamble line.") < result.index("## [0.2.0]")


def test_insert_changelog_section_appends_when_no_prior_entry():
    changelog = "# Changelog\n\nNothing released yet.\n"
    section = release.build_changelog_section("0.1.0", "2026-06-16", [])
    result = release.insert_changelog_section(changelog, section)
    assert "## [0.1.0]" in result
    assert result.index("Nothing released yet.") < result.index("## [0.1.0]")
