"""Tests for the daily-log chunker — H2-bounded verbatim extraction."""
from __future__ import annotations

from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "daily" / "2026-04-10.md"


class TestChunkDailyLog:
    def test_yields_one_chunk_per_h2(self):
        from chunk_daily import chunk_daily_log
        content = FIXTURE.read_text(encoding="utf-8")
        chunks = list(chunk_daily_log(content, source_rel="daily/2026-04-10.md"))
        titles = [c.section for c in chunks]
        assert "Morning Standup" in titles
        assert "Incident: Build Failure" in titles
        assert "Research" in titles
        assert len(chunks) == 3

    def test_skips_frontmatter_and_h1(self):
        from chunk_daily import chunk_daily_log
        content = FIXTURE.read_text(encoding="utf-8")
        chunks = list(chunk_daily_log(content, source_rel="daily/2026-04-10.md"))
        for c in chunks:
            assert "date: 2026-04-10" not in c.text
            assert not c.text.startswith("# Daily Log")

    def test_stable_chunk_ids(self):
        from chunk_daily import chunk_daily_log
        content = FIXTURE.read_text(encoding="utf-8")
        ids_a = [c.id for c in chunk_daily_log(content, source_rel="daily/2026-04-10.md")]
        ids_b = [c.id for c in chunk_daily_log(content, source_rel="daily/2026-04-10.md")]
        assert ids_a == ids_b
        assert all(i.startswith("daily/2026-04-10.md#") for i in ids_a)

    def test_chunk_ids_use_slugify_chunk_id_format(self):
        from chunk_daily import chunk_daily_log
        content = FIXTURE.read_text(encoding="utf-8")
        chunks = {c.section: c.id for c in chunk_daily_log(content, source_rel="daily/2026-04-10.md")}
        assert chunks["Morning Standup"] == "daily/2026-04-10.md#morning-standup"
        assert chunks["Incident: Build Failure"] == "daily/2026-04-10.md#incident-build-failure"
        assert chunks["Research"] == "daily/2026-04-10.md#research"

    def test_empty_section_dropped(self):
        from chunk_daily import chunk_daily_log
        content = "## Empty\n\n## Real\n\nactual content\n"
        chunks = list(chunk_daily_log(content, source_rel="daily/x.md"))
        sections = [c.section for c in chunks]
        assert "Real" in sections
        assert "Empty" not in sections

    def test_preserves_section_body_verbatim(self):
        from chunk_daily import chunk_daily_log
        content = FIXTURE.read_text(encoding="utf-8")
        chunks = {c.section: c.text for c in chunk_daily_log(content, source_rel="daily/2026-04-10.md")}
        assert "framework A" in chunks["Research"]
        assert "framework B" in chunks["Research"]
        # The H2 header is preserved at the start of each chunk so the
        # retrieved text carries its own context.
        assert chunks["Research"].startswith("## Research")

    def test_no_h2_sections_yields_nothing(self):
        from chunk_daily import chunk_daily_log
        content = "just some paragraph\n\nwith no headings at all\n"
        chunks = list(chunk_daily_log(content, source_rel="daily/x.md"))
        assert chunks == []

    def test_h1_is_not_a_section(self):
        from chunk_daily import chunk_daily_log
        content = "# Top Level Title\n\nSome body\n\n## Real Section\n\ncontent\n"
        chunks = list(chunk_daily_log(content, source_rel="daily/x.md"))
        assert len(chunks) == 1
        assert chunks[0].section == "Real Section"

    def test_no_frontmatter_still_works(self):
        from chunk_daily import chunk_daily_log
        content = "## Section One\n\nbody A\n\n## Section Two\n\nbody B\n"
        chunks = list(chunk_daily_log(content, source_rel="daily/x.md"))
        assert [c.section for c in chunks] == ["Section One", "Section Two"]

    def test_h3_sections_become_chunks(self):
        """Real daily logs nest H3 sessions under H2 containers.

        The chunker must split on H3 so each ``### Session (14:08)``
        becomes a retrievable chunk rather than being absorbed into a
        mega-chunk under ``## Sessions``.
        """
        from chunk_daily import chunk_daily_log
        content = (
            "# Daily Log\n\n"
            "## Sessions\n\n"
            "### Session (14:08)\n\nfirst session content\n\n"
            "### Session (15:30)\n\nsecond session content\n\n"
            "## Memory Maintenance\n\n"
            "### Memory Flush (16:00)\n\nflush details\n"
        )
        chunks = list(chunk_daily_log(content, source_rel="daily/2026-04-12.md"))
        titles = [c.section for c in chunks]
        assert titles == [
            "Session (14:08)",
            "Session (15:30)",
            "Memory Flush (16:00)",
        ]
        # Container H2s with only H3 children should be dropped
        assert "Sessions" not in titles
        assert "Memory Maintenance" not in titles

    def test_duplicate_section_titles_get_unique_ids(self):
        from chunk_daily import chunk_daily_log
        content = (
            "## Research\n\nfirst research block\n\n"
            "## Research\n\nsecond research block\n\n"
            "## Research\n\nthird research block\n"
        )
        chunks = list(chunk_daily_log(content, source_rel="daily/x.md"))
        ids = [c.id for c in chunks]
        assert ids == [
            "daily/x.md#research",
            "daily/x.md#research-2",
            "daily/x.md#research-3",
        ]
        # All three sections still expose the same human-readable title
        assert all(c.section == "Research" for c in chunks)

    def test_h2_container_with_text_and_h3_children_yields_both(self):
        """An H2 that has its own non-H3 text AND H3 children should
        emit both — the H2's own intro (body before the first H3) as
        one chunk, and each H3 as its own chunk."""
        from chunk_daily import chunk_daily_log
        content = (
            "## Research\n\nIntro paragraph under Research.\n\n"
            "### Finding A\n\ndetail A\n\n"
            "### Finding B\n\ndetail B\n"
        )
        chunks = list(chunk_daily_log(content, source_rel="daily/x.md"))
        titles = [c.section for c in chunks]
        assert titles == ["Research", "Finding A", "Finding B"]
