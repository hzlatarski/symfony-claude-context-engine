"""Tests for the slugify_chunk_id helper used by the daily chunker + reindex."""
from __future__ import annotations


class TestSlugifyChunkId:
    def test_basic_section_title(self):
        from utils import slugify_chunk_id
        assert slugify_chunk_id("daily/2026-04-12.md", "Morning Standup") == "daily/2026-04-12.md#morning-standup"

    def test_section_with_punctuation(self):
        from utils import slugify_chunk_id
        assert (
            slugify_chunk_id("daily/2026-04-12.md", "Incident: Tailwind Build Failure!")
            == "daily/2026-04-12.md#incident-tailwind-build-failure"
        )

    def test_empty_section_title_falls_back(self):
        from utils import slugify_chunk_id
        assert slugify_chunk_id("daily/x.md", "") == "daily/x.md#section"

    def test_all_punctuation_section_falls_back(self):
        from utils import slugify_chunk_id
        assert slugify_chunk_id("daily/x.md", "!!!---???") == "daily/x.md#section"

    def test_stable_across_calls(self):
        from utils import slugify_chunk_id
        a = slugify_chunk_id("daily/2026-04-10.md", "Research")
        b = slugify_chunk_id("daily/2026-04-10.md", "Research")
        assert a == b

    def test_collapses_multiple_separators(self):
        from utils import slugify_chunk_id
        assert slugify_chunk_id("daily/x.md", "A   B_C--D") == "daily/x.md#a-b-c-d"
