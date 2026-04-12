"""Split daily logs into section-bounded chunks for verbatim retrieval.

Each ``## H2`` or ``### H3`` section of a daily log becomes one chunk.
Frontmatter and the file's ``# H1`` title are dropped so the chunks
don't carry ambient metadata the retrieval layer doesn't need. The
section header itself stays at the top of each chunk so retrieved
text is self-contextualizing.

Both H2 and H3 are treated as split points because the memory-compiler's
real daily logs follow a nested structure: two H2 *containers*
(``## Sessions`` and ``## Memory Maintenance``) with all the actual
session content under ``### Session (HH:MM)`` and
``### Memory Flush (HH:MM)`` H3 subsections. Splitting only on H2
would produce two mega-chunks per daily log — too coarse for useful
retrieval. Splitting on H3 gives one chunk per real session event.

Design notes
------------
- Chunk ids come from ``utils.slugify_chunk_id`` so the same section
  title produces the same id on every run (hash-cache-friendly). If
  two sections share an exact title inside one file, a numeric suffix
  keeps the ids distinct so neither chunk silently overwrites the other
  during upsert.
- Empty sections (header with whitespace-only body, or H2 containers
  whose only content is H3 children) are dropped. This avoids poisoning
  Chroma with near-zero-content chunks that would otherwise match
  everything with mediocre distance.
- We do NOT chunk paragraphs, sentences, or by character count. Author
  headings are the natural unit. Oversized sections should be split in
  the source file, not by the chunker.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from utils import slugify_chunk_id


@dataclass
class DailyChunk:
    id: str
    section: str
    text: str


_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
# Match H2 or H3 headings at start of line.
_SECTION_RE = re.compile(r"^(#{2,3}) +(.+?)\s*$", re.MULTILINE)


def chunk_daily_log(content: str, source_rel: str) -> Iterator[DailyChunk]:
    """Yield one ``DailyChunk`` per H2 or H3 section.

    Frontmatter and ``# H1`` are dropped. The section header line is
    preserved at the top of each chunk's text so the model sees the
    section title in the retrieved window.

    A section whose body is empty (or whitespace-only) below the header
    is skipped — including H2 containers whose only contents are H3
    children, since the H3s get their own chunks.

    Duplicate section titles inside one file get a numeric suffix
    (``#section-2``, ``#section-3``...) so upserts don't collide.
    """
    body = _FRONTMATTER_RE.sub("", content, count=1)

    matches = list(_SECTION_RE.finditer(body))
    if not matches:
        return

    seen_ids: dict[str, int] = {}

    for i, match in enumerate(matches):
        section_title = match.group(2).strip()
        start = match.start()
        header_end = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)

        body_below = body[header_end:end].strip()
        if not body_below:
            continue

        text = body[start:end].strip()
        base_id = slugify_chunk_id(source_rel, section_title)
        count = seen_ids.get(base_id, 0)
        seen_ids[base_id] = count + 1
        chunk_id = base_id if count == 0 else f"{base_id}-{count + 1}"

        yield DailyChunk(id=chunk_id, section=section_title, text=text)
