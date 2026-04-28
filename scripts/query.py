"""
Query the knowledge base using hybrid retrieval.

Calls ``hybrid_search.search_articles`` (BM25 + vector via RRF) to pick
the top-N most relevant articles for the question, reads only those
articles, and asks the LLM to synthesize an answer from the bounded
context. Replaces the original "dump every article into the prompt"
strategy which broke once the KB grew past the model's context window.

Usage:
    uv run python query.py "How should I handle auth redirects?"
    uv run python query.py "What patterns do I use for API design?" --file-back
    uv run python query.py "..." --top-k 12   # widen the retrieval pool
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from config import KNOWLEDGE_DIR, MODEL_QUERY, QA_DIR, now_iso
from utils import load_state, read_wiki_index, save_state

ROOT_DIR = Path(__file__).resolve().parent.parent

# Default retrieval pool. Tuned so the prompt stays small enough for the
# Sonnet 200k context window with room for the index, system prompt,
# tool definitions, and the synthesized answer. Eight ~3k-token articles
# is roughly 24k context tokens for retrieved content — well within
# budget. Override with --top-k when the question needs more breadth.
DEFAULT_TOP_K = 8


def select_relevant_articles(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Pick the top-``top_k`` articles for ``question`` via hybrid search.

    Deduplicates by article slug — hybrid search returns one result per
    (slug, zone) pair, so an article whose Observed AND Synthesized
    zones both score high would otherwise consume two of the budget
    slots for the same content. Keeps the highest-RRF zone per slug.
    """
    import hybrid_search

    raw = hybrid_search.search_articles(question, limit=top_k * 2)
    seen: dict[str, dict] = {}
    for result in raw:
        slug = result["slug"]
        if slug not in seen:
            seen[slug] = result
    return list(seen.values())[:top_k]


def build_retrieved_context(selected: list[dict]) -> str:
    """Read each selected article's full content and concatenate.

    Uses the same ``## <slug>`` separator the legacy ``read_all_wiki_content``
    used so the LLM-facing format is unchanged. Articles that vanish
    between selection and read (e.g. a concurrent compile run) are
    silently skipped — no point failing the whole query on a race.
    """
    parts: list[str] = []
    for result in selected:
        slug = result["slug"]
        path = KNOWLEDGE_DIR / f"{slug}.md"
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        parts.append(f"## {slug}\n\n{content}")
    return "\n\n".join(parts)


async def run_query(
    question: str,
    file_back: bool = False,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """Query the knowledge base and optionally file the answer back."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    selected = select_relevant_articles(question, top_k=top_k)
    if not selected:
        return (
            "Hybrid retrieval returned zero results for this query. "
            "Either the knowledge base does not contain relevant content, "
            "or the BM25/vector indexes are stale — try running "
            "`reindex.py --all` and retry."
        )

    retrieved = build_retrieved_context(selected)
    index = read_wiki_index()
    selected_slugs = ", ".join(r["slug"] for r in selected)

    tools = ["Read", "Glob", "Grep"]
    if file_back:
        tools.extend(["Write", "Edit"])

    file_back_instructions = ""
    if file_back:
        timestamp = now_iso()
        file_back_instructions = f"""

## File Back Instructions

After answering, do the following:
1. Create a Q&A article at {QA_DIR}/ with the filename being a slugified version
   of the question (e.g., knowledge/qa/how-to-handle-auth-redirects.md)
2. Use the Q&A article format from the schema (frontmatter with title, question,
   consulted articles, filed date)
3. Update {KNOWLEDGE_DIR / 'index.md'} with a new row for this Q&A article
4. Append to {KNOWLEDGE_DIR / 'log.md'}:
   ## [{timestamp}] query (filed) | question summary
   - Question: {question}
   - Consulted: [[list of articles read]]
   - Filed to: [[qa/article-name]]
"""

    prompt = f"""You are a knowledge base query engine. Answer the user's question
using the retrieved articles below.

## How to Answer

1. The articles below were pre-selected by hybrid BM25 + vector search
   as the most relevant to the question. Read them carefully.
2. If you need more context, use the Read tool to fetch additional
   articles by slug (consult the INDEX section for what's available).
3. Synthesize a clear, thorough answer.
4. Cite your sources using [[wikilinks]] (e.g. [[concepts/supabase-auth]]).
5. If the retrieved articles don't contain relevant information, say so
   honestly rather than guessing.

## Pre-selected articles (top {len(selected)} by hybrid retrieval)

Slugs: {selected_slugs}

{retrieved}

## INDEX (full catalog — for fetching additional articles if needed)

{index}

## Question

{question}
{file_back_instructions}"""

    answer = ""
    cost = 0.0

    os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=str(ROOT_DIR),
                model=MODEL_QUERY,
                system_prompt={"type": "preset", "preset": "claude_code"},
                allowed_tools=tools,
                permission_mode="acceptEdits",
                max_turns=15,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        answer += block.text
            elif isinstance(message, ResultMessage):
                cost = message.total_cost_usd or 0.0
    except Exception as e:
        answer = f"Error querying knowledge base: {e}"

    # Update state
    state = load_state()
    state["query_count"] = state.get("query_count", 0) + 1
    state["total_cost"] = state.get("total_cost", 0.0) + cost

    # Increment access counts for every article we pre-selected — they
    # were the basis of the answer regardless of whether the LLM cited
    # them explicitly. This gives compile_truth.py better signal about
    # which articles are load-bearing for real questions.
    access_counts = state.setdefault("access_counts", {})
    for result in selected:
        slug = result["slug"]
        if slug.startswith("daily/"):
            continue
        access_counts[slug] = access_counts.get(slug, 0) + 1

    save_state(state)

    return answer


def main():
    parser = argparse.ArgumentParser(description="Query the personal knowledge base")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument(
        "--file-back",
        action="store_true",
        help="File the answer back into the knowledge base as a Q&A article",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            f"Number of articles to retrieve via hybrid search "
            f"(default {DEFAULT_TOP_K}). Increase for breadth-heavy "
            f"questions, decrease for tight context budgets."
        ),
    )
    args = parser.parse_args()

    print(f"Question: {args.question}")
    print(f"Top-K: {args.top_k}")
    print(f"File back: {'yes' if args.file_back else 'no'}")
    print("-" * 60)

    answer = asyncio.run(
        run_query(args.question, file_back=args.file_back, top_k=args.top_k)
    )
    print(answer)

    if args.file_back:
        print("\n" + "-" * 60)
        qa_count = len(list(QA_DIR.glob("*.md"))) if QA_DIR.exists() else 0
        print(f"Answer filed to knowledge/qa/ ({qa_count} Q&A articles total)")


if __name__ == "__main__":
    main()
