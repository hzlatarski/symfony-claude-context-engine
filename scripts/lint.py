"""
Lint the knowledge base for structural and semantic health.

Runs 7 checks: broken links, orphan pages, orphan sources, stale articles,
contradictions (LLM), missing backlinks, and sparse articles.

Usage:
    uv run python lint.py                    # all checks
    uv run python lint.py --structural-only  # skip LLM checks (faster, cheaper)
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from config import KNOWLEDGE_DIR, REPORTS_DIR, now_iso, today_iso
from utils import (
    count_inbound_links,
    extract_wikilinks,
    file_hash,
    get_article_word_count,
    list_raw_files,
    list_wiki_articles,
    load_state,
    read_all_wiki_content,
    save_contradictions,
    save_state,
    wiki_article_exists,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

# Valid slug prefixes for the contradiction quarantine. Any slug extracted
# from the LLM's output that doesn't start with one of these is dropped —
# this is what prevents the LLM hallucinating "compiled-truth" or "daily/foo"
# from corrupting the quarantine set.
VALID_SLUG_PREFIXES = ("concepts/", "connections/", "qa/")


def check_broken_links() -> list[dict]:
    """Check for [[wikilinks]] that point to non-existent articles."""
    issues = []
    for article in list_wiki_articles():
        content = article.read_text(encoding="utf-8")
        rel = article.relative_to(KNOWLEDGE_DIR)
        for link in extract_wikilinks(content):
            if link.startswith("daily/"):
                continue  # daily log references are valid
            if not wiki_article_exists(link):
                issues.append({
                    "severity": "error",
                    "check": "broken_link",
                    "file": str(rel),
                    "detail": f"Broken link: [[{link}]] - target does not exist",
                })
    return issues


def check_orphan_pages() -> list[dict]:
    """Check for articles with zero inbound links."""
    issues = []
    for article in list_wiki_articles():
        rel = article.relative_to(KNOWLEDGE_DIR)
        link_target = str(rel).replace(".md", "").replace("\\", "/")
        inbound = count_inbound_links(link_target)
        if inbound == 0:
            issues.append({
                "severity": "warning",
                "check": "orphan_page",
                "file": str(rel),
                "detail": f"Orphan page: no other articles link to [[{link_target}]]",
            })
    return issues


def check_orphan_sources() -> list[dict]:
    """Check for daily logs that haven't been compiled yet."""
    state = load_state()
    ingested = state.get("ingested_daily", state.get("ingested", {}))
    issues = []
    for log_path in list_raw_files():
        if log_path.name not in ingested:
            issues.append({
                "severity": "warning",
                "check": "orphan_source",
                "file": f"daily/{log_path.name}",
                "detail": f"Uncompiled daily log: {log_path.name} has not been ingested",
            })
    return issues


def check_stale_articles() -> list[dict]:
    """Check if source daily logs have changed since compilation."""
    state = load_state()
    ingested = state.get("ingested_daily", state.get("ingested", {}))
    issues = []
    for log_path in list_raw_files():
        rel = log_path.name
        if rel in ingested:
            stored_hash = ingested[rel].get("hash", "")
            current_hash = file_hash(log_path)
            if stored_hash != current_hash:
                issues.append({
                    "severity": "warning",
                    "check": "stale_article",
                    "file": f"daily/{rel}",
                    "detail": f"Stale: {rel} has changed since last compilation",
                })
    return issues


def check_missing_backlinks() -> list[dict]:
    """Check for asymmetric links: A links to B but B doesn't link to A."""
    issues = []
    for article in list_wiki_articles():
        content = article.read_text(encoding="utf-8")
        rel = article.relative_to(KNOWLEDGE_DIR)
        source_link = str(rel).replace(".md", "").replace("\\", "/")

        for link in extract_wikilinks(content):
            if link.startswith("daily/"):
                continue
            target_path = KNOWLEDGE_DIR / f"{link}.md"
            if target_path.exists():
                target_content = target_path.read_text(encoding="utf-8")
                if f"[[{source_link}]]" not in target_content:
                    issues.append({
                        "severity": "suggestion",
                        "check": "missing_backlink",
                        "file": str(rel),
                        "detail": f"[[{source_link}]] links to [[{link}]] but not vice versa",
                        "auto_fixable": True,
                    })
    return issues


def check_sparse_articles() -> list[dict]:
    """Check for articles with fewer than 200 words."""
    issues = []
    for article in list_wiki_articles():
        word_count = get_article_word_count(article)
        if word_count < 200:
            rel = article.relative_to(KNOWLEDGE_DIR)
            issues.append({
                "severity": "suggestion",
                "check": "sparse_article",
                "file": str(rel),
                "detail": f"Sparse article: {word_count} words (minimum recommended: 200)",
            })
    return issues


def check_source_anchors() -> list[dict]:
    """Verify that [src:...] anchors in articles point to existing files.

    Articles without any anchors get a "suggestion" (non-blocking) so that
    legacy articles continue to work until they migrate organically. Broken
    anchors (file doesn't exist) get an "error" because they are falsifiable
    claims about source existence.
    """
    from utils import extract_source_anchors, verify_source_anchor

    issues = []
    for article in list_wiki_articles():
        content = article.read_text(encoding="utf-8")
        rel = article.relative_to(KNOWLEDGE_DIR)
        anchors = extract_source_anchors(content)

        if not anchors:
            issues.append({
                "severity": "suggestion",
                "check": "missing_source_anchor",
                "file": str(rel),
                "detail": "No [src:...] anchors found - Truth bullets cannot be re-verified",
            })
            continue

        for anchor in anchors:
            if not verify_source_anchor(anchor):
                issues.append({
                    "severity": "error",
                    "check": "broken_source_anchor",
                    "file": str(rel),
                    "detail": f"Broken anchor: [src:{anchor}] - file does not exist",
                })

    return issues


def check_memory_types() -> list[dict]:
    """Validate that every article's ``type:`` frontmatter is in MEMORY_TYPES.

    The type taxonomy (fact / event / discovery / preference / advice /
    decision) is a first-class filter in the knowledge MCP server's
    ``search_knowledge`` tool. Unknown values break that filter and are
    treated as an error. Missing values get a "suggestion" so legacy
    articles keep working while Task 8's compile prompt update and a
    gradual migration backfill the field.
    """
    from compile_truth import parse_frontmatter
    from config import MEMORY_TYPES

    issues = []
    for article in list_wiki_articles():
        content = article.read_text(encoding="utf-8")
        rel = article.relative_to(KNOWLEDGE_DIR)
        fm = parse_frontmatter(content)
        t = fm.get("type")

        if t is None:
            issues.append({
                "severity": "suggestion",
                "check": "missing_memory_type",
                "file": str(rel),
                "detail": (
                    f"No type: field. Add one of {sorted(MEMORY_TYPES)} so "
                    f"search_knowledge type_filter queries can find this article."
                ),
            })
            continue

        if t not in MEMORY_TYPES:
            issues.append({
                "severity": "error",
                "check": "invalid_memory_type",
                "file": str(rel),
                "detail": f"type: {t!r} is not one of {sorted(MEMORY_TYPES)}",
            })

    return issues


def check_low_priority_articles() -> list[dict]:
    """Check for articles that score below the inclusion threshold.

    These are articles that won't make it into compiled-truth.md under the
    default budget. Flags articles that are both old (>90 days) and never
    accessed, as candidates for consolidation or archiving.
    """
    from compile_truth import (
        DEFAULT_BUDGET_CHARS,
        ScoredArticle,
        build_inbound_link_map,
        extract_fallback_truth,
        extract_truth_section,
        parse_frontmatter,
        score_recency,
        score_linkedness,
        score_access,
        compute_score,
    )
    from datetime import date

    today = date.today()
    state = load_state()
    access_counts = state.get("access_counts", {})
    inbound_map = build_inbound_link_map()

    issues = []
    for article in list_wiki_articles():
        content = article.read_text(encoding="utf-8")
        rel = article.relative_to(KNOWLEDGE_DIR)
        slug = str(rel).replace(".md", "").replace("\\", "/")

        fm = parse_frontmatter(content)
        if fm.get("pinned"):
            continue  # pinned articles are always included

        updated = fm.get("updated") or fm.get("created")
        acc_count = access_counts.get(slug, 0)

        # Check age
        days_old = 0
        if updated:
            try:
                days_old = (today - date.fromisoformat(updated)).days
            except ValueError:
                pass

        if days_old > 90 and acc_count == 0:
            inbound = inbound_map.get(slug, 0)
            score = compute_score(
                score_recency(updated, today),
                score_linkedness(inbound),
                score_access(acc_count),
            )
            issues.append({
                "severity": "suggestion",
                "check": "low_priority_article",
                "file": str(rel),
                "detail": (
                    f"Low priority: {days_old} days old, never accessed, "
                    f"{inbound} inbound links, score={score:.3f} — "
                    f"candidate for consolidation or archiving"
                ),
            })

    return issues


def check_orphan_source_files() -> list[dict]:
    """Check for source files declared in sources.yaml that haven't been ingested."""
    from utils import load_sources_config, resolve_source_files

    state = load_state()
    ingested_sources = state.get("ingested_sources", {})
    issues = []

    for group in load_sources_config():
        for fpath in resolve_source_files(group):
            key = f"{group.id}/{fpath.name}"
            if key not in ingested_sources:
                issues.append({
                    "severity": "warning",
                    "check": "orphan_source_file",
                    "file": f"sources/{key}",
                    "detail": f"Uningested source: {fpath.name} (group: {group.id})",
                })

    return issues


async def check_contradictions() -> list[dict]:
    """Use LLM to detect contradictions across articles."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    wiki_content = read_all_wiki_content()

    prompt = f"""Review this knowledge base for contradictions, inconsistencies, or
conflicting claims across articles.

## Knowledge Base

{wiki_content}

## Instructions

Look for:
- Direct contradictions (article A says X, article B says not-X)
- Inconsistent recommendations (different articles recommend conflicting approaches)
- Outdated information that conflicts with newer entries

For each issue found, output EXACTLY one line in this format, using the
article's full path including folder prefix (concepts/, connections/, or qa/),
without the .md extension. Examples of valid slugs: [concepts/auth-middleware],
[qa/pricing-faq], [connections/auth-and-webhooks].

CONTRADICTION: [slug1] vs [slug2] - description of the conflict
INCONSISTENCY: [slug] - description of the inconsistency

If no issues found, output exactly: NO_ISSUES

Do NOT output anything else - no preamble, no explanation, just the formatted lines.
Always include the folder prefix in slugs. Bare filenames without a prefix will be ignored."""

    response = ""
    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=str(ROOT_DIR),
                allowed_tools=[],
                max_turns=2,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response += block.text
    except Exception as e:
        return [{"severity": "error", "check": "contradiction", "file": "(system)", "detail": f"LLM check failed: {e}"}]

    issues = []
    flagged_slugs: set[str] = set()
    if "NO_ISSUES" not in response:
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("CONTRADICTION:") or line.startswith("INCONSISTENCY:"):
                issues.append({
                    "severity": "warning",
                    "check": "contradiction",
                    "file": "(cross-article)",
                    "detail": line,
                })
                # Extract [file1] and [file2] slugs from the line so we can quarantine them
                bracket_matches = re.findall(r"\[([^\]]+)\]", line)
                for match in bracket_matches:
                    slug = match.replace(".md", "").lstrip("/")
                    if any(slug.startswith(p) for p in VALID_SLUG_PREFIXES):
                        flagged_slugs.add(slug)

    # Replace (not union) the quarantine with this run's findings. The LLM
    # sees the whole KB every run, so its latest snapshot is authoritative —
    # flags that disappear because the contradiction was fixed should
    # auto-clear, not require operator nuke-and-pave via --resolve.
    save_contradictions(flagged_slugs)

    return issues


def generate_report(all_issues: list[dict]) -> str:
    """Generate a markdown lint report."""
    errors = [i for i in all_issues if i["severity"] == "error"]
    warnings = [i for i in all_issues if i["severity"] == "warning"]
    suggestions = [i for i in all_issues if i["severity"] == "suggestion"]

    lines = [
        f"# Lint Report - {today_iso()}",
        "",
        f"**Total issues:** {len(all_issues)}",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        f"- Suggestions: {len(suggestions)}",
        "",
    ]

    for severity, issues, marker in [
        ("Errors", errors, "x"),
        ("Warnings", warnings, "!"),
        ("Suggestions", suggestions, "?"),
    ]:
        if issues:
            lines.append(f"## {severity}")
            lines.append("")
            for issue in issues:
                fixable = " (auto-fixable)" if issue.get("auto_fixable") else ""
                lines.append(f"- **[{marker}]** `{issue['file']}` - {issue['detail']}{fixable}")
            lines.append("")

    if not all_issues:
        lines.append("All checks passed. Knowledge base is healthy.")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Lint the knowledge base")
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="Skip LLM-based checks (contradictions) - faster and free",
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Clear the contradiction quarantine (after human review)",
    )
    args = parser.parse_args()

    if args.resolve:
        save_contradictions(set())
        print("Contradiction quarantine cleared.")
        return 0

    print("Running knowledge base lint checks...")
    all_issues: list[dict] = []

    # Structural checks (free, instant)
    checks = [
        ("Broken links", check_broken_links),
        ("Orphan pages", check_orphan_pages),
        ("Orphan sources (daily)", check_orphan_sources),
        ("Orphan sources (files)", check_orphan_source_files),
        ("Stale articles", check_stale_articles),
        ("Missing backlinks", check_missing_backlinks),
        ("Sparse articles", check_sparse_articles),
        ("Low priority articles", check_low_priority_articles),
        ("Source anchors", check_source_anchors),
        ("Memory types", check_memory_types),
    ]

    for name, check_fn in checks:
        print(f"  Checking: {name}...")
        issues = check_fn()
        all_issues.extend(issues)
        print(f"    Found {len(issues)} issue(s)")

    # LLM check (costs money)
    if not args.structural_only:
        print("  Checking: Contradictions (LLM)...")
        issues = asyncio.run(check_contradictions())
        all_issues.extend(issues)
        print(f"    Found {len(issues)} issue(s)")
    else:
        print("  Skipping: Contradictions (--structural-only)")

    # Generate and save report
    report = generate_report(all_issues)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"lint-{today_iso()}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    # Update state
    state = load_state()
    state["last_lint"] = now_iso()
    save_state(state)

    # Summary
    errors = sum(1 for i in all_issues if i["severity"] == "error")
    warnings = sum(1 for i in all_issues if i["severity"] == "warning")
    suggestions = sum(1 for i in all_issues if i["severity"] == "suggestion")
    print(f"\nResults: {errors} errors, {warnings} warnings, {suggestions} suggestions")

    if errors > 0:
        print("\nErrors found - knowledge base needs attention!")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
