"""
Run canary questions against the knowledge base to detect drift.

Loads a YAML file of (question, must_contain[]) pairs, runs each via the
query pipeline using Haiku (cheap), and checks whether each expected
substring appears in the answer. Writes a report and exits non-zero
if any canary fails.

Usage:
    uv run python scripts/canary.py                      # run all canaries
    uv run python scripts/canary.py --id compile-model   # run a specific canary
    uv run python scripts/canary.py --dry-run            # list canaries, don't run
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from config import KNOWLEDGE_DIR, MODEL_CANARY, REPORTS_DIR, now_iso, today_iso

CANARIES_FILE = KNOWLEDGE_DIR / "canaries.yaml"


@dataclass
class Canary:
    id: str
    question: str
    must_contain: list[str]
    must_not_contain: list[str] = field(default_factory=list)


@dataclass
class CanaryResult:
    canary_id: str
    passed: bool
    missing: list[str]
    answer: str
    forbidden_found: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    status: str = "passed"  # "passed" | "failed" | "errored"
    error: str | None = None


def load_canaries(path: Path | None = None) -> list[Canary]:
    """Load canaries from YAML. Returns empty list if file is missing or empty.

    Schema:
        version: 1
        canaries:
          - id: str
            question: str
            must_contain: [str, ...]
    """
    target = path or CANARIES_FILE
    if not target.exists():
        return []

    import yaml

    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not raw or not isinstance(raw, dict):
        return []

    canaries = []
    for i, entry in enumerate(raw.get("canaries", [])):
        canary_id = entry.get("id", f"<entry {i}>")
        if not entry.get("id"):
            raise ValueError(f"Canary {canary_id} missing required field: id")
        if not entry.get("question"):
            raise ValueError(f"Canary '{canary_id}' missing required field: question")
        must_contain = entry.get("must_contain")
        if not must_contain or not isinstance(must_contain, list):
            raise ValueError(
                f"Canary '{canary_id}' missing or empty must_contain "
                f"(silent-drop would hide this canary from runs)"
            )
        canaries.append(Canary(
            id=entry["id"],
            question=entry["question"],
            must_contain=list(must_contain),
            must_not_contain=list(entry.get("must_not_contain", [])),
        ))
    return canaries


def check_answer(
    answer: str,
    must_contain: list[str],
    must_not_contain: list[str] | None = None,
) -> CanaryResult:
    """Case-insensitive substring check with negation guard.

    The canary passes only if ALL must_contain substrings appear AND NONE of
    the must_not_contain substrings appear. The must_not_contain list exists
    to defeat the classic false-positive where a wrong answer happens to
    contain the trigger word (e.g., "does NOT use sonnet" contains "sonnet").
    """
    must_not_contain = must_not_contain or []
    lowered = answer.lower()
    missing = [s for s in must_contain if s.lower() not in lowered]
    forbidden_found = [s for s in must_not_contain if s.lower() in lowered]
    passed = (len(missing) == 0 and len(forbidden_found) == 0)
    return CanaryResult(
        canary_id="",
        passed=passed,
        missing=missing,
        answer=answer,
        forbidden_found=forbidden_found,
        status="passed" if passed else "failed",
    )


async def run_canary(canary: Canary) -> CanaryResult:
    """Run one canary question via the knowledge-base query pipeline with Haiku."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )
    from utils import read_all_wiki_content

    wiki_content = read_all_wiki_content()

    prompt = f"""You are a knowledge base query engine. Answer the question below using ONLY the knowledge base provided. Be concise (2-4 sentences). Cite articles with [[wikilinks]].

## Knowledge Base

{wiki_content}

## Question

{canary.question}
"""

    answer = ""
    cost = 0.0

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=str(Path(__file__).resolve().parent.parent),
                model=MODEL_CANARY,
                system_prompt={"type": "preset", "preset": "claude_code"},
                allowed_tools=[],
                max_turns=2,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        answer += block.text
            elif isinstance(message, ResultMessage):
                cost = message.total_cost_usd or 0.0
    except Exception as e:
        return CanaryResult(
            canary_id=canary.id,
            passed=False,
            missing=[],
            answer=f"ERROR: {e}",
            status="errored",
            error=str(e),
            cost_usd=cost,
        )

    result = check_answer(answer, canary.must_contain, canary.must_not_contain)
    result.canary_id = canary.id
    result.cost_usd = cost
    return result


def generate_report(results: list[CanaryResult]) -> str:
    passed = [r for r in results if r.status == "passed"]
    failed = [r for r in results if r.status == "failed"]
    errored = [r for r in results if r.status == "errored"]
    total_cost = sum(r.cost_usd for r in results)

    lines = [
        f"# Canary Report - {today_iso()}",
        "",
        f"**Total:** {len(results)} canaries | Passed: {len(passed)} | "
        f"Failed: {len(failed)} | Errored: {len(errored)}",
        f"**Cost:** ${total_cost:.4f}",
        "",
    ]

    if errored:
        lines.append("## Errored Canaries (infrastructure failures)")
        lines.append("")
        for r in errored:
            lines.append(f"- **{r.canary_id}**: {r.error}")
        lines.append("")

    if failed:
        lines.append("## Failed Canaries")
        lines.append("")
        for r in failed:
            lines.append(f"### {r.canary_id}")
            lines.append("")
            if r.missing:
                lines.append(f"**Missing substrings:** {', '.join(r.missing)}")
            if r.forbidden_found:
                lines.append(f"**Forbidden substrings matched:** {', '.join(r.forbidden_found)}")
            lines.append("")
            lines.append("**Answer received:**")
            lines.append("")
            lines.append("> " + r.answer.replace("\n", "\n> "))
            lines.append("")

    if passed:
        lines.append("## Passed Canaries")
        lines.append("")
        for r in passed:
            lines.append(f"- {r.canary_id}")
        lines.append("")

    return "\n".join(lines)


def run_whisper_canary() -> dict:
    """Drift canary for the grounded rewrite pipeline.

    Uses a pre-transcribed voice-style phrase ('how do I run the tailwind
    rebuild command?') so the canary is deterministic and doesn't require
    Whisper to be loaded. Asserts the rewrite cites the rebuild command
    string from the feedback memory.

    Returns a dict compatible with the other canary entries:
        {"name": str, "passed": bool, "detail": str}
    """
    from whisper.orchestrator import enhance_from_transcript

    transcript = "how do I run the tailwind rebuild command"
    expected_substrings = ["@tailwindcss/cli", "var/tailwind/app.built.css"]

    try:
        result = enhance_from_transcript(transcript=transcript, mode="rewrite")
    except Exception as exc:  # noqa: BLE001
        return {"name": "whisper:tailwind-rebuild", "passed": False, "detail": f"exception: {exc}"}

    missing = [s for s in expected_substrings if s not in result.enhanced_prompt]
    if missing:
        return {
            "name": "whisper:tailwind-rebuild",
            "passed": False,
            "detail": f"prompt missing expected substrings: {missing}",
        }
    return {"name": "whisper:tailwind-rebuild", "passed": True, "detail": "ok"}


def main():
    parser = argparse.ArgumentParser(description="Run canary questions against the knowledge base")
    parser.add_argument("--id", type=str, help="Run a specific canary by id")
    parser.add_argument("--dry-run", action="store_true", help="List canaries without running")
    args = parser.parse_args()

    canaries = load_canaries()
    if not canaries:
        print(f"No canaries found at {CANARIES_FILE}. Create the file to get started.")
        return 0

    if args.id:
        canaries = [c for c in canaries if c.id == args.id]
        if not canaries:
            print(f"No canary with id '{args.id}' found.")
            return 1

    if args.dry_run:
        print(f"Canaries ({len(canaries)}):")
        for c in canaries:
            print(f"  - {c.id}: {c.question}")
        return 0

    print(f"Running {len(canaries)} canaries with {MODEL_CANARY}...")
    results: list[CanaryResult] = []
    for i, canary in enumerate(canaries, 1):
        print(f"\n[{i}/{len(canaries)}] {canary.id}: {canary.question}")
        result = asyncio.run(run_canary(canary))
        results.append(result)
        status_label = {"passed": "PASS", "failed": "FAIL", "errored": "ERROR"}.get(result.status, "FAIL")
        print(f"  {status_label} (cost: ${result.cost_usd:.4f})")
        if result.status == "failed":
            if result.missing:
                print(f"  Missing: {', '.join(result.missing)}")
            if result.forbidden_found:
                print(f"  Forbidden matched: {', '.join(result.forbidden_found)}")
        elif result.status == "errored":
            print(f"  Error: {result.error}")

    report = generate_report(results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"canary-{today_iso()}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    # Run whisper drift canary (only when not filtering by --id)
    whisper_failed = False
    whisper_errored = False
    if not args.id:
        print("\n[whisper] whisper:tailwind-rebuild")
        wresult = run_whisper_canary()
        label = "PASS" if wresult["passed"] else "FAIL"
        print(f"  {label} — {wresult['detail']}")
        if not wresult["passed"]:
            if wresult["detail"].startswith("exception:"):
                whisper_errored = True
            else:
                whisper_failed = True

    passed_count = sum(1 for r in results if r.status == "passed")
    failed_count = sum(1 for r in results if r.status == "failed")
    errored_count = sum(1 for r in results if r.status == "errored")

    if errored_count > 0 or whisper_errored:
        total_errored = errored_count + int(whisper_errored)
        print(f"\n{total_errored} canary/canaries errored (infrastructure failure - retry may help)")
        return 2  # different exit code from drift failures
    if failed_count > 0 or whisper_failed:
        print(f"\n{failed_count + int(whisper_failed)} canary/canaries failed - knowledge base may be drifting!")
        return 1
    print(f"\nAll {passed_count} canaries passed.")
    return 0


if __name__ == "__main__":
    exit(main())
