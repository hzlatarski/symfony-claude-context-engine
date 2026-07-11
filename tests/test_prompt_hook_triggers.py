"""Tests for the UserPromptSubmit hook's conceptual-question trigger.

The hook file has a hyphenated name (not importable as a normal module), so
we load it by path. Importing is side-effect-free: main() only runs under
``__main__`` and the disabled-hooks check is a no-op when the env var is unset.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HOOK_PATH = Path(__file__).parent.parent / "hooks" / "user-prompt-submit.py"


@pytest.fixture(scope="module")
def hook():
    spec = importlib.util.spec_from_file_location("user_prompt_submit_hook", HOOK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("prompt", [
    "Why do we skip grading when the transcript is empty?",
    "What was the decision behind per-org credit pools?",
    "How does the arena resolve the voice gender?",
    "What's the rationale for the phase-aware grading axes?",
    "Explain our pricing strategy for corporate tiers",
    # Derivatives must fire (the old trailing-\b killed these):
    "Explain the architecture of the grading pipeline",
    "What were the reasons for choosing Mercure?",
    "Summarize past decisions about billing",
    "user preferences for copy tone",
])
def test_conceptual_prompts_trigger_kb(hook, prompt):
    assert hook._looks_conceptual(prompt)


@pytest.mark.parametrize("prompt", [
    "Fix the typo in the header",
    "Run the tests",
    "Rename this variable to totalCount",
    "Add a null check on line 40",
    # Domain nouns alone must NOT trigger the ~1.5s KB load:
    "improve the grading UI spacing",
    "bump the persona image size to 512px",
    "adjust the scenario card padding",
])
def test_mechanical_prompts_do_not_trigger_kb(hook, prompt):
    assert not hook._looks_conceptual(prompt)
