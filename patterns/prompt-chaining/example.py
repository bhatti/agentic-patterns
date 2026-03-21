#!/usr/bin/env python3
"""
Prompt chaining — sequential steps with structured handoffs (Agentic Design Patterns)

Each step has a single job; output is structured for the next. This demo uses mocks instead
of live LLM calls. Swap ``mock_llm`` bodies for Ollama / APIs.

Reference: Antonio Gulli, Agentic Design Patterns — prompt-chainer agent spec.

Usage:
    python example.py
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass
class ChainState:
    """Mutable pipeline state passed through sequential steps."""

    user_query: str
    intent: str = ""
    sub_questions: list[str] = field(default_factory=list)
    bullet_answer: str = ""
    final_markdown: str = ""


def step_classify_intent(state: ChainState) -> ChainState:
    """
    Step 1: classify user intent (mock rule-based stand-in for an LLM).

    Args:
        state: Current chain state.

    Returns:
        Updated state with ``intent`` set.
    """
    q = state.user_query.lower()
    if "compare" in q or " vs " in q or "versus" in q:
        intent = "compare"
    elif "how" in q or "steps" in q:
        intent = "how_to"
    else:
        intent = "explain"
    state.intent = intent
    return state


def step_decompose(state: ChainState) -> ChainState:
    """
    Step 2: produce sub-questions as structured lines (mock).

    Args:
        state: State after intent classification.

    Returns:
        State with ``sub_questions`` populated.
    """
    if state.intent == "compare":
        state.sub_questions = [
            "What are the defining traits of option A?",
            "What are the defining traits of option B?",
            "What tradeoffs matter for the user?",
        ]
    elif state.intent == "how_to":
        state.sub_questions = [
            "What prerequisites are required?",
            "What is the ordered procedure?",
            "What validation confirms success?",
        ]
    else:
        state.sub_questions = [
            "What is the core definition?",
            "What is one concrete example?",
            "What caveat should the user know?",
        ]
    return state


def step_answer_bullets(state: ChainState) -> ChainState:
    """
    Step 3: answer each sub-question in one line (mock LLM aggregation).

    Args:
        state: State with ``sub_questions``.

    Returns:
        State with ``bullet_answer`` as newline-separated bullets.
    """
    lines = []
    for i, sq in enumerate(state.sub_questions, start=1):
        lines.append(f"{i}. ({sq[:40]}…) — [synthetic fact tied to query keywords]")
    kw = " ".join(re.findall(r"[a-zA-Z]{4,}", state.user_query))[:60]
    lines.append(f"Context anchor: {kw!r}")
    state.bullet_answer = "\n".join(lines)
    return state


def step_format_markdown(state: ChainState) -> ChainState:
    """
    Step 4: wrap bullets in Markdown for UI (deterministic reformat).

    Args:
        state: State with ``bullet_answer``.

    Returns:
        State with ``final_markdown``.
    """
    state.final_markdown = (
        f"## Answer ({state.intent})\n\n"
        f"**Original question:** {state.user_query}\n\n"
        f"### Working notes\n```\n{state.bullet_answer}\n```\n\n"
        "*Replace synthetic lines with real LLM calls per sub-question.*\n"
    )
    return state


def run_prompt_chain(user_query: str) -> ChainState:
    """
    Execute the full chain in order.

    Args:
        user_query: End-user text.

    Returns:
        Final chain state.
    """
    s = ChainState(user_query=user_query.strip())
    s = step_classify_intent(s)
    s = step_decompose(s)
    s = step_answer_bullets(s)
    s = step_format_markdown(s)
    return s


def chain_plan_as_json(state: ChainState) -> str:
    """
    Serialize intermediate structure for logging or the next service (structured output).

    Args:
        state: State after decomposition.

    Returns:
        JSON string of intent and sub_questions.
    """
    payload: dict[str, Any] = {"intent": state.intent, "sub_questions": state.sub_questions}
    return json.dumps(payload, indent=2)


def main() -> None:
    """Run a sample complex query through the chain."""
    query = "Compare REST and GraphQL for a public API; what should we pick first?"
    print("Pattern 33: Prompt chaining\n")
    mid = ChainState(user_query=query)
    mid = step_classify_intent(mid)
    mid = step_decompose(mid)
    print("--- Intermediate (structured JSON handoff) ---")
    print(chain_plan_as_json(mid))
    final = run_prompt_chain(query)
    print("\n--- Final markdown ---")
    print(final.final_markdown)


if __name__ == "__main__":
    main()
