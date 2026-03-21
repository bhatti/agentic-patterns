#!/usr/bin/env python3
"""
Dependency Injection Pattern - Support Ticket Pipeline

This example shows how to keep a GenAI pipeline LLM-agnostic and testable by
injecting the components that call the LLM (or external tools). In production
you pass real implementations; in tests and dev you pass lightweight mocks
that return hardcoded, deterministic results.

Real-World Problem:
-------------------
A support pipeline: (1) summarize the ticket, (2) suggest an action (e.g.,
route to billing, send template). Developing and testing is hard: LLM output
is nondeterministic, models change, and you want to run tests without API keys.
Solution: inject summarize_fn and suggest_action_fn; production uses real LLM
calls, tests use mocks that return fixed strings. The pipeline code never
talks to an LLM directly — it only calls the injected functions.

Usage:
    python example.py
    # Runs pipeline with mocks (no API). In production, inject real LLM callables.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Project root for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class TicketResult:
    """Output of the support ticket pipeline."""
    summary: str
    suggested_action: str
    raw_ticket: str


# =============================================================================
# PIPELINE (LLM-agnostic: depends only on injected callables)
# =============================================================================

def run_ticket_pipeline(
    ticket_text: str,
    summarize_fn: Callable[[str], str],
    suggest_action_fn: Callable[[str, str], str],
) -> TicketResult:
    """
    Two-step pipeline: summarize ticket, then suggest action.
    Both steps are injected so they can be real LLM calls or mocks.
    """
    summary = summarize_fn(ticket_text)
    suggested_action = suggest_action_fn(ticket_text, summary)
    return TicketResult(
        summary=summary,
        suggested_action=suggested_action,
        raw_ticket=ticket_text,
    )


# =============================================================================
# REAL IMPLEMENTATIONS (would call LLM in production)
# =============================================================================

def real_summarize(ticket_text: str) -> str:
    """
    Production: call LLM to summarize the ticket.
    Here we simulate with a simple heuristic so the example runs without API.
    """
    # In production: return llm_client.generate(f"Summarize this support ticket:\n{ticket_text}")
    if "refund" in ticket_text.lower() or "charge" in ticket_text.lower():
        return "Customer is asking about a refund or charge."
    if "login" in ticket_text.lower() or "password" in ticket_text.lower():
        return "Customer has a login or password issue."
    return "Customer support request; details in ticket."


def real_suggest_action(ticket_text: str, summary: str) -> str:
    """
    Production: call LLM to suggest action given ticket and summary.
    Here we simulate so the example runs without API.
    """
    # In production: return llm_client.generate(f"Ticket: {ticket_text}\nSummary: {summary}\nSuggest one action.")
    if "refund" in summary.lower() or "charge" in summary.lower():
        return "Route to billing; consider refund template if eligible."
    if "login" in summary.lower():
        return "Send password reset link; if persists, route to technical."
    return "Reply with acknowledgment; route to general support."


# =============================================================================
# MOCK IMPLEMENTATIONS (hardcoded, deterministic — for tests and dev)
# =============================================================================

def mock_summarize(ticket_text: str) -> str:
    """Mock: always returns a fixed summary. No LLM, no network."""
    return "Customer reports an issue with their order and requests assistance."


def mock_suggest_action(ticket_text: str, summary: str) -> str:
    """Mock: always returns a fixed action. No LLM, no network."""
    return "Route to order-support queue; use template ORDER_DELAYED."


# =============================================================================
# DEMO: RUN WITH MOCKS (deterministic, no API)
# =============================================================================

def main() -> None:
    ticket = (
        "Hi, I was charged twice for my subscription last month. "
        "I only want one charge. Can you refund the extra charge? Thanks."
    )

    print("Dependency Injection — Support ticket pipeline")
    print("Running with MOCKS (no LLM calls, deterministic)")
    print()

    result = run_ticket_pipeline(
        ticket,
        summarize_fn=mock_summarize,
        suggest_action_fn=mock_suggest_action,
    )

    print("Ticket (excerpt):", ticket[:60] + "...")
    print()
    print("Summary (from mock):", result.summary)
    print("Suggested action (from mock):", result.suggested_action)
    print()
    print("To use real LLM: pass real_summarize and real_suggest_action")
    print("(or callables that invoke Ollama/API). Tests can pass mocks for fast, deterministic runs.")


# =============================================================================
# SIMPLE TEST (demonstrates deterministic assertion with mocks)
# =============================================================================

def test_pipeline_with_mocks() -> None:
    """With mocks, output is deterministic; we can assert exact values."""
    ticket = "My login does not work."
    result = run_ticket_pipeline(
        ticket,
        summarize_fn=mock_summarize,
        suggest_action_fn=mock_suggest_action,
    )
    assert result.summary == "Customer reports an issue with their order and requests assistance."
    assert "order-support" in result.suggested_action or "ORDER" in result.suggested_action
    print("test_pipeline_with_mocks passed (deterministic mock output)")


if __name__ == "__main__":
    main()
    print()
    test_pipeline_with_mocks()
