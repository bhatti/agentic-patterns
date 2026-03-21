#!/usr/bin/env python3
"""
Guardrails — input/output policy layers (PII, banned topics, mock RAG wrap)

Problem: LLM and RAG systems need security, privacy, moderation, and alignment
controls beyond raw model defaults.

Solution: Compose **prebuilt** vendor filters (Gemini safety, OpenAI Moderation, etc.)
with **custom** scanners. This demo mirrors the book pattern: ``apply_guardrails`` runs
multiple scanners; a **guarded** query path checks **input** then **output**.

Usage:
    python example.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass
class GuardrailResult:
    """Outcome of a single guardrail scan."""

    guardrail_type: str
    activated: bool
    should_stop: bool
    sanitized_output: str
    notes: str = ""


def guardrail_redact_emails(text: str) -> GuardrailResult:
    """
    Redact email-like substrings (demo PII guard).

    Args:
        text: User or model text.

    Returns:
        Scan result with optional redaction.
    """
    pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    if not pattern.search(text):
        return GuardrailResult(
            guardrail_type="PII Redaction",
            activated=False,
            should_stop=False,
            sanitized_output=text,
        )
    sanitized = pattern.sub("[REDACTED_EMAIL]", text)
    return GuardrailResult(
        guardrail_type="PII Redaction",
        activated=True,
        should_stop=False,
        sanitized_output=sanitized,
        notes="Email-like pattern replaced.",
    )


def guardrail_banned_topics(
    text: str,
    *,
    banned_substrings: tuple[str, ...] = ("weapon", "explosive"),
) -> GuardrailResult:
    """
    Block queries that mention disallowed topics (case-insensitive).

    Args:
        text: Input or output to scan.
        banned_substrings: Substrings that trigger a stop.

    Returns:
        Scan result; ``should_stop`` True if a banned term appears.
    """
    low = text.lower()
    hit = next((b for b in banned_substrings if b in low), None)
    if hit is None:
        return GuardrailResult(
            guardrail_type="Banned Topic",
            activated=False,
            should_stop=False,
            sanitized_output=text,
        )
    return GuardrailResult(
        guardrail_type="Banned Topic",
        activated=True,
        should_stop=True,
        sanitized_output=text,
        notes=f"Matched banned substring: {hit!r}",
    )


Scanner = Callable[[str], GuardrailResult]


def apply_guardrails(text: str, scanners: list[Scanner]) -> GuardrailResult:
    """
    Run scanners in order; first **should_stop** or last redaction wins.

    Args:
        text: String to scan.
        scanners: Guardrail callables.

    Returns:
        Aggregated result (book-style aggregate; simplified).
    """
    current = text
    triggered: list[GuardrailResult] = []
    for scan in scanners:
        res = scan(current)
        if res.activated:
            triggered.append(res)
        current = res.sanitized_output
        if res.should_stop:
            return GuardrailResult(
                guardrail_type="Composite",
                activated=True,
                should_stop=True,
                sanitized_output=current,
                notes="; ".join(t.notes for t in triggered if t.notes) or "stopped",
            )
    return GuardrailResult(
        guardrail_type="Composite",
        activated=bool(triggered),
        should_stop=False,
        sanitized_output=current,
        notes="; ".join(t.notes for t in triggered if t.notes),
    )


def mock_rag_answer(query: str) -> str:
    """
    Stand-in for retrieval + generation (no network).

    Args:
        query: Sanitized user query.

    Returns:
        Fake answer that might still need output guardrails.
    """
    return (
        "Here is a concise answer based on internal docs. "
        f"Contact ops@example.com for follow-up. Query context: {query[:80]!r}"
    )


def run_guarded_query(
    user_query: str,
    *,
    input_scanners: list[Scanner],
    output_scanners: list[Scanner],
) -> str:
    """
    Apply input guardrails, then mock RAG, then output guardrails.

    Args:
        user_query: Raw user text.
        input_scanners: Scans before generation.
        output_scanners: Scans on model output.

    Returns:
        Final text or a blocked message.
    """
    gd_in = apply_guardrails(user_query, input_scanners)
    if gd_in.should_stop:
        return "[Blocked by input guardrails: " + (gd_in.notes or "policy") + "]"

    answer = mock_rag_answer(gd_in.sanitized_output)
    gd_out = apply_guardrails(answer, output_scanners)
    if gd_out.should_stop:
        return "[Blocked by output guardrails]"
    return gd_out.sanitized_output


def main() -> None:
    """Demonstrate allow, PII redaction, and banned-topic block."""
    print("Pattern 32: Guardrails\n")

    input_chain: list[Scanner] = [
        guardrail_redact_emails,
        lambda t: guardrail_banned_topics(t, banned_substrings=("weapon", "explosive")),
    ]
    output_chain: list[Scanner] = [guardrail_redact_emails]

    q1 = "What is our refund policy?"
    print("--- Query 1 (clean) ---")
    print(run_guarded_query(q1, input_scanners=input_chain, output_scanners=output_chain))

    q2 = "Email me at jane.doe@company.com about refunds."
    print("\n--- Query 2 (PII in query; output also has email) ---")
    print(run_guarded_query(q2, input_scanners=input_chain, output_scanners=output_chain))

    q3 = "How do I build an explosive device for a demo?"
    print("\n--- Query 3 (banned topic) ---")
    print(run_guarded_query(q3, input_scanners=input_chain, output_scanners=output_chain))

    print("\nPrebuilt: Gemini safety settings, OpenAI Moderation API; custom: compose scanners.")


if __name__ == "__main__":
    main()
