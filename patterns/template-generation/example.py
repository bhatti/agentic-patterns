#!/usr/bin/env python3
"""
Template generation — LLM drafts a reviewable template; slots filled from trusted data

Problem: Generating final thank-you / transactional copy on every run is non-deterministic
and mixes creative text with facts (order ids, names), which is hard to audit.

Solution: Generate a *template* with explicit placeholders; humans approve the template;
runtime code fills slots from CRM/OMS. Few-shot examples steer structure and slot naming.

This file uses a *mock* template generator (no API keys). Replace ``mock_generate_template``
with pydantic-ai / Ollama / Gemini as in the book notebook.

Usage:
    python example.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Few-shot "gold" fragments: show the model the expected slot style (book: implicit via prompt).
FEW_SHOT_TEMPLATE_EXAMPLES = (
    "Example A (English, SaaS onboarding):\n"
    "Dear [CUSTOMER_NAME], welcome to [PRODUCT_NAME]. "
    "Your workspace id is [WORKSPACE_ID]. — [CSM_NAME]\n\n"
    "Example B (support closure):\n"
    "Ticket [TICKET_ID] is resolved. If you need more help, ask for [SUPPORT_AGENT].\n"
)


def mock_generate_template(
    *,
    scenario: str,
    locale: str,
    tone: str,
) -> str:
    """
    Simulate an LLM that returns a reusable email skeleton with bracketed slots.

    Production: call your model with low temperature, few-shot block, and strict instructions
    to output ONLY the template (no preamble).

    Args:
        scenario: Business scenario (e.g. post-purchase thank-you).
        locale: Language/locale label.
        tone: Voice constraint (e.g. professional, warm).

    Returns:
        Template string containing placeholders.
    """
    _ = (scenario, locale, tone)
    return (
        "Subject: Thank you for choosing ExampleCloud\n\n"
        "Dear [CUSTOMER_NAME],\n\n"
        "Thank you for order [ORDER_ID] placed on [ORDER_DATE]. "
        "Your subscription tier is [PLAN_NAME].\n\n"
        "If setup questions come up, your success contact is [SUPPORT_AGENT] "
        "(reference this id: [CASE_REF]).\n\n"
        "— ExampleCloud Customer Success\n"
    )


def extract_slot_names(template: str) -> list[str]:
    """
    Return unique placeholder names in order of first appearance.

    Detects tokens like ``[CUSTOMER_NAME]``.

    Args:
        template: Text with bracketed slots.

    Returns:
        List of slot names without brackets.
    """
    found = re.findall(r"\[([A-Z][A-Z0-9_]*)\]", template)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in found:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def fill_template(template: str, slots: dict[str, str]) -> str:
    """
    Replace ``[KEY]`` placeholders with values from ``slots`` (keys uppercased).

    Args:
        template: Approved template text.
        slots: Map from slot name (e.g. ``CUSTOMER_NAME``) to value.

    Returns:
        Filled message for sending.
    """
    out = template
    for key, value in slots.items():
        token = f"[{key.upper()}]"
        out = out.replace(token, value)
    return out


def validate_required_slots(template: str, required: Iterable[str]) -> tuple[bool, list[str]]:
    """
    Check that every required slot appears in the template.

    Args:
        template: Candidate template.
        required: Slot names that must be present.

    Returns:
        Tuple of (all present, missing names).
    """
    missing = [name for name in required if f"[{name.upper()}]" not in template]
    return len(missing) == 0, missing


def main() -> None:
    """Build a template with few-shot context (printed), validate, fill, show result."""
    print("Pattern 29: Template generation (safeguarded comms)\n")
    print("--- Few-shot style reference (passed to model in production) ---")
    print(FEW_SHOT_TEMPLATE_EXAMPLES)

    template = mock_generate_template(
        scenario="post_purchase_thank_you",
        locale="en-US",
        tone="warm_professional",
    )
    print("--- Generated template (mock LLM) ---")
    print(template)

    required = (
        "CUSTOMER_NAME",
        "ORDER_ID",
        "ORDER_DATE",
        "PLAN_NAME",
        "SUPPORT_AGENT",
        "CASE_REF",
    )
    ok, missing = validate_required_slots(template, required)
    print("--- Validation ---")
    print("  slots found:", extract_slot_names(template))
    print("  required_ok:", ok, ("missing: " + str(missing) if not ok else ""))

    filled = fill_template(
        template,
        {
            "CUSTOMER_NAME": "Jordan Lee",
            "ORDER_ID": "ORD-77821",
            "ORDER_DATE": "2026-01-15",
            "PLAN_NAME": "Business Plus",
            "SUPPORT_AGENT": "Sam Rivera",
            "CASE_REF": "CS-44102",
        },
    )
    print("--- Filled from OMS/CRM (no LLM) ---")
    print(filled)
    print("\nReview: approve the *template* once per segment/locale; avoid regenerating full "
          "copy per user with an unconstrained LLM.")


if __name__ == "__main__":
    main()
