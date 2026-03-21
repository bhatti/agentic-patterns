#!/usr/bin/env python3
"""
Assembled reformat — verified facts assembled first; LLM only for low-risk narrative

Problem: A single LLM call for a full product page can hallucinate high-risk facts (e.g. wrong
battery chemistry → incorrect checked-baggage guidance for air travel).

Solution: Identify **risk-bearing** attributes; load them from trusted records; **assemble**
compliance/shipping/spec blocks deterministically; optionally **reformat** marketing copy that
**only** references verified fields; **validate** output (Self-check in production).

This example uses a **camera** SKU with **NiMH** power: the assembled page must never claim
**lithium** unless the PIM record says so. ``mock_llm_marketing`` simulates a bad model that
invents "lithium" — ``validate_high_risk_copy`` flags it.

Usage:
    python example.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class BatteryChemistry(str, Enum):
    """Manufacturer-verified battery chemistry from PIM (ground truth)."""

    NONE = "none"
    NIMH = "nimh"
    LI_ION = "li_ion"
    LIPO = "lipo"


@dataclass(frozen=True)
class VerifiedProductRecord:
    """
    Trusted product facts — never invented by an LLM in this pattern.

    Attributes:
        sku: Stock keeping unit.
        name: Display name.
        battery: Verified chemistry category.
        battery_wh: Watt-hour rating if applicable; None if unknown or N/A.
    """

    sku: str
    name: str
    battery: BatteryChemistry
    battery_wh: float | None


def render_compliance_and_shipping_block(facts: VerifiedProductRecord) -> str:
    """
    Deterministic regulatory copy derived only from ``facts``.

    Args:
        facts: Verified product row.

    Returns:
        Markdown section for battery / travel.
    """
    lines = [
        "## Power & travel (verified data)",
        f"- **SKU:** {facts.sku}",
        f"- **Battery chemistry (PIM):** {facts.battery.value.upper()}",
    ]
    if facts.battery_wh is not None:
        lines.append(f"- **Rated energy:** {facts.battery_wh} Wh (where applicable)")
    if facts.battery == BatteryChemistry.NONE:
        lines.append("- **Travel note:** No user-replaceable battery listed in PIM; confirm with manual.")
    elif facts.battery == BatteryChemistry.NIMH:
        lines.append(
            "- **Travel note:** NiMH packs are generally not classified like large Li-ion; "
            "always follow your airline and IATA rules for your specific pack."
        )
    elif facts.battery in (BatteryChemistry.LI_ION, BatteryChemistry.LIPO):
        lines.append(
            "- **Travel note:** Lithium-class batteries may be subject to **carry-on** limits and "
            "per-cell Wh caps; do **not** rely on marketing copy—use the Wh on the battery label."
        )
    return "\n".join(lines) + "\n"


def render_assembled_stub_page(facts: VerifiedProductRecord) -> str:
    """
    Build a minimal PDP from verified fields only (assembly step).

    Args:
        facts: Verified record.

    Returns:
        Markdown document shell with deterministic sections.
    """
    header = f"# {facts.name}\n\n"
    body = render_compliance_and_shipping_block(facts)
    return header + body


def mock_llm_marketing_go_bad(facts: VerifiedProductRecord) -> str:
    """
    Simulate an unsafe LLM that invents lithium wording despite NiMH in PIM.

    In production, replace with a constrained prompt + low temperature; still run validation.

    Args:
        facts: Verified record (ignored by this *bad* mock to show failure mode).

    Returns:
        Risky marketing paragraph.
    """
    _ = facts
    return (
        "Our lightweight travel camera includes a **lithium** battery for long shoots—"
        "toss it in checked baggage without worry."
    )


def mock_llm_marketing_safe(facts: VerifiedProductRecord) -> str:
    """
    Simulate marketing copy that only reflects verified chemistry.

    Args:
        facts: Verified record.

    Returns:
        Safer marketing snippet.
    """
    chem = facts.battery.value.replace("_", "-").upper()
    return (
        f"Designed for travelers: power comes from a **{chem}** battery per manufacturer specs—"
        "check airline rules for your exact pack."
    )


def validate_high_risk_copy(page_text: str, facts: VerifiedProductRecord) -> tuple[bool, str]:
    """
    Reject copy that mentions lithium chemistry when verified data says otherwise.

    Extend with allergen rules, medical claims, etc.

    Args:
        page_text: Full or partial generated page.
        facts: Ground truth.

    Returns:
        Tuple of (ok, reason).
    """
    low = page_text.lower()
    lithium_markers = ("lithium", "li-ion", "li ion", "lipo", "li-po")
    if facts.battery in (BatteryChemistry.NIMH, BatteryChemistry.NONE):
        if any(m in low for m in lithium_markers):
            return False, "Lithium-class wording present but PIM chemistry is not Li-ion/LiPo."
    return True, ""


def strip_unsafe_marketing(marketing: str, facts: VerifiedProductRecord) -> str:
    """
    Last-resort scrub: remove sentences that trip validation (demo only).

    Production should regenerate with stricter prompts or block publish.

    Args:
        marketing: Candidate marketing text.
        facts: Verified record.

    Returns:
        Possibly shortened text.
    """
    ok, _ = validate_high_risk_copy(marketing, facts)
    if ok:
        return marketing
    return "[Marketing blocked: failed high-risk validation — regenerate with grounded prompt.]"


def main() -> None:
    """Show assembled page, bad vs safe marketing, and validation."""
    facts = VerifiedProductRecord(
        sku="CAM-NIMH-100",
        name="ExampleTravel Cam 100",
        battery=BatteryChemistry.NIMH,
        battery_wh=12.0,
    )
    print("Pattern 30: Assembled reformat\n")
    print("--- Assembled (deterministic) ---")
    print(render_assembled_stub_page(facts))

    bad = mock_llm_marketing_go_bad(facts)
    print("--- Mock LLM marketing (intentionally unsafe) ---")
    print(bad)
    ok_bad, reason = validate_high_risk_copy(bad, facts)
    print("Validation:", ok_bad, reason)

    safe = mock_llm_marketing_safe(facts)
    print("\n--- Mock LLM marketing (grounded) ---")
    print(safe)
    print("Validation:", validate_high_risk_copy(safe, facts)[0])

    print("\n--- Full page + scrub ---")
    page = render_assembled_stub_page(facts) + "\n## Marketing\n" + strip_unsafe_marketing(bad, facts)
    print(page[:800], "..." if len(page) > 800 else "")


if __name__ == "__main__":
    main()
