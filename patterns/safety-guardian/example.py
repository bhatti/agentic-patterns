#!/usr/bin/env python3
"""Safety layers — tiered allow/block (stdlib). Usage: python example.py"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SafetyAction(str, Enum):
    """Response when a policy layer triggers."""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    SHUTDOWN = "shutdown"


@dataclass
class SafetyContext:
    """Context passed through safety layers."""

    user_tier: str
    content_class: str
    risk_score: float


def evaluate_layers(ctx: SafetyContext) -> SafetyAction:
    """
    Stub multi-layer policy: PII -> risk score -> tier.

    Args:
        ctx: Request context.

    Returns:
        Coarse action for the orchestrator.
    """
    if ctx.content_class == "pii_heavy":
        return SafetyAction.BLOCK
    if ctx.risk_score > 0.85:
        return SafetyAction.SHUTDOWN
    if ctx.risk_score > 0.5:
        return SafetyAction.WARN
    return SafetyAction.ALLOW


def main() -> None:
    print(evaluate_layers(SafetyContext("enterprise", "general", 0.2)))
    print(evaluate_layers(SafetyContext("free", "general", 0.9)))


if __name__ == "__main__":
    main()
