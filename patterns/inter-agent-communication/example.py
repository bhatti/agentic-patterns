#!/usr/bin/env python3
"""Inter-agent message envelopes (stdlib). Usage: python example.py"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentMessage:
    """Minimal envelope for routed agent traffic."""

    correlation_id: str
    sender: str
    recipient: str
    intent: str
    payload: dict[str, Any]


def make_message(sender: str, recipient: str, intent: str, payload: dict[str, Any]) -> AgentMessage:
    """Build a message with a fresh correlation id."""
    return AgentMessage(
        correlation_id=str(uuid.uuid4()),
        sender=sender,
        recipient=recipient,
        intent=intent,
        payload=payload,
    )


def main() -> None:
    m = make_message("router", "billing_agent", "handoff", {"ticket_id": "T-9"})
    print(m)


if __name__ == "__main__":
    main()
