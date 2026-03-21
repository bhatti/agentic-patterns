#!/usr/bin/env python3
"""
Human-in-the-loop (HITL) — escalation policy, review queue, approval stub.

No live UI or LLM. Models production patterns: confidence + stakes → queue → decision.

Reference: Antonio Gulli, Agentic Design Patterns — human-validator agent spec.

Usage:
    python example.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
from uuid import uuid4


class StakesLevel(str, Enum):
    """Coarse risk tier for gating human review."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def should_escalate_to_human(
    *,
    model_confidence: float,
    stakes: StakesLevel,
    confidence_threshold: float = 0.85,
) -> bool:
    """
    Return whether a decision should leave full automation and enter HITL.

    Policy: high stakes always review; medium stakes if confidence is low;
    low stakes only if confidence is below threshold.

    Args:
        model_confidence: Score in ``[0, 1]`` (e.g. from calibrator or judge).
        stakes: Business risk tier for this decision class.
        confidence_threshold: Cutoff for automatic path when stakes are low/medium.

    Returns:
        True if a human should review before commit/publish.
    """
    if stakes == StakesLevel.HIGH:
        return True
    if stakes == StakesLevel.MEDIUM:
        return model_confidence < confidence_threshold
    return model_confidence < confidence_threshold


ReviewStatus = Literal["pending", "approved", "rejected", "edited"]


@dataclass
class ReviewTicket:
    """A single item waiting for or completed by human review."""

    ticket_id: str
    payload_summary: str
    stakes: StakesLevel
    model_confidence: float
    status: ReviewStatus = "pending"
    reviewer_note: str = ""


@dataclass
class HumanReviewQueue:
    """In-memory queue stub; production uses durable store + worker assignments."""

    tickets: list[ReviewTicket] = field(default_factory=list)

    def enqueue(
        self,
        payload_summary: str,
        stakes: StakesLevel,
        model_confidence: float,
    ) -> ReviewTicket:
        """
        Add a review item and return its ticket.

        Args:
            payload_summary: Short description of what is under review.
            stakes: Risk tier.
            model_confidence: Automation confidence score.

        Returns:
            The created ``ReviewTicket``.
        """
        ticket = ReviewTicket(
            ticket_id=str(uuid4()),
            payload_summary=payload_summary,
            stakes=stakes,
            model_confidence=model_confidence,
        )
        self.tickets.append(ticket)
        return ticket

    def resolve(
        self,
        ticket_id: str,
        status: ReviewStatus,
        reviewer_note: str = "",
    ) -> ReviewTicket | None:
        """
        Mark a ticket reviewed (stub for analyst UI callback).

        Args:
            ticket_id: Identifier returned from ``enqueue``.
            status: Final disposition.
            reviewer_note: Optional rationale for audit.

        Returns:
            Updated ticket, or None if id not found.
        """
        for t in self.tickets:
            if t.ticket_id == ticket_id:
                t.status = status
                t.reviewer_note = reviewer_note
                return t
        return None


def main() -> None:
    """Demonstrate escalation and queue resolution."""
    print("escalate (high stakes, conf=0.99):", should_escalate_to_human(model_confidence=0.99, stakes=StakesLevel.HIGH))
    print("escalate (medium, conf=0.5):", should_escalate_to_human(model_confidence=0.5, stakes=StakesLevel.MEDIUM))

    queue = HumanReviewQueue()
    t1 = queue.enqueue("Trade: sell 10k shares MSFT", StakesLevel.HIGH, 0.92)
    t2 = queue.enqueue("Post: community comment #8821", StakesLevel.LOW, 0.72)
    print("queued:", t1.ticket_id[:8], "...", t2.ticket_id[:8], "...")
    queue.resolve(t1.ticket_id, "approved", "within desk limits")
    queue.resolve(t2.ticket_id, "rejected", "policy: harassment edge case")
    for t in queue.tickets:
        print(" ", t.payload_summary[:40], "->", t.status, t.reviewer_note[:30])


if __name__ == "__main__":
    main()
