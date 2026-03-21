#!/usr/bin/env python3
"""
Prioritization — weighted scores for competing tasks (stdlib).

Reference: Antonio Gulli, Agentic Design Patterns — prioritizer agent spec.

Usage:
    python example.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(order=True)
class WorkItem:
    """
    Sortable work unit: priority uses a composite score (higher = more urgent).

    Attributes:
        sort_index: Internal tie-break for dataclass ordering (set to negative score).
        task_id: Stable identifier.
        urgency: 0–1 normalized urgency.
        importance: 0–1 business impact.
        effort: 0–1 relative cost (higher = more expensive).
        sla_slack_s: Seconds until SLA breach (lower = more critical).
        security_severity: 0 = none, 1 = critical incident.
    """

    sort_index: float
    task_id: str
    urgency: float
    importance: float
    effort: float
    sla_slack_s: float
    security_severity: float

    @staticmethod
    def build(
        task_id: str,
        urgency: float,
        importance: float,
        effort: float,
        sla_slack_s: float,
        security_severity: float,
        weights: dict[str, float] | None = None,
    ) -> "WorkItem":
        """
        Construct a ``WorkItem`` with ``sort_index`` from a weighted score.

        Args:
            task_id: Identifier.
            urgency: Urgency in ``[0, 1]``.
            importance: Importance in ``[0, 1]``.
            effort: Effort in ``[0, 1]``.
            sla_slack_s: Seconds of slack; mapped inversely into score.
            security_severity: ``[0, 1]`` severity bump.
            weights: Optional keys ``urgency``, ``importance``, ``effort``, ``sla``, ``security``.

        Returns:
            A ``WorkItem`` with ``sort_index`` = negative score (for descending sort).
        """
        w = weights or {
            "urgency": 0.25,
            "importance": 0.25,
            "effort": -0.15,
            "sla": 0.25,
            "security": 0.35,
        }
        # Map slack: if very low, push score up (use inverse tanh-like cap)
        sla_term = 1.0 / (1.0 + max(0.0, sla_slack_s) / 60.0)
        raw = (
            w["urgency"] * urgency
            + w["importance"] * importance
            + w["effort"] * effort
            + w["sla"] * sla_term
            + w["security"] * security_severity
        )
        return WorkItem(
            sort_index=-raw,
            task_id=task_id,
            urgency=urgency,
            importance=importance,
            effort=effort,
            sla_slack_s=sla_slack_s,
            security_severity=security_severity,
        )


def order_queue(items: Iterable[WorkItem]) -> list[WorkItem]:
    """
    Return items highest priority first (max composite score).

    Args:
        items: Work items with ``sort_index`` set.

    Returns:
        Sorted list, highest priority first.
    """
    return sorted(items)


def main() -> None:
    """Demo queue for support / security / batch jobs."""
    q = [
        WorkItem.build("t1", 0.4, 0.5, 0.2, sla_slack_s=3600, security_severity=0.0),
        WorkItem.build("t2", 0.6, 0.3, 0.5, sla_slack_s=120, security_severity=0.0),
        WorkItem.build("t3", 0.3, 0.4, 0.3, sla_slack_s=600, security_severity=0.9),
    ]
    ordered = order_queue(q)
    print("processing order:", [x.task_id for x in ordered])


if __name__ == "__main__":
    main()
