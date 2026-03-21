#!/usr/bin/env python3
"""Goal monitoring — progress vs target, deviation flag. Usage: python example.py"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Goal:
    """A measurable goal with optional deadline (unit-agnostic demo)."""

    goal_id: str
    target_value: float
    current_value: float
    unit: str = "score"


def progress_ratio(goal: Goal) -> float:
    """Return progress in [0, 1] toward target (higher is better)."""
    if goal.target_value <= 0:
        raise ValueError("target must be positive")
    return min(1.0, max(0.0, goal.current_value / goal.target_value))


def is_off_track(goal: Goal, threshold: float = 0.8) -> bool:
    """True if progress is below threshold of expected linear pace (caller sets pace)."""
    return progress_ratio(goal) < threshold


def main() -> None:
    g = Goal("q1_nps", target_value=50.0, current_value=32.0, unit="nps")
    print("progress:", progress_ratio(g), "off_track:", is_off_track(g, 0.7))


if __name__ == "__main__":
    main()
