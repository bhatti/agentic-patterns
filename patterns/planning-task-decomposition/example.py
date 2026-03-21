#!/usr/bin/env python3
"""Planning — task DAG and topological order (stdlib). See README. Usage: python example.py"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlannedTask:
    """A node: task_id depends on completing all tasks in depends_on first."""

    task_id: str
    depends_on: set[str] = field(default_factory=set)


def topological_order(tasks: dict[str, PlannedTask]) -> list[str]:
    """
    Kahn topological sort: valid linear order respecting dependencies.

    Args:
        tasks: Map task_id -> PlannedTask.

    Returns:
        Task ids in execution order.

    Raises:
        ValueError: Unknown dependency or cycle detected.
    """
    for t in tasks.values():
        for d in t.depends_on:
            if d not in tasks:
                raise ValueError(f"unknown dependency {d!r} for {t.task_id!r}")

    remaining: dict[str, int] = {tid: len(tasks[tid].depends_on) for tid in tasks}
    queue: list[str] = [tid for tid in tasks if remaining[tid] == 0]
    out: list[str] = []
    while queue:
        n = queue.pop(0)
        out.append(n)
        for t in tasks.values():
            if n in t.depends_on:
                remaining[t.task_id] -= 1
                if remaining[t.task_id] == 0:
                    queue.append(t.task_id)
    if len(out) != len(tasks):
        raise ValueError("cycle in task graph")
    return out


def main() -> None:
    g = {
        "ingest": PlannedTask("ingest"),
        "analyze": PlannedTask("analyze", {"ingest"}),
        "report": PlannedTask("report", {"analyze"}),
    }
    print("plan order:", topological_order(g))


if __name__ == "__main__":
    main()
