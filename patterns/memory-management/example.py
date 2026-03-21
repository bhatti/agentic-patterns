#!/usr/bin/env python3
"""
Memory management — tiered in-memory stores (stdlib) + optional LangGraph checkpoint.

Install optional graph: pip install langgraph (see repo root requirements comments).

Reference: Antonio Gulli, Agentic Design Patterns — memory-manager agent spec.

Usage:
    python example.py
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - optional dependency
    MemorySaver = None  # type: ignore[misc,assignment]
    StateGraph = None  # type: ignore[misc,assignment]
    START = END = None  # type: ignore[misc,assignment]


@dataclass
class EpisodicEvent:
    """One episodic log row (demo)."""

    ts: datetime
    summary: str
    session_id: str


@dataclass
class TieredMemoryStore:
    """In-memory illustration of working + episodic + procedural + semantic slots."""

    working: list[str] = field(default_factory=list)
    episodic: list[EpisodicEvent] = field(default_factory=list)
    procedural: dict[str, str] = field(default_factory=dict)
    semantic: dict[str, str] = field(default_factory=dict)

    def append_working(self, line: str, max_lines: int = 20) -> None:
        """Push a line into working memory and trim from the head."""
        self.working.append(line)
        if len(self.working) > max_lines:
            self.working = self.working[-max_lines:]

    def log_episode(self, session_id: str, summary: str) -> None:
        """Record an episodic event with UTC timestamp."""
        self.episodic.append(
            EpisodicEvent(
                ts=datetime.now(timezone.utc),
                summary=summary,
                session_id=session_id,
            )
        )

    def recent_episodes(self, limit: int = 5) -> list[EpisodicEvent]:
        """Return most recent episodic events."""
        return self.episodic[-limit:]


def retrieve_episodic_snippets(store: TieredMemoryStore, query: str, limit: int = 3) -> list[str]:
    """
    Naive substring match over episodic summaries (stand-in for embedding search).

    Args:
        store: Backing store.
        query: User query text.
        limit: Max events to return.

    Returns:
        Matching summary strings, most recent first.
    """
    q = query.lower()
    hits = [e.summary for e in reversed(store.episodic) if q in e.summary.lower()]
    return hits[:limit]


class MemGraphState(TypedDict, total=False):
    """LangGraph state: incremental turn counter via reducer."""

    turn_count: Annotated[int, operator.add]


def _increment_turn(_: MemGraphState) -> MemGraphState:
    """Node: add one completed turn."""
    return {"turn_count": 1}


def build_langgraph_checkpoint_demo() -> Any:
    """
    Minimal compiled graph with in-memory checkpointer (thread_id scoped).

    Returns:
        Compiled LangGraph runnable.

    Raises:
        RuntimeError: If ``langgraph`` is not installed.
    """
    if StateGraph is None or MemorySaver is None or START is None:
        raise RuntimeError(
            "langgraph is required for build_langgraph_checkpoint_demo(); "
            "pip install langgraph"
        )
    builder = StateGraph(MemGraphState)
    builder.add_node("record_turn", _increment_turn)
    builder.add_edge(START, "record_turn")
    builder.add_edge("record_turn", END)
    return builder.compile(checkpointer=MemorySaver())


def main() -> None:
    """Run tiered memory demo and optional LangGraph checkpoint demo."""
    m = TieredMemoryStore()
    m.procedural["refund"] = "1. verify order 2. check policy 3. issue credit"
    m.semantic["user:42:timezone"] = "America/Los_Angeles"
    m.append_working("User asked about refund status.")
    m.log_episode("sess-1", "User requested refund for order 99")
    m.log_episode("sess-1", "User confirmed email on file")
    print("working tail:", m.working[-1])
    print("episodic retrieve 'refund':", retrieve_episodic_snippets(m, "refund"))

    try:
        graph = build_langgraph_checkpoint_demo()
        cfg = {"configurable": {"thread_id": "demo-thread"}}
        s1 = graph.invoke({}, cfg)
        s2 = graph.invoke({}, cfg)
        print("LangGraph checkpoint turn_count after 2 invokes:", s2.get("turn_count"))
        print("first invoke state:", s1)
    except RuntimeError as exc:
        print("LangGraph:", exc)


if __name__ == "__main__":
    main()
