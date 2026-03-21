#!/usr/bin/env python3
"""
Long-term memory — four memory types (working, episodic, procedural, semantic)

Problem: Stateless LLM calls do not retain context unless you inject it. Agents
simulate state by passing history or explicit state; durable products need
structured memory.

This example uses stdlib-only in-memory structures to show:
  - Working memory: recent dialogue turns (session buffer)
  - Episodic memory: timestamped events / conversations
  - Procedural memory: named task playbooks (how-to steps)
  - Semantic memory: keyword-addressable facts (production: embed + vector DB)

For production Mem0 (extract + embed + search), see the book reference
``examples/28_long_term_memory/basic_memory.py`` and ``pip install mem0ai``.

Usage:
    python example.py
"""

from __future__ import annotations

import re
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass
class EpisodicEvent:
    """A single user-visible episode for audit and recall."""

    ts: datetime
    summary: str
    session_id: str


@dataclass
class MemoryBundle:
    """Working + long-term slices assembled for a prompt."""

    working_block: str
    episodic_block: str
    procedural_block: str
    semantic_block: str

    def as_system_context(self) -> str:
        """
        Format all layers for injection into a system or developer message.

        Returns:
            Single string block for the model.
        """
        parts = [
            "### Working memory (recent turns)",
            self.working_block,
            "### Episodic memory (what happened)",
            self.episodic_block,
            "### Procedural memory (how to)",
            self.procedural_block,
            "### Semantic memory (stable facts)",
            self.semantic_block,
        ]
        return "\n".join(parts)


class WorkingMemory:
    """
    Short sliding window of recent user/assistant turns (session-scoped).

    This approximates the limited context humans use for *current* focus.
    """

    def __init__(self, max_turns: int = 6) -> None:
        """
        Args:
            max_turns: Maximum number of user+assistant pairs to retain as strings.
        """
        self._max_turns = max(1, max_turns)
        self._lines: deque[str] = deque(maxlen=self._max_turns * 2)

    def append_user(self, text: str) -> None:
        """Record a user utterance."""
        self._lines.append(f"User: {text.strip()}")

    def append_assistant(self, text: str) -> None:
        """Record an assistant utterance."""
        self._lines.append(f"Assistant: {text.strip()}")

    def format_block(self) -> str:
        """Return a compact transcript for prompting."""
        if not self._lines:
            return "(empty)"
        return "\n".join(self._lines)


class EpisodicMemory:
    """Append-only log of interactions for 'what happened' and timelines."""

    def __init__(self) -> None:
        """Initialize an empty log."""
        self._events: list[EpisodicEvent] = []

    def add(self, summary: str, session_id: str) -> None:
        """
        Record an episode summary.

        Args:
            summary: Short description of the interaction.
            session_id: Logical session identifier.
        """
        self._events.append(
            EpisodicEvent(
                ts=datetime.now(timezone.utc),
                summary=summary.strip(),
                session_id=session_id,
            )
        )

    def recent(self, limit: int = 5) -> list[EpisodicEvent]:
        """Return the last ``limit`` events (oldest first within slice)."""
        if limit <= 0:
            return []
        return self._events[-limit:]

    def format_block(self, limit: int = 5) -> str:
        """Format recent episodes for prompting."""
        lines = []
        for ev in self.recent(limit):
            ts = ev.ts.isoformat()
            lines.append(f"- [{ts}] ({ev.session_id}) {ev.summary}")
        return "\n".join(lines) if lines else "(none)"


class ProceduralMemory:
    """Named procedures: tool recipes, checklists, org-specific workflows."""

    def __init__(self) -> None:
        """Initialize empty procedures."""
        self._playbooks: dict[str, list[str]] = {}

    def set_playbook(self, name: str, steps: list[str]) -> None:
        """
        Store ordered steps for a task.

        Args:
            name: Task key (e.g. 'onboard_vendor').
            steps: Ordered human-readable steps.
        """
        self._playbooks[name.strip()] = [s.strip() for s in steps if s.strip()]

    def get(self, name: str) -> Optional[list[str]]:
        """Return steps for ``name`` if present."""
        return self._playbooks.get(name)

    def format_relevant(self, names: list[str]) -> str:
        """Format named playbooks for injection."""
        chunks = []
        for n in names:
            steps = self._playbooks.get(n)
            if not steps:
                continue
            body = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
            chunks.append(f"**{n}**\n{body}")
        return "\n\n".join(chunks) if chunks else "(none)"


class SemanticMemory:
    """
    Stable facts keyed by short phrases (production: embeddings + vector DB).

    Mem0 automates extraction and search over this layer.
    """

    def __init__(self) -> None:
        """Initialize an empty fact store."""
        self._facts: dict[str, str] = {}

    def upsert(self, key: str, value: str) -> None:
        """
        Store or update a fact.

        Args:
            key: Short label (e.g. 'user_preferred_language').
            value: Fact text.
        """
        self._facts[key.strip()] = value.strip()

    def get(self, key: str) -> Optional[str]:
        """Return fact by key."""
        return self._facts.get(key)

    def search_keywords(self, query: str, limit: int = 8) -> list[tuple[str, str]]:
        """
        Naive token overlap (splits identifiers on non-alphanumeric boundaries).

        Production systems use embeddings (e.g. Mem0 + vector DB).

        Args:
            query: User query text.
            limit: Max number of facts to return.

        Returns:
            List of (key, value) pairs ranked by simple overlap score.
        """

        def lex(s: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", s.lower()))

        q = lex(query)
        scored: list[tuple[int, str, str]] = []
        for k, v in self._facts.items():
            hay = lex(k + " " + v)
            score = len(q & hay)
            if score > 0:
                scored.append((score, k, v))
        scored.sort(key=lambda x: -x[0])
        return [(k, v) for _, k, v in scored[:limit]]

    def format_block(self, query: str) -> str:
        """Format top keyword hits for prompting."""
        hits = self.search_keywords(query)
        if not hits:
            return "(no matching facts)"
        lines = [f"- {k}: {v}" for k, v in hits]
        return "\n".join(lines)


class LongTermMemoryOrchestrator:
    """Compose the four memory types into one injection bundle."""

    def __init__(self, session_id: str) -> None:
        """
        Args:
            session_id: Identifier for episodic scoping.
        """
        self.session_id = session_id
        self.working = WorkingMemory(max_turns=4)
        self.episodic = EpisodicMemory()
        self.procedural = ProceduralMemory()
        self.semantic = SemanticMemory()

    def prime_demo_data(self) -> None:
        """Load illustrative procedural and semantic content."""
        self.procedural.set_playbook(
            "refund_request",
            [
                "Verify order id in OMS.",
                "Check policy window (30 days).",
                "If eligible, issue refund or store credit per SKU.",
            ],
        )
        self.semantic.upsert("user_display_name", "Alex")
        self.semantic.upsert("user_timezone", "America/Los_Angeles")
        self.semantic.upsert("dietary_restriction", "Avoid peanuts in meal suggestions.")

    def build_bundle(self, user_query: str, playbook_names: list[str]) -> MemoryBundle:
        """
        Assemble a ``MemoryBundle`` for the current user message.

        Args:
            user_query: Latest user text.
            playbook_names: Procedural playbooks to include.

        Returns:
            Bundle for system prompt assembly.
        """
        return MemoryBundle(
            working_block=self.working.format_block(),
            episodic_block=self.episodic.format_block(),
            procedural_block=self.procedural.format_relevant(playbook_names),
            semantic_block=self.semantic.format_block(user_query),
        )

    def record_turn(self, user_text: str, assistant_text: str, episode_summary: str) -> None:
        """
        Append a turn to working memory and episodic log.

        Args:
            user_text: User message.
            assistant_text: Assistant reply.
            episode_summary: Short summary for episodic store.
        """
        self.working.append_user(user_text)
        self.working.append_assistant(assistant_text)
        self.episodic.add(episode_summary, self.session_id)


def main() -> None:
    """Demonstrate a two-turn conversation with layered memory."""
    print("Pattern 28: Long-term memory (four types)\n")

    orch = LongTermMemoryOrchestrator(session_id="sess_001")
    orch.prime_demo_data()

    q1 = "Hi — what timezone am I in for scheduling?"
    b1 = orch.build_bundle(q1, playbook_names=["refund_request"])
    print("--- Turn 1 (before recording) ---")
    print(b1.as_system_context())
    orch.record_turn(
        user_text=q1,
        assistant_text="(mock) You prefer America/Los_Angeles.",
        episode_summary="User asked about timezone; answered with stored preference.",
    )

    q2 = (
        "I need a refund for order ORD-123. Remind me of my timezone and dietary restriction."
    )
    b2 = orch.build_bundle(q2, playbook_names=["refund_request"])
    print("\n--- Turn 2 (after one turn in working + episodic) ---")
    print(b2.as_system_context())
    print("\nMem0: use Memory.from_config(...) and memory.add/search per user_id — see book basic_memory.py.")


if __name__ == "__main__":
    main()
