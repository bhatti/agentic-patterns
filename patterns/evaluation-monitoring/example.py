#!/usr/bin/env python3
"""
Evaluation and monitoring — token tallies, latency stats, A/B assignment (stdlib).

No network or vendor SDKs; wire MetricSample to your OTLP/Prometheus stack.

Reference: Antonio Gulli, Agentic Design Patterns — evaluator agent spec.

Usage:
    python example.py
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricSample:
    """Single scalar metric with dimensions (tags must stay low-cardinality)."""

    name: str
    value: float
    timestamp_s: float
    tags: dict[str, str] = field(default_factory=dict)


class TokenUsageTracker:
    """Accumulate prompt and completion token counts (application-level billing)."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        """
        Add usage from one LLM call.

        Args:
            prompt_tokens: Input tokens billed.
            completion_tokens: Output tokens billed.
        """
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def total(self) -> int:
        """Return sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens


class LatencyWindow:
    """Track recent end-to-end latencies in milliseconds (naive p50/p95)."""

    def __init__(self, max_samples: int = 10_000) -> None:
        self._max = max_samples
        self._ms: list[float] = []

    def record(self, duration_ms: float) -> None:
        """
        Append one observation.

        Args:
            duration_ms: Elapsed time in milliseconds.
        """
        if duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        self._ms.append(duration_ms)
        if len(self._ms) > self._max:
            self._ms.pop(0)

    def percentile(self, p: float) -> float | None:
        """
        Return approximate percentile (0–100) or None if empty.

        Args:
            p: Percentile line (e.g. 95 for p95).

        Returns:
            Latency in ms at ``p``, or None.
        """
        if not self._ms:
            return None
        if not 0 <= p <= 100:
            raise ValueError("p must be in [0, 100]")
        sorted_ms = sorted(self._ms)
        k = (len(sorted_ms) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_ms) - 1)
        return sorted_ms[f] + (k - f) * (sorted_ms[c] - sorted_ms[f])


def ab_variant(user_key: str, experiment_name: str, variants: tuple[str, ...]) -> str:
    """
    Deterministic A/B assignment from stable identifiers (no PII in user_key).

    Args:
        user_key: Opaque stable id (hashed internally).
        experiment_name: Experiment bucket namespace.
        variants: Named arms (e.g. ("control", "treatment")).

    Returns:
        One of ``variants``.
    """
    if not variants:
        raise ValueError("variants must be non-empty")
    h = hashlib.sha256(f"{experiment_name}:{user_key}".encode()).hexdigest()
    bucket = int(h[:8], 16) % len(variants)
    return variants[bucket]


def compliance_evidence_stub(
    *,
    trace_id: str,
    policy_pack: str,
    checks_passed: list[str],
) -> dict[str, Any]:
    """
    Build a minimal evidence record for audits (extend with signatures, storage URIs).

    Args:
        trace_id: Distributed trace id.
        policy_pack: Named policy version.
        checks_passed: Human-readable check names that succeeded.

    Returns:
        Serializable dict suitable for an audit store.
    """
    return {
        "trace_id": trace_id,
        "policy_pack": policy_pack,
        "checks_passed": list(checks_passed),
        "recorded_at_s": time.time(),
    }


def multi_agent_span_stub(
    workflow: str,
    agent_steps: list[tuple[str, float]],
) -> list[MetricSample]:
    """
    Emit synthetic per-agent latency metrics for dashboards (demo only).

    Args:
        workflow: Workflow name tag.
        agent_steps: (agent_name, duration_ms) pairs.

    Returns:
        MetricSample list.
    """
    now = time.time()
    out: list[MetricSample] = []
    for agent, ms in agent_steps:
        out.append(
            MetricSample(
                name="agent_step_latency_ms",
                value=ms,
                timestamp_s=now,
                tags={"workflow": workflow, "agent": agent},
            )
        )
    return out


def main() -> None:
    """Exercise helpers."""
    tok = TokenUsageTracker()
    tok.record(120, 45)
    tok.record(200, 60)
    print("total tokens:", tok.total())

    win = LatencyWindow()
    for x in (10, 12, 50, 11, 300, 15):
        win.record(float(x))
    print("p50 ms:", win.percentile(50), "p95 ms:", win.percentile(95))

    print("A/B:", ab_variant("user-42", "prompt_v2", ("control", "treatment")))

    print("compliance:", compliance_evidence_stub(trace_id="tr-1", policy_pack="v2025-01", checks_passed=["pii_scan", "moderation"]))

    spans = multi_agent_span_stub("research_crew", [("planner", 80.0), ("retriever", 220.0)])
    print("spans:", len(spans), "first tags:", spans[0].tags)


if __name__ == "__main__":
    main()
