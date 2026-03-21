#!/usr/bin/env python3
"""
Resource-aware optimization — budgets, tier selection, degradation modes (stdlib).

Illustrates policies for cost/latency; wire to real metrics (queue depth, $ spend).

Reference: Antonio Gulli, Agentic Design Patterns — resource-optimizer agent spec.

Usage:
    python example.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class DegradationMode(str, Enum):
    """Coarse service posture under resource pressure."""

    FULL = "full"
    REDUCED = "reduced"
    MINIMAL = "minimal"


@dataclass
class TokenBudget:
    """Per-request or per-session token ceiling."""

    limit: int
    used: int = 0

    def consume(self, n: int) -> None:
        """
        Record token usage.

        Args:
            n: Tokens to add toward the limit.

        Raises:
            ValueError: If usage would exceed the limit.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if self.used + n > self.limit:
            raise ValueError("token budget exceeded")
        self.used += n

    def remaining(self) -> int:
        """Return tokens left before the limit."""
        return max(0, self.limit - self.used)


@dataclass
class ModelTier:
    """Named model option with relative cost and latency scores (unitless)."""

    name: str
    relative_cost_per_1k_tokens: float
    relative_latency_ms: float


TIERS: dict[str, ModelTier] = {
    "small": ModelTier("small", relative_cost_per_1k_tokens=1.0, relative_latency_ms=40.0),
    "large": ModelTier("large", relative_cost_per_1k_tokens=8.0, relative_latency_ms=200.0),
}


def select_model_tier(*, priority: str, budget_remaining: int, pressure: float) -> str:
    """
    Choose a model tier from heuristics (demo policy).

    Args:
        priority: ``"high"`` or other—high keeps large longer if budget allows.
        budget_remaining: Tokens left in the active budget.
        pressure: Load signal in ``[0, 1]`` (higher → prefer smaller/faster).

    Returns:
        ``"small"`` or ``"large"`` key into ``TIERS``.
    """
    if budget_remaining < 500 or pressure > 0.85:
        return "small"
    if priority == "high" and pressure < 0.5:
        return "large"
    return "small" if pressure > 0.45 else "large"


def estimate_cost_usd(
    tokens_in: int,
    tokens_out: int,
    price_in_per_1k: float,
    price_out_per_1k: float,
) -> float:
    """
    Linear cost estimate from token counts and per-1k prices.

    Args:
        tokens_in: Prompt tokens billed.
        tokens_out: Completion tokens billed.
        price_in_per_1k: USD per 1k input tokens.
        price_out_per_1k: USD per 1k output tokens.

    Returns:
        Estimated cost in USD.
    """
    return (tokens_in / 1000.0) * price_in_per_1k + (tokens_out / 1000.0) * price_out_per_1k


def prune_messages_to_budget(messages: list[str], max_chars: int) -> list[str]:
    """
    Drop oldest messages until total chars fit (context pruning stub).

    Production: summarize middle turns instead of only dropping.

    Args:
        messages: Ordered chat turns (oldest first).
        max_chars: Character budget for context.

    Returns:
        A suffix of ``messages`` that fits ``max_chars``.
    """
    total = 0
    kept: list[str] = []
    for m in reversed(messages):
        if total + len(m) > max_chars:
            break
        kept.append(m)
        total += len(m)
    return list(reversed(kept))


def degradation_mode(
    *,
    cpu_util: float,
    queue_depth: int,
    queue_threshold: int = 100,
) -> DegradationMode:
    """
    Map coarse infra signals to a degradation level.

    Args:
        cpu_util: CPU fraction in ``[0, 1]``.
        queue_depth: Pending requests (or similar backlog).
        queue_threshold: Depth at which to escalate degradation.

    Returns:
        A ``DegradationMode`` for policy gates (tools, model tier, verbosity).
    """
    if cpu_util > 0.9 or queue_depth > queue_threshold * 2:
        return DegradationMode.MINIMAL
    if cpu_util > 0.75 or queue_depth > queue_threshold:
        return DegradationMode.REDUCED
    return DegradationMode.FULL


@dataclass
class ToolGate:
    """Adaptive tool allowlist under degradation."""

    allow_web: bool = True
    allow_code_run: bool = True

    def apply_mode(self, mode: DegradationMode) -> None:
        """
        Restrict tools based on service degradation level.

        Args:
            mode: Current ``DegradationMode``.
        """
        if mode == DegradationMode.REDUCED:
            self.allow_code_run = False
        elif mode == DegradationMode.MINIMAL:
            self.allow_web = False
            self.allow_code_run = False


@dataclass
class ParallelCostEstimate:
    """Rough comparison of sequential vs parallel frontier calls."""

    sequential_calls: int
    parallel_workers: int
    cost_per_call: float = 1.0

    def sequential_total(self) -> float:
        """Total cost if calls run one after another."""
        return self.sequential_calls * self.cost_per_call

    def parallel_wall_units(self) -> float:
        """Abstract time units as ceil(n/k) when each call has unit cost."""
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be >= 1")
        batches = math.ceil(self.sequential_calls / self.parallel_workers)
        return batches * self.cost_per_call


def main() -> None:
    """Print demo outputs for budgets, tiers, pruning, degradation."""
    b = TokenBudget(limit=4000)
    b.consume(100)
    print("token budget remaining:", b.remaining())

    print("model tier (high prio, ok budget, low pressure):", select_model_tier(priority="high", budget_remaining=3000, pressure=0.2))
    print("model tier (pressure high):", select_model_tier(priority="low", budget_remaining=3000, pressure=0.9))

    print("estimated $:", round(estimate_cost_usd(2000, 500, 0.003, 0.015), 6))

    long_thread = ["msg " * 50] * 10
    pruned = prune_messages_to_budget(long_thread, max_chars=200)
    print("pruned messages count:", len(pruned))

    mode = degradation_mode(cpu_util=0.8, queue_depth=150)
    gates = ToolGate()
    gates.apply_mode(mode)
    print("degradation:", mode, "allow_web", gates.allow_web, "allow_code_run", gates.allow_code_run)

    pe = ParallelCostEstimate(sequential_calls=8, parallel_workers=4)
    print("parallel wall units vs sequential cost units:", pe.parallel_wall_units(), pe.sequential_total())


if __name__ == "__main__":
    main()
