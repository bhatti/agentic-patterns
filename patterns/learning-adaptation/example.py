#!/usr/bin/env python3
"""
Learning and adaptation — PPO clipped surrogate (illustrative) and DPO-style loss stub.

Educational scalars only: not a full RL trainer or DPO trainer. Swap in your
framework (e.g. PyTorch, TRL) for real policy and preference optimization.

Reference: Antonio Gulli, Agentic Design Patterns — learning-adapter agent spec.

Usage:
    python example.py
"""

from __future__ import annotations

import math
from typing import Iterable


def ppo_clipped_surrogate_loss(
    ratios: Iterable[float],
    advantages: Iterable[float],
    epsilon: float,
) -> float:
    """
    Scalar PPO-style objective: mean of -min(r*A, clip(r)*A) over samples.

    ``ratio`` is pi_theta(a|s) / pi_old(a|s); clipping limits how far the
    effective ratio can move, stabilizing updates (trust-region behavior).

    Args:
        ratios: Importance sampling ratios per timestep or token batch element.
        advantages: Advantage estimates (e.g. GAE) aligned with ``ratios``.
        epsilon: Clip range parameter (often denoted epsilon in PPO).

    Returns:
        A scalar loss to **minimize** (negative of the usual clipped surrogate).
    """
    rs = list(ratios)
    advs = list(advantages)
    if len(rs) != len(advs) or not rs:
        raise ValueError("ratios and advantages must be non-empty and equal length")
    losses: list[float] = []
    lower = 1.0 - epsilon
    upper = 1.0 + epsilon
    for r, adv in zip(rs, advs, strict=True):
        unclipped = r * adv
        r_clipped = max(lower, min(upper, r))
        clipped = r_clipped * adv
        losses.append(-min(unclipped, clipped))
    return sum(losses) / len(losses)


def dpo_style_preference_loss(
    logp_preferred: float,
    logp_rejected: float,
    beta: float,
) -> float:
    """
    Simplified DPO-style term: -log sigmoid(beta * (log pi(y_w) - log pi(y_l))).

    Increasing ``logp_preferred`` relative to ``logp_rejected`` lowers this loss.
    Full DPO includes reference-model KL shaping; this stub isolates the core
    preference ranking term for intuition.

    Args:
        logp_preferred: Sum (or mean) log-probability of the **chosen** response.
        logp_rejected: Sum (or mean) log-probability of the **rejected** response.
        beta: Temperature controlling how sharp the preference is.

    Returns:
        A scalar loss to minimize.
    """
    delta = beta * (logp_preferred - logp_rejected)
    # numerically stable: -log sigmoid(delta) = log(1 + exp(-delta))
    if delta >= 0:
        return math.log(1.0 + math.exp(-delta))
    return -delta + math.log(1.0 + math.exp(delta))


def main() -> None:
    """Print illustrative PPO and DPO scalar values."""
    ratios = [0.9, 1.1, 1.4, 0.8]
    advantages = [0.5, -0.2, 1.0, -0.3]
    eps = 0.2
    ppo_loss = ppo_clipped_surrogate_loss(ratios, advantages, eps)
    print("PPO clipped surrogate loss (minimize):", round(ppo_loss, 6))
    print("  (ratios clipped to [{}, {}] per term before min with unclipped)".format(1 - eps, 1 + eps))

    # Preferred completion gets higher log prob under policy → lower DPO-style loss
    l_w, l_l = -2.1, -3.4
    beta = 0.5
    dpo = dpo_style_preference_loss(l_w, l_l, beta)
    print("DPO-style preference loss (minimize):", round(dpo, 6))
    print("  logp_preferred:", l_w, "logp_rejected:", l_l, "beta:", beta)


if __name__ == "__main__":
    main()
