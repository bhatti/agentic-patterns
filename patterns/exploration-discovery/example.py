#!/usr/bin/env python3
"""Exploration — epsilon-greedy stub over discrete actions. Usage: python example.py"""

from __future__ import annotations

import random


def epsilon_greedy_select(
    *,
    epsilon: float,
    action_rewards: dict[str, float],
    rng: random.Random | None = None,
) -> str:
    """
    With probability epsilon pick a random action; else pick best known reward.

    Args:
        epsilon: Exploration probability in [0, 1].
        action_rewards: Estimated mean reward per action id.
        rng: Optional RNG for tests.

    Returns:
        Chosen action id.
    """
    r = rng or random.Random()
    actions = list(action_rewards.keys())
    if r.random() < epsilon:
        return r.choice(actions)
    return max(action_rewards, key=lambda a: action_rewards[a])


def main() -> None:
    rewards = {"search_web": 0.4, "read_docs": 0.7, "ask_user": 0.2}
    rng = random.Random(0)
    print([epsilon_greedy_select(epsilon=0.3, action_rewards=rewards, rng=rng) for _ in range(5)])


if __name__ == "__main__":
    main()
