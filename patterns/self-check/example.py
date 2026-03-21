#!/usr/bin/env python3
"""
Self-check — logprobs, logits→softmax, perplexity, low-confidence token flags

Problem: LLM outputs may be fabricated or unstable; when the API exposes token
log-probabilities, you can flag uncertain spans for review or re-retrieval.

This file is pure math (no API keys):
  - softmax from logits
  - log p of the chosen token
  - sequence perplexity from per-token log p
  - flag tokens with probability below a threshold

Production: request logprobs (and top_logprobs) from your provider; see book
``1_hallucination_detection.ipynb`` / ``2_self_check.ipynb`` for OpenAI-style parsing.

Usage:
    python example.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def softmax(logits: list[float]) -> list[float]:
    """
    Map a vector of logits to a probability distribution.

    Args:
        logits: Unnormalized scores for each class/token.

    Returns:
        Probabilities summing to 1.
    """
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp((x - m)) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def logprob_of_chosen_token(probs: list[float], index: int) -> float:
    """
    Natural log of the probability mass at ``index`` (with clamping).

    Args:
        probs: Probability distribution.
        index: Chosen token index.

    Returns:
        log p (natural log).
    """
    p = probs[index]
    eps = 1e-12
    return math.log(max(p, eps))


def sequence_perplexity(token_logprobs: Iterable[float]) -> float:
    """
    Compute perplexity from per-token log-probabilities log p(w_i | prefix).

    PPL = exp(- (1/N) * sum_i log p_i) = exp(mean negative log-likelihood).

    Args:
        token_logprobs: Natural log of each selected token's probability.

    Returns:
        Scalar perplexity (>= 1 for typical distributions).
    """
    logs = list(token_logprobs)
    if not logs:
        return float("inf")
    n = len(logs)
    mean_nll = -sum(logs) / n
    return math.exp(mean_nll)


def flag_low_confidence_tokens(
    tokens: list[str],
    token_logprobs: list[float],
    *,
    min_prob: float = 0.2,
) -> list[tuple[int, str, float]]:
    """
    Return indices where exp(logprob) < min_prob.

    Args:
        tokens: Token strings (same length as token_logprobs).
        token_logprobs: log p for each selected token.
        min_prob: Probability floor below which a token is flagged.

    Returns:
        List of (index, token, probability).
    """
    if len(tokens) != len(token_logprobs):
        raise ValueError("tokens and token_logprobs length mismatch")
    out: list[tuple[int, str, float]] = []
    for i, (tok, lp) in enumerate(zip(tokens, token_logprobs, strict=True)):
        p = math.exp(lp)
        if p < min_prob:
            out.append((i, tok, p))
    return out


def demo_from_logits() -> None:
    """Show logits → softmax → logprob for one position."""
    # Toy: vocab size 4; model picked index 2
    logits = [0.2, -1.0, 2.5, 0.0]
    probs = softmax(logits)
    chosen = 2
    lp = logprob_of_chosen_token(probs, chosen)
    print("--- Toy logits → softmax ---")
    print("  logits:", [round(x, 2) for x in logits])
    print("  probs:", [round(p, 4) for p in probs])
    print("  chosen index:", chosen, "log p:", round(lp, 4), "p:", round(math.exp(lp), 4))


def demo_perplexity() -> None:
    """Show perplexity on a short synthetic sequence."""
    # Higher confidence (less negative log p) → lower perplexity
    logps_high = [-0.15, -0.22, -0.18, -0.20]
    logps_low = [-1.2, -2.0, -0.9, -1.5]
    print("\n--- Perplexity (synthetic token log p) ---")
    print("  confident-ish seq PPL:", round(sequence_perplexity(logps_high), 3))
    print("  uncertain seq PPL:     ", round(sequence_perplexity(logps_low), 3))


def demo_flag_tokens() -> None:
    """Flag subword tokens with low probability."""
    tokens = ["The", " capital", " of", " France", " is", " London", "."]
    # synthetic log p: last content word is wrong and low-prob
    logps = [-0.2, -0.3, -0.1, -0.4, -0.2, -2.0, -0.15]
    flagged = flag_low_confidence_tokens(tokens, logps, min_prob=0.25)
    print("\n--- Low-confidence flags (min_prob=0.25) ---")
    for idx, tok, p in flagged:
        print(f"  idx={idx} p={p:.3f} tok={repr(tok)}")
    print("  (Low p does not prove factual error—only model uncertainty.)")


def main() -> None:
    """Run demos."""
    print("Pattern 31: Self-check (logprobs & perplexity)\n")
    demo_from_logits()
    demo_perplexity()
    demo_flag_tokens()
    print("\nNote: combine with RAG, tools, and Pattern 30 for factual grounding.")


if __name__ == "__main__":
    main()
