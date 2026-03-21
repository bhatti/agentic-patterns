#!/usr/bin/env python3
"""
Prompt Optimization Pattern - Support Ticket One-Line Summary

This example demonstrates the four components of prompt optimization:
(1) Pipeline of steps that use the prompt, (2) Dataset to evaluate on,
(3) Evaluator that scores outputs, (4) Optimizer that searches for the best
prompt. When you change the foundational model, you re-run the optimizer
instead of redoing all manual prompt trials.

Real-World Problem:
-------------------
You need one-line summaries of support tickets. Prompt wording ("Summarize
in one sentence" vs "Write a one-sentence summary") and model changes
affect quality. Instead of hand-tweaking and re-testing every time the
model changes, we run an optimization loop: try candidate prompts on a
fixed dataset, score with an evaluator, pick the best prompt.

Four components:
- Pipeline: prompt template + ticket -> summary (here simulated; production: LLM).
- Dataset: list of sample tickets.
- Evaluator: score summary (length, key-info presence) -> 0-1.
- Optimizer: try N candidate prompts, average score on dataset, return best.

Usage:
    python example.py
    # Uses simulated pipeline so it runs without LLM. In production, pipeline
    # would call the real model; optimizer would be run when the model changes.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

# Project root for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# =============================================================================
# 1. PIPELINE (prompt + input -> output; prompt is the parameter we optimize)
# =============================================================================

def run_pipeline(
    prompt_template: str,
    ticket: str,
    generate_fn: Callable[[str, str], str] | None = None,
) -> str:
    """
    Pipeline: given a prompt template and ticket, produce a one-line summary.
    In production, generate_fn would call the LLM with prompt_template + ticket.
    Here we simulate so the example runs without an API.
    """
    if generate_fn is not None:
        return generate_fn(prompt_template, ticket)

    # Simulated: vary length by prompt wording so optimizer can prefer one (mock LLM behavior)
    words = re.sub(r"[^\w\s]", "", ticket).split()
    if "short" in prompt_template.lower() or "one line" in prompt_template.lower():
        words = words[:5]
    elif "one sentence" in prompt_template.lower():
        words = words[:8]
    else:
        words = words[:10]
    return " ".join(words).strip() or "No summary"


# =============================================================================
# 2. DATASET (inputs to evaluate on)
# =============================================================================

def get_dataset() -> List[str]:
    """Fixed dataset of support tickets. In production, use a curated dev set."""
    return [
        "Hi, I was charged twice for my subscription last month. I only want one charge. Can you refund the extra?",
        "My login does not work. I reset my password but still cannot access the dashboard.",
        "Order #88492 has not arrived. It was supposed to be delivered last week. Can you track it?",
        "I need to update my billing address and payment method for my account.",
        "The app crashes when I open the reports tab. I am on iOS 17.",
    ]


# =============================================================================
# 3. EVALUATOR (score output; the metric we optimize)
# =============================================================================

def evaluate_summary(summary: str, ticket: str) -> float:
    """
    Score a one-line summary for a ticket. Returns 0-1.
    Criteria: reasonable length (20-120 chars), and at least one important
    word from the ticket (refund, login, order, etc.).
    """
    score = 0.0
    length = len(summary)
    if 20 <= length <= 120:
        score += 0.5
    elif 10 <= length <= 150:
        score += 0.3
    # Prefer summaries that capture key intent
    ticket_lower = ticket.lower()
    summary_lower = summary.lower()
    keywords = ["refund", "charge", "login", "password", "order", "track", "billing", "crash", "app"]
    if any(kw in ticket_lower and kw in summary_lower for kw in keywords):
        score += 0.5
    elif any(kw in ticket_lower for kw in keywords) and len(summary) >= 15:
        score += 0.2
    return min(1.0, score)


# =============================================================================
# 4. OPTIMIZER (propose candidates, run pipeline on dataset, pick best)
# =============================================================================

def optimize_prompt(
    candidate_prompts: List[str],
    dataset: List[str],
    run_fn: Callable[[str, str], str],
    eval_fn: Callable[[str, str], float],
) -> tuple[str, float]:
    """
    Try each candidate prompt on the full dataset; average evaluator score;
    return the prompt with the highest average score.
    """
    best_prompt = candidate_prompts[0]
    best_avg = -1.0

    for prompt in candidate_prompts:
        scores: List[float] = []
        for ticket in dataset:
            output = run_fn(prompt, ticket)
            scores.append(eval_fn(output, ticket))
        avg = sum(scores) / len(scores) if scores else 0.0
        if avg > best_avg:
            best_avg = avg
            best_prompt = prompt
    return best_prompt, best_avg


# =============================================================================
# MAIN: Run optimization loop and show best prompt
# =============================================================================

def main() -> None:
    dataset = get_dataset()
    # Candidate prompts to optimize over (in practice: more variants or LLM-generated)
    candidates = [
        "Summarize this support ticket in one short sentence.",
        "Write a one-sentence summary of the customer's issue.",
        "In one line, what is this ticket about?",
    ]

    print("Prompt Optimization — Support ticket one-line summary")
    print("Components: Pipeline, Dataset, Evaluator, Optimizer")
    print()
    print("Dataset size:", len(dataset))
    print("Candidate prompts:", len(candidates))
    print()

    def run_fn(prompt: str, ticket: str) -> str:
        return run_pipeline(prompt, ticket)

    best_prompt, best_score = optimize_prompt(
        candidate_prompts=candidates,
        dataset=dataset,
        run_fn=run_fn,
        eval_fn=evaluate_summary,
    )

    print("Best prompt:", best_prompt)
    print("Average score on dataset:", f"{best_score:.2f}")
    print()
    print("When you change the foundational model, re-run the optimizer with")
    print("the same dataset and evaluator to find a prompt suited to the new model.")


if __name__ == "__main__":
    main()
