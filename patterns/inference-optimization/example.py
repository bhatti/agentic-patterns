#!/usr/bin/env python3
"""
Inference optimization — batching overhead, prompt compression, pointers to serving stacks

Problem: Self-hosted LLMs need high throughput and manageable KV memory. Fixed-shape
batching wastes work on variable-length sequences; long prompts bloat the KV cache.

This educational script (stdlib only) demonstrates:
  1) Padding overhead for a naive "batch to max length" schedule.
  2) A simple prompt compression pass (dedupe lines / whitespace) and size reduction.
  3) Printed notes on continuous batching (vLLM, SGLang) and speculative decoding.

Usage:
    python example.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def padded_batch_token_cost(seq_lengths: list[int]) -> tuple[int, int, float]:
    """
    Approximate token "slots" used if every sequence in a batch is padded to max length.

    This models the waste static batching incurs when lengths differ. Continuous batching
    avoids much of this by dynamic scheduling and block-wise KV (see vLLM PagedAttention).

    Args:
        seq_lengths: Prompt lengths (in tokens or arbitrary comparable units).

    Returns:
        Tuple of (padded_total, sum_of_lengths, efficiency_ratio sum/padded).
    """
    if not seq_lengths:
        return 0, 0, 1.0
    max_len = max(seq_lengths)
    padded_total = len(seq_lengths) * max_len
    sum_len = sum(seq_lengths)
    ratio = sum_len / padded_total if padded_total else 1.0
    return padded_total, sum_len, ratio


def compress_prompt_lightweight(text: str) -> str:
    """
    Reduce prompt size for inference by normalizing whitespace and dropping duplicate lines.

    Production systems may add summarization, retrieval distillation, or learned compressors.
    Always measure downstream task quality after aggressive compression.

    Args:
        text: Raw prompt or injected context.

    Returns:
        Shorter string suitable for re-injection into the model.
    """
    lines = text.splitlines()
    seen: set[str] = set()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        key = stripped.casefold()
        if key in seen:
            continue
        seen.add(key)
        kept.append(stripped)
    body = "\n".join(kept)
    body = re.sub(r"[ \t]+", " ", body)
    return body.strip()


def summarize_batching_scenarios() -> list[tuple[str, list[int], str]]:
    """
    Build a few illustrative batch length scenarios.

    Returns:
        List of (label, lengths, note).
    """
    return [
        ("similar_lengths", [512, 520, 505, 518], "Padding waste is modest."),
        ("mixed_lengths", [128, 4096, 256, 2048], "Naive pad-to-max wastes most of the batch."),
        ("many_short", [32, 48, 40, 36, 44, 50, 38], "Short prompts; still pad to max within batch."),
    ]


def serving_stack_notes() -> str:
    """
    Return documentation for continuous batching stacks (no imports).

    Returns:
        Multi-line note for stdout.
    """
    return (
        "Continuous batching: vLLM (PagedAttention + iteration-level scheduling), "
        "SGLang, TensorRT-LLM, TGI — pick based on model family and ops constraints.\n"
        "Speculative decoding: see patterns/small-language-model (Pattern 24) and "
        "examples/26_inference_optimization/speculative_decoding.py (vLLM)."
    )


def main() -> None:
    """Print batching efficiency toy metrics, a compression demo, and serving notes."""
    print("Pattern 26: Inference optimization\n")

    print("1) Naive pad-to-max within a single batch (toy units = tokens)")
    for label, lengths, note in summarize_batching_scenarios():
        padded, sum_len, eff = padded_batch_token_cost(lengths)
        print(f"   [{label}] n={len(lengths)}  padded_total={padded}  sum_lengths={sum_len}")
        print(f"      efficiency sum/padded={eff:.3f}  — {note}")

    bloated = """

    SYSTEM: You are a helpful assistant.

    CONTEXT: The policy requires logging all access.

    CONTEXT: The policy requires logging all access.

    USER: Summarize the duplicate policy lines above in one sentence.


    """
    compressed = compress_prompt_lightweight(bloated)
    print("\n2) Lightweight prompt compression (dedupe lines + spaces)")
    print("   before chars:", len(bloated.strip()))
    print("   after chars: ", len(compressed))
    print("   preview:\n   ", compressed[:200].replace("\n", " "), "...")

    print("\n3) Serving stacks & speculative decoding")
    print("   ", serving_stack_notes())


if __name__ == "__main__":
    main()
