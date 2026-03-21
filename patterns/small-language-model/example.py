#!/usr/bin/env python3
"""
Small language model (SLM) pattern — distillation (KL), memory footprint, speculative decoding

Problem: Frontier LLMs need large GPU pools and memory; many tasks are served well by
smaller models, distillation, quantization, or faster decoding via speculative decoding.

This script uses **stdlib only** (no NumPy) so it runs anywhere. It demonstrates:

1. **KL divergence** for matching a student distribution to a teacher (distillation target).
2. **Weight memory** estimates for FP32 / FP16 / 8-bit / 4-bit for a given parameter count.
3. A **speculative decoding** micro-simulation: draft proposes token prefixes; target
   verifies a fixed oracle sequence; we report draft acceptance rate.

For production training (teacher + student Gemma), **BitsAndBytes** loading, and **vLLM**
speculative decoding, see the book reference under ``examples/24_small_language_model``.

Usage:
    python example.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _softmax(logits: list[float], temperature: float) -> list[float]:
    """Apply temperature-scaled softmax to a 1-D logit vector."""
    t = float(temperature)
    if t <= 0.0:
        raise ValueError("temperature must be positive")
    m = max(x / t for x in logits)
    exps = [math.exp(x / t - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def kl_divergence(
    teacher_logits: list[float],
    student_logits: list[float],
    *,
    temperature: float = 1.0,
) -> float:
    """
    Compute KL( P_teacher || P_student ) over a vocabulary for one position.

    Args:
        teacher_logits: Logits from the teacher model.
        student_logits: Logits from the student model (same length).
        temperature: Softmax temperature (higher = softer teacher distribution).

    Returns:
        Scalar KL divergence in nats.
    """
    p = _softmax(teacher_logits, temperature)
    q = _softmax(student_logits, temperature)
    eps = 1e-12
    total = 0.0
    for pi, qi in zip(p, q, strict=True):
        total += pi * math.log((pi + eps) / (qi + eps))
    return total


def estimate_weight_memory_bytes(
    num_parameters: int,
    precision_bits: int,
) -> int:
    """
    Estimate bytes to store weights only (no optimizer state, no KV cache).

    Args:
        num_parameters: Total trainable parameter count.
        precision_bits: Bits per weight (e.g. 32, 16, 8, 4).

    Returns:
        Approximate byte count.
    """
    if precision_bits <= 0:
        raise ValueError("precision_bits must be positive")
    return (num_parameters * precision_bits + 7) // 8


def bytes_to_gib(x: int) -> float:
    """Convert bytes to GiB for display."""
    return x / (1024.0**3)


def speculative_decode_micro(
    *,
    vocab_size: int,
    target_sequence: list[int],
    draft_error_rate: float,
    rng: random.Random,
    draft_tokens_per_step: int = 3,
) -> dict[str, Any]:
    """
    Simulate speculative decoding: draft proposes ``draft_tokens_per_step`` tokens;
    oracle (target) sequence defines ground truth; accept matching prefix each step.

    Args:
        vocab_size: Token id range is ``0 .. vocab_size - 1``.
        target_sequence: Oracle token ids to generate.
        draft_error_rate: Probability a draft token differs from oracle at a position.
        rng: Python random generator.
        draft_tokens_per_step: How many draft tokens proposed before verification.

    Returns:
        Metrics including acceptance counts and simulated verify steps.
    """
    if not 0.0 <= draft_error_rate <= 1.0:
        raise ValueError("draft_error_rate must be in [0, 1]")

    pos = 0
    total_draft = 0
    total_accepted = 0
    verify_steps = 0
    n = len(target_sequence)

    while pos < n:
        draft_len = min(draft_tokens_per_step, n - pos)
        draft: list[int] = []
        for i in range(draft_len):
            true_t = target_sequence[pos + i]
            if rng.random() < draft_error_rate:
                wrong = rng.randrange(vocab_size)
                draft.append(wrong if wrong != true_t else (wrong + 1) % vocab_size)
            else:
                draft.append(true_t)
            total_draft += 1

        accepted = 0
        for i, tok in enumerate(draft):
            if tok == target_sequence[pos + i]:
                accepted += 1
            else:
                break
        total_accepted += accepted
        pos += max(accepted, 1)
        verify_steps += 1

    return {
        "target_length": n,
        "draft_tokens_proposed": total_draft,
        "draft_tokens_accepted": total_accepted,
        "draft_acceptance_rate": total_accepted / total_draft if total_draft else 0.0,
        "verify_steps": verify_steps,
    }


def bitsandbytes_config_snippet() -> str:
    """
    Return a short Python snippet for 4-bit NF4 loading (documentation only).

    Returns:
        Source code string for README-style copy-paste.
    """
    return (
        "from transformers import AutoModelForCausalLM, BitsAndBytesConfig\n"
        "import torch\n"
        "\n"
        "quantization_config = BitsAndBytesConfig(\n"
        "    load_in_4bit=True,\n"
        "    bnb_4bit_compute_dtype=torch.float16,\n"
        "    bnb_4bit_quant_type='nf4',\n"
        "    bnb_4bit_use_double_quant=True,\n"
        ")\n"
        "model = AutoModelForCausalLM.from_pretrained(\n"
        "    'your-model-id',\n"
        "    quantization_config=quantization_config,\n"
        "    device_map='auto',\n"
        ")\n"
    )


def main() -> None:
    """Run the three demos and print a short summary."""
    print("Pattern 24: Small language model (SLM)\n")

    rng = random.Random(0)
    vocab = 8
    teacher = [rng.gauss(0, 1) * 2.0 for _ in range(vocab)]
    student = [rng.gauss(0, 1) * 0.5 for _ in range(vocab)]
    kl = kl_divergence(teacher, student, temperature=1.0)
    print("1) Distillation objective (KL at one position)")
    print("   KL(teacher || student) =", round(kl, 4), "nats")
    print("   After training, student logits are tuned to reduce KL + task loss.\n")

    params_b = 70_000_000_000
    fp32 = estimate_weight_memory_bytes(params_b, 32)
    fp16 = estimate_weight_memory_bytes(params_b, 16)
    q4 = estimate_weight_memory_bytes(params_b, 4)
    print("2) Weight memory (weights only; 70B parameters, illustrative)")
    print("   FP32:", round(bytes_to_gib(fp32), 1), "GiB")
    print("   FP16:", round(bytes_to_gib(fp16), 1), "GiB")
    print("   4-bit (packed):", round(bytes_to_gib(q4), 1), "GiB")
    print("   Real deployments add KV cache, activations, and framework overhead.\n")

    target = [rng.randrange(256) for _ in range(40)]
    sim = speculative_decode_micro(
        vocab_size=256,
        target_sequence=target,
        draft_error_rate=0.15,
        rng=rng,
        draft_tokens_per_step=3,
    )
    print("3) Speculative decoding (toy simulation)")
    print("   draft_tokens_proposed:", sim["draft_tokens_proposed"])
    print("   draft_tokens_accepted:", sim["draft_tokens_accepted"])
    print("   draft_acceptance_rate:", round(sim["draft_acceptance_rate"], 3))
    print("   verify_steps:", sim["verify_steps"])
    print("\n   Production: vLLM / other engines batch-verify draft tokens against the target model.\n")

    print("4) BitsAndBytesConfig (copy-paste for PyTorch + transformers)")
    print(bitsandbytes_config_snippet())


if __name__ == "__main__":
    main()
