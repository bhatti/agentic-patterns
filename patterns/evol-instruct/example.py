#!/usr/bin/env python3
"""
Evol-Instruct Pattern - Internal Policy Q&A Dataset Builder

This example demonstrates the Evol-Instruct pipeline for building an
instruction-tuning dataset: evolve simple instructions into harder ones,
generate answers, evaluate and filter, then output a dataset ready for
SFT (instruction tuning) on an open-weight model.

Real-World Problem:
-------------------
Enterprises need a model that answers complex questions from internal
policy documents (vacation, expenses, remote work, etc.) under data
privacy. Manually writing thousands of hard (question, answer) pairs
is costly. Evol-Instruct: start from simple seed questions, evolve
them into deeper/concrete/multi-step instructions, generate answers,
score and filter, then use the result for SFT (e.g., LoRA on Llama).

Pipeline (4 steps):
1. Evolve instructions — make seeds harder (deeper, concrete, multi-step).
2. Generate answers — produce an answer for each instruction.
3. Evaluate and filter — score each (instruction, answer), keep high scores.
4. Instruction tuning — in production: SFT/LoRA on HuggingFace; here we
   output the training-ready dataset.

This script uses rule-based evolution and heuristic scoring so it runs
without any LLM or GPU. In production, replace evolve/generate/score
with an LLM and run SFT with HuggingFace + PEFT/TRL.

Usage:
    python example.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Project root for shared utilities if needed
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class InstructionAnswer:
    """Single (instruction, answer) pair with optional score."""
    instruction: str
    answer: str
    score: int = 0  # 1-5
    explanation: str = ""


# =============================================================================
# STEP 1: SEED INSTRUCTIONS (domain: internal policy)
# =============================================================================

def get_seed_instructions() -> List[str]:
    """
    Simple policy questions that will be evolved into harder instructions.
    In production, these can come from FAQs, templates, or an LLM over policy docs.
    """
    return [
        "What is the vacation policy?",
        "How do I submit an expense report?",
        "What is the remote work policy?",
        "When am I eligible for parental leave?",
        "How does the performance review cycle work?",
    ]


# =============================================================================
# STEP 2: EVOLVE INSTRUCTIONS (rule-based stand-in for LLM evolution)
# =============================================================================

def evolve_instructions(seeds: List[str]) -> List[str]:
    """
    Evolve seed instructions into harder variants: deeper, more concrete,
    and multi-step. In production, use an LLM with prompts like:
    - Deeper: add constraints/hypotheticals (e.g. "If you are in region X...")
    - Concrete: ask for "List 3 reasons...", "What are the steps..."
    - Multi-step: combine two questions into one.
    """
    evolved: List[str] = []

    for q in seeds:
        q_lower = q.lower()

        # Deeper: add hypothetical/constraint
        if "vacation" in q_lower:
            evolved.append(
                "If you have been with the company less than one year and need to take vacation during a blackout period, what are your options according to the vacation policy?"
            )
        if "expense" in q_lower:
            evolved.append(
                "You are traveling internationally and have a mix of personal and business expenses. How do you submit an expense report and what documentation is required for reimbursement?"
            )
        if "remote" in q_lower:
            evolved.append(
                "If you want to work remotely from a different country for three months, what does the remote work policy require in terms of approval and tax implications?"
            )
        if "parental" in q_lower:
            evolved.append(
                "How does parental leave interact with short-term disability and vacation accrual? When are you eligible and what is the process to apply?"
            )
        if "performance" in q_lower:
            evolved.append(
                "If your performance review is delayed because your manager left the company, what is the process and how does it affect promotion eligibility?"
            )

        # More concrete: ask for 3 reasons or steps
        if "vacation" in q_lower:
            evolved.append("List three key conditions under the vacation policy that affect how many days you can carry over to the next year.")
        if "expense" in q_lower:
            evolved.append("What are the steps to submit an expense report, and what is the typical turnaround time for reimbursement?")
        if "remote" in q_lower:
            evolved.append("Give three reasons the remote work policy might require you to work from an office on certain days.")

        # Multi-step: combine two themes (simplified — in production LLM combines two questions)
        if "vacation" in q_lower and len(evolved) > 0:
            evolved.append(
                "How do the vacation policy and the remote work policy interact if you want to take vacation while working from a different time zone?"
            )

    # Dedupe and include seeds
    seen: set[str] = set()
    out: List[str] = []
    for s in seeds + evolved:
        if s.strip() and s.strip() not in seen:
            seen.add(s.strip())
            out.append(s.strip())
    return out


# =============================================================================
# STEP 3: GENERATE ANSWERS (simulated; in production use LLM + context)
# =============================================================================

def generate_answers(instructions: List[str]) -> List[InstructionAnswer]:
    """
    For each instruction, produce an answer. In production, call an LLM
    with your policy documents as context and optionally enforce format.
    """
    # Simulated policy answers (stand-in for LLM output grounded in docs)
    policy_answers: List[tuple[str, str]] = [
        (
            "What is the vacation policy?",
            "Full-time employees accrue 15 days per year. Vacation must be requested in the HR portal at least two weeks in advance. Unused days can be carried over up to 5 days to the next year.",
        ),
        (
            "How do I submit an expense report?",
            "Submit via the Expense tool in the HR portal. Attach receipts for expenses over $25. Reports require manager approval. Reimbursement typically takes 5–7 business days.",
        ),
        (
            "What is the remote work policy?",
            "Eligible roles may work remotely up to 3 days per week. You must be in the same country as your team for tax and compliance. Approval is required for more than 2 weeks abroad.",
        ),
        (
            "When am I eligible for parental leave?",
            "After 6 months of employment. Primary caregivers get 16 weeks; secondary get 4 weeks. Short-term disability may run concurrently. Coordinate with HR and your manager to start leave.",
        ),
        (
            "How does the performance review cycle work?",
            "Annual cycle runs January–December. Self-review is due in February; manager reviews in March. Calibration happens in April. Outcomes affect merit increases and promotion eligibility.",
        ),
    ]

    # Map seed-like instructions to answers; for evolved ones use a generic template
    answer_by_instruction: dict[str, str] = {q: a for q, a in policy_answers}

    results: List[InstructionAnswer] = []
    for inst in instructions:
        if inst in answer_by_instruction:
            ans = answer_by_instruction[inst]
        else:
            # Evolved question: use a short generic answer (in production: LLM with context)
            ans = (
                "Policy specifics depend on your situation. Check the internal policy hub and, if needed, contact HR or your manager for your specific case."
            )
        results.append(InstructionAnswer(instruction=inst, answer=ans))
    return results


# =============================================================================
# STEP 4: EVALUATE AND FILTER (heuristic scorer; in production use LLM or model)
# =============================================================================

def score_instruction_answer(ia: InstructionAnswer) -> InstructionAnswer:
    """
    Score (instruction, answer) 1–5 with explanation. In production, use an
    LLM with a rubric (insight, correctness, clarity) or a trained scorer.
    """
    # Heuristic: longer, more structured answers and concrete questions score higher
    score = 3
    explanation_parts: List[str] = []

    if len(ia.answer) > 100:
        score += 1
        explanation_parts.append("Answer is detailed.")
    if "steps" in ia.instruction.lower() or "list" in ia.instruction.lower() or "three" in ia.instruction.lower():
        score += 1
        explanation_parts.append("Instruction requires concrete output.")
    if "?" in ia.instruction and len(ia.instruction) > 60:
        score = min(5, score + 1)
        explanation_parts.append("Instruction is complex.")

    if score < 1:
        score = 1
    if score > 5:
        score = 5
    explanation = " ".join(explanation_parts) if explanation_parts else "Baseline quality."

    return InstructionAnswer(
        instruction=ia.instruction,
        answer=ia.answer,
        score=score,
        explanation=explanation,
    )


def filter_by_score(examples: List[InstructionAnswer], min_score: int = 4) -> List[InstructionAnswer]:
    """Keep only examples with score >= min_score."""
    return [ex for ex in examples if ex.score >= min_score]


# =============================================================================
# OUTPUT: TRAINING-READY DATASET (chat format for SFT)
# =============================================================================

def to_sft_messages(examples: List[InstructionAnswer]) -> List[dict]:
    """Convert to chat format suitable for SFT (e.g., HuggingFace/TRL)."""
    return [
        {
            "messages": [
                {"role": "user", "content": ex.instruction},
                {"role": "assistant", "content": ex.answer},
            ],
        }
        for ex in examples
    ]


def main() -> None:
    # 1. Seed instructions
    seeds = get_seed_instructions()
    print(f"Seed instructions: {len(seeds)}")
    for s in seeds[:3]:
        print(f"  - {s[:60]}...")
    print()

    # 2. Evolve instructions
    all_instructions = evolve_instructions(seeds)
    print(f"After evolution: {len(all_instructions)} instructions")
    print()

    # 3. Generate answers
    qa_pairs = generate_answers(all_instructions)
    print(f"Generated (instruction, answer) pairs: {len(qa_pairs)}")
    print()

    # 4. Score and filter
    scored = [score_instruction_answer(ia) for ia in qa_pairs]
    filtered = filter_by_score(scored, min_score=4)
    print(f"After filter (score >= 4): {len(filtered)} examples")
    for ex in filtered[:3]:
        print(f"  [score={ex.score}] {ex.instruction[:50]}...")
    print()

    # 5. Training-ready dataset (chat format)
    sft_dataset = to_sft_messages(filtered)
    print(f"SFT-ready examples (messages format): {len(sft_dataset)}")
    if sft_dataset:
        print("Sample message:")
        print(json.dumps(sft_dataset[0], indent=2))
    print()

    # Optional: write JSONL for use with HuggingFace/TRL
    out_path = Path(__file__).parent / "evol_instruct_sft_dataset.jsonl"
    with open(out_path, "w") as f:
        for item in sft_dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(sft_dataset)} examples to {out_path.name}")
    print()
    print(
        "In production: run SFT (e.g., HuggingFace transformers + peft LoRA + trl SFTTrainer)\n"
        "on this dataset to instruction-tune an open-weight model (Llama, Gemma, etc.)."
    )


if __name__ == "__main__":
    main()
