#!/usr/bin/env python3
"""
LLM as Judge Pattern - Support Reply Quality Evaluation

This example uses an LLM (or a simulated judge) to evaluate customer support
replies against a scoring rubric: scores per criterion (1-5) plus brief
justifications. This scales better than human-only review and gives more
nuance than single automated metrics, forming a feedback loop for model
and prompt improvements.

Real-World Problem:
-------------------
Support teams need to evaluate reply quality at scale (helpfulness, tone,
accuracy, clarity, completeness). Human review does not scale; simple
metrics (length, keyword match) miss nuance. LLM as Judge: define criteria
in a prompt, call the judge with temperature=0, get scores and justifications
for logging, filtering, or training.

Three options (this example implements Option 1 - prompting):
- Option 1: Prompting — rubric in prompt, LLM returns scores + justification.
- Option 2: ML — train a classifier on historical (reply, human/LLM scores).
- Option 3: Fine-tune a dedicated judge model on labeled data.

Usage:
    python example.py
    # Uses simulated judge by default. To use Ollama: set OLLAMA_BASE_URL
    # and run with Ollama available; the example will try shared.ollama_client.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

# Project root for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class CriterionScore:
    """Score and justification for one criterion."""
    criterion: str
    score: int  # 1-5
    justification: str


@dataclass
class JudgeResult:
    """Full result of LLM-as-Judge evaluation."""
    item: str
    scores: List[CriterionScore] = field(default_factory=list)
    raw_response: str = ""


# =============================================================================
# SCORING RUBRIC (support reply quality)
# =============================================================================

SUPPORT_REPLY_CRITERIA = """
- Helpfulness: Addresses the customer's question or issue directly; provides actionable next steps or information they need.
- Tone: Professional, empathetic, and appropriate; no dismissive or robotic phrasing.
- Accuracy: Factually correct and consistent with policy/product; no misleading or outdated information.
- Clarity: Easy to read; short sentences, clear structure, no unnecessary jargon.
- Completeness: Covers the main ask; does not leave critical follow-up questions unanswered.
"""


def build_judge_prompt(item: str, criteria: str = SUPPORT_REPLY_CRITERIA) -> str:
    """
    Build the prompt for the LLM judge: rubric + item + instruction to
    return a score (1-5) and brief justification per criterion.
    """
    return f"""You are evaluating a customer support reply for quality.

Provide a score from 1 to 5 for each of the following criteria (1 = poor, 5 = excellent), and give a brief justification for each score.

Criteria:
{criteria}

Support reply to evaluate:
---
{item}
---

For each criterion, respond with exactly:
**<Criterion name>: <score>**
Justification: <one or two sentences>

Scores:"""


# =============================================================================
# SIMULATED JUDGE (runnable without LLM; production: use Ollama/API with temp=0)
# =============================================================================

def _simulated_judge(prompt: str) -> str:
    """
    Rule-based stand-in for the LLM judge. In production, replace with
    a call to an LLM (Ollama, OpenAI, etc.) with temperature=0.
    Produces output in the same format as the prompt requests.
    """
    # Extract the "Support reply to evaluate" section from the prompt
    match = re.search(r"Support reply to evaluate:\n---\n(.*?)\n---", prompt, re.DOTALL)
    reply = (match.group(1).strip() if match else "").lower()

    # Heuristic scores based on reply content
    length = len(reply)
    has_apology = "sorry" in reply or "apologize" in reply
    has_action = "you can" in reply or "please" in reply or "step" in reply
    has_question = "?" in reply
    short_sentences = reply.count(".") + reply.count("!")
    jargon = sum(1 for w in ["hereby", "pursuant", "utilize", "facilitate"] if w in reply)

    helpfulness = min(5, 2 + (1 if has_action else 0) + (1 if length > 80 else 0))
    tone = min(5, 3 + (1 if has_apology else 0) + (1 if "thank" in reply else 0))
    accuracy = 4  # Cannot verify facts in simulation
    clarity = min(5, max(1, 4 - jargon + (1 if short_sentences >= 2 else 0)))
    completeness = min(5, 2 + (1 if has_action else 0) + (1 if length > 100 else 0) + (0 if has_question and length < 80 else 1))

    lines = [
        "**Helpfulness: " + str(helpfulness) + "**",
        "Justification: Reply " + ("provides actionable guidance and addresses the issue." if has_action else "could be more actionable."),
        "",
        "**Tone: " + str(tone) + "**",
        "Justification: " + ("Professional and empathetic." if has_apology or "thank" in reply else "Neutral; could add empathy."),
        "",
        "**Accuracy: " + str(accuracy) + "**",
        "Justification: Content appears consistent with typical support guidance.",
        "",
        "**Clarity: " + str(clarity) + "**",
        "Justification: " + ("Clear and readable." if jargon == 0 else "Some jargon; could simplify."),
        "",
        "**Completeness: " + str(completeness) + "**",
        "Justification: " + ("Covers the main ask." if length > 100 else "Could add more detail."),
    ]
    return "\n".join(lines)


def _invoke_ollama(prompt: str, model: Optional[str] = None) -> Optional[str]:
    """Try to use Ollama for judge; return None if unavailable."""
    try:
        from shared.ollama_client import get_ollama_client
        client = get_ollama_client()
        if client:
            return client.generate(prompt, model=model or getattr(client, "default_model", "llama3"))
    except Exception:
        pass
    return None


def run_judge(prompt: str, judge_fn: Optional[Callable[[str], str]] = None) -> str:
    """
    Run the judge: use provided callable, or Ollama if available, or simulated judge.
    In production, use an LLM with temperature=0 for consistency.
    """
    if judge_fn is not None:
        return judge_fn(prompt)
    response = _invoke_ollama(prompt)
    if response:
        return response
    return _simulated_judge(prompt)


# =============================================================================
# PARSE JUDGE OUTPUT INTO STRUCTURED RESULT
# =============================================================================

def parse_judge_response(raw: str, item: str) -> JudgeResult:
    """
    Parse the judge's text response into criterion scores and justifications.
    Expects lines like "**Criterion: N**" and "Justification: ..."
    """
    scores: List[CriterionScore] = []
    pattern = re.compile(r"\*\*(.+?):\s*(\d)\*\*", re.IGNORECASE)
    just_pattern = re.compile(r"Justification:\s*(.+?)(?=\n\n|\n\*\*|\Z)", re.DOTALL | re.IGNORECASE)

    pos = 0
    for m in pattern.finditer(raw):
        criterion = m.group(1).strip()
        score_val = int(m.group(2))
        score_val = max(1, min(5, score_val))
        # Find next justification after this match
        just_match = just_pattern.search(raw, m.end())
        justification = (just_match.group(1).strip() if just_match else "").split("\n")[0]
        scores.append(CriterionScore(criterion=criterion, score=score_val, justification=justification))

    return JudgeResult(item=item, scores=scores, raw_response=raw)


# =============================================================================
# MAIN: EVALUATE SAMPLE SUPPORT REPLIES
# =============================================================================

def main() -> None:
    # Sample support replies to evaluate
    replies = [
        """Thank you for reaching out. I'm sorry to hear you're having trouble with your order.

You can check the status of your shipment by visiting the Track Your Order page in your account. If the delivery date has passed and you still don't have the package, please reply to this email with your order number and we'll open an investigation.

We appreciate your patience.""",
        """Hi, your order might be late. Check the website. Thanks.""",
    ]

    print("LLM as Judge — Support Reply Quality")
    print("Criteria: Helpfulness, Tone, Accuracy, Clarity, Completeness (1-5 each)")
    print()

    for i, reply in enumerate(replies, 1):
        print(f"--- Reply {i} ---")
        print(reply[:200] + "..." if len(reply) > 200 else reply)
        print()

        prompt = build_judge_prompt(reply)
        raw = run_judge(prompt)
        result = parse_judge_response(raw, reply)

        print("Scores:")
        for s in result.scores:
            print(f"  {s.criterion}: {s.score} — {s.justification[:80]}...")
        if result.scores:
            avg = sum(s.score for s in result.scores) / len(result.scores)
            print(f"  Average: {avg:.1f}")
        print()

    print("In production: use an LLM (Ollama/API) with temperature=0 for the judge.")
    print("Options: (1) Prompting (this example), (2) Train classifier on historical scores, (3) Fine-tune judge model.")


if __name__ == "__main__":
    main()
