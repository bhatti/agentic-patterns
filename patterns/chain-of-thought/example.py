#!/usr/bin/env python3
"""
Chain of Thought (CoT) Pattern - Refund Eligibility Advisor

This example implements CoT prompting for multi-step policy reasoning.
Given refund policy rules and a customer case, the system determines
eligibility using step-by-step reasoning (auditable and interpretable).

Real-World Problem:
-------------------
Customer support must decide refund eligibility from policy rules and case
facts. Zero-shot often returns a bare "Yes/No" with no justification, or
misinterprets rules (e.g., "within 30 days" from which date?). CoT produces
an explicit reasoning trace for compliance and customer trust.

Variants Demonstrated:
- Zero-shot CoT: "Think step by step"
- Few-shot CoT: Example (question → steps → answer)
- Auto CoT: Auto-generate few-shot examples via zero-shot CoT, then query

Usage:
    python example.py
"""

import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable

# Add project root for shared utilities if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class RefundCase:
    """Customer case facts for refund eligibility."""
    purchase_date: str  # YYYY-MM-DD
    return_request_date: str
    item_opened: bool
    has_receipt: bool
    reason: str  # e.g. "changed_mind", "defective", "wrong_item"
    days_since_purchase: Optional[int] = None

    def to_description(self) -> str:
        """Turn case into a short natural language description."""
        opened = "opened" if self.item_opened else "unopened"
        out = (
            f"Purchased on {self.purchase_date}, return requested on {self.return_request_date}. "
            f"Item {opened}, receipt {'available' if self.has_receipt else 'not available'}. "
            f"Reason: {self.reason}."
        )
        if self.days_since_purchase is not None:
            out += f" Days since purchase: {self.days_since_purchase}."
        return out


@dataclass
class CoTResult:
    """Result of a Chain of Thought query."""
    question: str
    reasoning: str
    conclusion: str
    variant: str  # "zero_shot", "few_shot", "auto_cot"


# =============================================================================
# POLICY RULES (for prompt context)
# =============================================================================

REFUND_POLICY = """
Refund policy:
- Full refund if: return requested within 30 days of purchase date AND item unopened AND receipt available.
- Partial refund (restocking fee): within 30 days, item opened, receipt available.
- No refund: no receipt, OR return requested more than 30 days after purchase date.
- Defective or wrong item: full refund within 90 days with receipt, regardless of opened/unopened.
"""


# =============================================================================
# LLM INTERFACE (simulated for demo; replace with Ollama/OpenAI in production)
# =============================================================================

def _extract_case_text(prompt: str) -> str:
    """Extract the case/customer description from prompt (last Case: or New question: block)."""
    for sep in ("New question:", "Case:", "Case :"):
        if sep in prompt:
            parts = prompt.split(sep)
            if parts:
                return parts[-1].strip().lower()
    return prompt.lower()


def simulate_llm(prompt: str, _model: Optional[str] = None) -> str:
    """
    Simulated LLM response for demo. In production, replace with Ollama/OpenAI/etc.
    Uses simple rule-based logic to mimic step-by-step reasoning for refund cases.
    """
    # Parse key facts from the *current* case only (ignore few-shot example text)
    case_text = _extract_case_text(prompt)
    prompt_lower = case_text
    # Infer 30/90 day window from "days since purchase" or explicit phrasing
    has_30 = "30 days" in prompt_lower or "within 30" in prompt_lower or "days since purchase: 1" in prompt_lower or "days since purchase: 2" in prompt_lower
    if "days since purchase:" in prompt_lower:
        m = re.search(r"days since purchase:\s*(\d+)", prompt_lower)
        if m:
            d = int(m.group(1))
            has_30 = d <= 30
            has_90 = d <= 90
        else:
            has_90 = "90 days" in prompt_lower or "within 90" in prompt_lower
    else:
        has_90 = "90 days" in prompt_lower or "within 90" in prompt_lower or "45 days" in prompt_lower or "50 days" in prompt_lower
    opened = "item opened" in prompt_lower or ("opened" in prompt_lower and "unopened" not in prompt_lower)
    unopened = "item unopened" in prompt_lower or ("unopened" in prompt_lower)
    receipt = "receipt available" in prompt_lower or ("receipt" in prompt_lower and "not available" not in prompt_lower)
    no_receipt = "receipt" in prompt_lower and "not available" in prompt_lower
    defective = "reason: defective" in prompt_lower or "reason: wrong" in prompt_lower or "reason: wrong item" in prompt_lower

    # Simulated step-by-step reasoning
    steps = []
    steps.append("Step 1: Identify key facts from the case (purchase date, return date, item condition, receipt, reason).")
    steps.append("Step 2: Check time window: compare return request date to purchase date.")
    if defective:
        steps.append("Step 3: Reason is defective or wrong item; check 90-day rule and receipt.")
    else:
        steps.append("Step 3: Reason is not defective/wrong item; apply standard 30-day rule.")
    steps.append("Step 4: Apply policy: full refund vs partial vs no refund.")
    steps.append("Step 5: State conclusion.")

    if no_receipt:
        conclusion = "Conclusion: No refund (receipt required)."
    elif defective and receipt and has_90:
        conclusion = "Conclusion: Full refund (defective/wrong item within 90 days with receipt)."
    elif defective and (not receipt or not has_90):
        conclusion = "Conclusion: No refund (defective/wrong item requires receipt and within 90 days)."
    elif unopened and receipt and has_30:
        conclusion = "Conclusion: Full refund (within 30 days, unopened, receipt available)."
    elif opened and receipt and has_30:
        conclusion = "Conclusion: Partial refund with restocking fee (within 30 days, opened, receipt available)."
    elif not has_30 and not (defective and has_90):
        conclusion = "Conclusion: No refund (outside 30-day window and not defective within 90 days)."
    else:
        conclusion = "Conclusion: Eligibility depends on exact dates; verify 30/90-day window and receipt."

    reasoning = "\n".join(steps) + "\n\n" + conclusion
    return reasoning


def invoke_llm(prompt: str, model: Optional[str] = None) -> str:
    """
    Invoke LLM. Uses shared Ollama client if available, else simulated response.
    """
    try:
        from shared.ollama_client import get_ollama_client
        client = get_ollama_client()
        if client and model:
            response = client.generate(prompt, model=model)
            if response:
                return response
    except Exception:
        pass
    return simulate_llm(prompt, model)


# =============================================================================
# VARIANT 1: ZERO-SHOT COT
# =============================================================================

ZERO_SHOT_COT_SUFFIX = "\n\nThink step by step. Show your reasoning and then state the final eligibility conclusion."


def zero_shot_cot(
    policy: str,
    case_description: str,
    question: str,
    llm: Optional[Callable[[str], str]] = None,
) -> CoTResult:
    """
    Zero-shot CoT: append "Think step by step" to the prompt.
    No examples; model generates reasoning structure on its own.
    """
    llm = llm or invoke_llm
    prompt = f"{policy}\n\nCase: {case_description}\n\nQuestion: {question}{ZERO_SHOT_COT_SUFFIX}"
    full_response = llm(prompt)

    # Parse conclusion (last sentence or line containing "Conclusion" or "refund")
    lines = full_response.strip().split("\n")
    conclusion = ""
    for line in reversed(lines):
        line = line.strip()
        if line and (line.lower().startswith("conclusion") or "refund" in line.lower() or "eligible" in line.lower()):
            conclusion = line
            break
    if not conclusion and lines:
        conclusion = lines[-1].strip()

    reasoning = full_response
    if conclusion and conclusion in full_response:
        reasoning = full_response.replace(conclusion, "").strip()

    return CoTResult(
        question=question,
        reasoning=reasoning,
        conclusion=conclusion or full_response[-200:] if len(full_response) > 200 else full_response,
        variant="zero_shot",
    )


# =============================================================================
# VARIANT 2: FEW-SHOT COT
# =============================================================================

FEW_SHOT_EXAMPLES = """
Example 1:
Q: Customer purchased 10 days ago, item unopened, has receipt, reason: changed mind. Eligible for full refund?
A:
Step 1: Within 30 days? Yes (10 days).
Step 2: Item unopened? Yes.
Step 3: Receipt available? Yes.
Step 4: Policy says: full refund if within 30 days AND unopened AND receipt. All conditions met.
Conclusion: Yes, eligible for full refund.

Example 2:
Q: Purchased 45 days ago, item opened, has receipt, reason: defective. Eligible for refund?
A:
Step 1: Reason is defective; 90-day rule applies.
Step 2: Within 90 days? Yes (45 days).
Step 3: Receipt available? Yes.
Step 4: Policy says: defective/wrong item — full refund within 90 days with receipt.
Conclusion: Yes, eligible for full refund (defective within 90 days with receipt).
"""


def few_shot_cot(
    policy: str,
    case_description: str,
    question: str,
    llm: Optional[Callable[[str], str]] = None,
) -> CoTResult:
    """
    Few-shot CoT: provide example (question → step-by-step reasoning → conclusion).
    Model follows the same format. "Shows how to fish" vs RAG "gives fish."
    """
    llm = llm or invoke_llm
    prompt = (
        f"{policy}\n\n"
        f"Use the following examples to reason step by step, then answer the new question in the same format.\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"New question:\nQ: {question}\n\nCase: {case_description}\n\nA:"
    )
    full_response = llm(prompt)

    lines = full_response.strip().split("\n")
    conclusion = ""
    for line in reversed(lines):
        line = line.strip()
        if line and (line.lower().startswith("conclusion") or "refund" in line.lower() or "eligible" in line.lower()):
            conclusion = line
            break
    if not conclusion and lines:
        conclusion = lines[-1].strip()

    reasoning = full_response
    if conclusion and conclusion in full_response:
        reasoning = full_response.replace(conclusion, "").strip()

    return CoTResult(
        question=question,
        reasoning=reasoning,
        conclusion=conclusion or full_response[-200:] if len(full_response) > 200 else full_response,
        variant="few_shot",
    )


# =============================================================================
# VARIANT 3: AUTO COT
# =============================================================================

# Pool of sample questions for Auto CoT (diverse eligibility scenarios)
AUTO_COT_QUESTION_POOL = [
    "Customer purchased 5 days ago, item unopened, has receipt, reason: changed mind. Eligible for full refund?",
    "Purchased 20 days ago, item opened, has receipt, reason: changed mind. Eligible for any refund?",
    "Purchased 50 days ago, item unopened, has receipt, reason: defective. Eligible for refund?",
    "Purchased 15 days ago, item unopened, no receipt. Eligible for refund?",
]


def auto_cot(
    policy: str,
    case_description: str,
    question: str,
    num_demos: int = 2,
    llm: Optional[Callable[[str], str]] = None,
) -> CoTResult:
    """
    Auto CoT: build few-shot CoT automatically.
    1. Sample questions from pool (or cluster); here we take first num_demos.
    2. For each, generate reasoning using zero-shot CoT.
    3. Build few-shot prompt with these (Q, reasoning) pairs.
    4. Ask the actual question with that prompt.
    """
    llm = llm or invoke_llm
    demos = []
    for i, sample_q in enumerate(AUTO_COT_QUESTION_POOL[:num_demos]):
        prompt = f"{policy}\n\nCase: {sample_q}\n\nQuestion: {sample_q}{ZERO_SHOT_COT_SUFFIX}"
        response = llm(prompt)
        demos.append(f"Q: {sample_q}\nA:\n{response}\n")

    few_shot_block = "\n".join(demos)
    prompt = (
        f"{policy}\n\n"
        f"Use the following example reasonings to guide your format. Then answer the new question step by step.\n\n"
        f"{few_shot_block}\n\n"
        f"New question:\nQ: {question}\n\nCase: {case_description}\n\nA:"
    )
    full_response = llm(prompt)

    lines = full_response.strip().split("\n")
    conclusion = ""
    for line in reversed(lines):
        line = line.strip()
        if line and (line.lower().startswith("conclusion") or "refund" in line.lower() or "eligible" in line.lower()):
            conclusion = line
            break
    if not conclusion and lines:
        conclusion = lines[-1].strip()

    reasoning = full_response
    if conclusion and conclusion in full_response:
        reasoning = full_response.replace(conclusion, "").strip()

    return CoTResult(
        question=question,
        reasoning=reasoning,
        conclusion=conclusion or full_response[-200:] if len(full_response) > 200 else full_response,
        variant="auto_cot",
    )


# =============================================================================
# REFUND ELIGIBILITY ADVISOR
# =============================================================================

class RefundEligibilityAdvisor:
    """
    Refund eligibility advisor using Chain of Thought.
    Supports zero-shot CoT, few-shot CoT, and Auto CoT.
    """

    def __init__(self, policy: str = REFUND_POLICY, llm: Optional[Callable[[str], str]] = None):
        self.policy = policy
        self.llm = llm

    def check_eligibility(
        self,
        case: RefundCase,
        variant: str = "zero_shot",
        question: Optional[str] = None,
    ) -> CoTResult:
        """
        Determine refund eligibility with step-by-step reasoning.

        Args:
            case: Customer case facts.
            variant: "zero_shot", "few_shot", or "auto_cot".
            question: Override question (default: "Is this customer eligible for a refund, and if so what type?").

        Returns:
            CoTResult with reasoning and conclusion.
        """
        desc = case.to_description()
        q = question or "Is this customer eligible for a refund, and if so what type (full, partial, or none)?"
        if variant == "zero_shot":
            return zero_shot_cot(self.policy, desc, q, self.llm)
        if variant == "few_shot":
            return few_shot_cot(self.policy, desc, q, self.llm)
        if variant == "auto_cot":
            return auto_cot(self.policy, desc, q, num_demos=2, llm=self.llm)
        raise ValueError("variant must be one of: zero_shot, few_shot, auto_cot")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("CHAIN OF THOUGHT PATTERN - REFUND ELIGIBILITY ADVISOR")
    print("=" * 70)
    print()
    print("Demonstrates step-by-step reasoning for policy eligibility.")
    print("Variants: Zero-shot CoT, Few-shot CoT, Auto CoT.")
    print()

    advisor = RefundEligibilityAdvisor()

    # Case 1: Clear full refund (within 30 days, unopened, receipt)
    case1 = RefundCase(
        purchase_date="2024-01-01",
        return_request_date="2024-01-20",
        item_opened=False,
        has_receipt=True,
        reason="changed_mind",
        days_since_purchase=19,
    )
    print("-" * 70)
    print("Case 1: " + case1.to_description())
    print("-" * 70)
    for v in ["zero_shot", "few_shot", "auto_cot"]:
        result = advisor.check_eligibility(case1, variant=v)
        print(f"\n[{v}]")
        print(result.reasoning[:600] + ("..." if len(result.reasoning) > 600 else ""))
        print("\n→ " + result.conclusion)
    print()

    # Case 2: Defective within 90 days
    case2 = RefundCase(
        purchase_date="2024-01-01",
        return_request_date="2024-02-15",
        item_opened=True,
        has_receipt=True,
        reason="defective",
        days_since_purchase=45,
    )
    print("-" * 70)
    print("Case 2: " + case2.to_description())
    print("-" * 70)
    result2 = advisor.check_eligibility(case2, variant="few_shot")
    print(result2.reasoning[:500] + ("..." if len(result2.reasoning) > 500 else ""))
    print("\n→ " + result2.conclusion)
    print()

    # Case 3: No receipt
    case3 = RefundCase(
        purchase_date="2024-01-10",
        return_request_date="2024-01-25",
        item_opened=False,
        has_receipt=False,
        reason="changed_mind",
        days_since_purchase=15,
    )
    print("-" * 70)
    print("Case 3: " + case3.to_description())
    print("-" * 70)
    result3 = advisor.check_eligibility(case3, variant="zero_shot")
    print(result3.reasoning[:500] + ("..." if len(result3.reasoning) > 500 else ""))
    print("\n→ " + result3.conclusion)
    print()
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
