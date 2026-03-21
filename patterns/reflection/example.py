#!/usr/bin/env python3
"""
Reflection — apology email demo + multi-round iteration + optional LangChain LCEL

Demonstrates Reflection for stateless APIs (Lakshmanan Pattern 18; *Gulli* reflector):
(1) generate, (2) evaluate/critique, (3) modified prompt, (4) revise — optionally
repeat (generator–critic iteration). Optional ``RunnableLambda`` chain for LangChain.

Real-world hook: apology emails for delayed shipments (draft → rules/judge → revise).

Usage:
    python example.py
    # Simulated generator/evaluator; swap in LLM + LLM-as-Judge (Pattern 17).
    # LCEL: pip install langchain-core (repo root requirements.txt).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# Project root for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from langchain_core.runnables import RunnableLambda
except ImportError:  # pragma: no cover - optional when langchain-core not installed
    RunnableLambda = None  # type: ignore[misc,assignment]


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ReflectionResult:
    """Result of the Reflection pipeline: initial, feedback, revised."""
    initial_response: str
    feedback: str
    revised_response: str
    evaluator_notes: str = ""


# =============================================================================
# STEP 1: GENERATE INITIAL RESPONSE (simulated; production: LLM)
# =============================================================================

def generate_initial(user_prompt: str, generate_fn: Optional[Callable[[str], str]] = None) -> str:
    """
    First call: user prompt -> initial response. Do not return this to the
    client yet; send it to the evaluator.
    """
    if generate_fn is not None:
        return generate_fn(user_prompt)

    # Simulated draft: intentionally incomplete so revision is visible
    if "apology" in user_prompt.lower() and "delay" in user_prompt.lower():
        return """Hi,

We're sorry for the delay with your order. We know this is frustrating.

We're working to get your package to you as soon as we can. If you have any questions, contact us.

Thanks,
Support"""
    return "Generated response for: " + user_prompt[:80]


# =============================================================================
# STEP 2: EVALUATE (simulated; production: LLM-as-Judge or human)
# =============================================================================

def evaluate(
    user_prompt: str,
    initial_response: str,
    evaluate_fn: Optional[Callable[[str, str], tuple[str, str]]] = None,
) -> tuple[str, str]:
    """
    Evaluator: given original request and initial response, return (feedback,
    notes). Feedback is used in the modified prompt; notes are for logging.
    In production, use LLM-as-Judge (Pattern 17) or human review.
    """
    if evaluate_fn is not None:
        return evaluate_fn(user_prompt, initial_response)

    feedback_parts: list[str] = []
    notes_parts: list[str] = []

    initial_lower = initial_response.lower()
    # Check for order reference placeholder or mention
    if "order" in initial_lower and ("#" not in initial_response and "number" not in initial_lower and "reference" not in initial_lower):
        feedback_parts.append("Include a placeholder for the order number or reference (e.g., 'Regarding order #[ORDER_ID]') so the customer can identify the shipment.")
        notes_parts.append("Missing order ref placeholder")
    elif "order" not in initial_lower:
        feedback_parts.append("Mention the order or shipment and add a placeholder for order/reference number.")
        notes_parts.append("Missing order mention")

    # Check for clear next step / action
    if "contact" in initial_lower or "reply" in initial_lower or "reach" in initial_lower:
        pass  # has some action
    else:
        feedback_parts.append("Add one clear next step (e.g., how to track the shipment or who to contact).")
        notes_parts.append("Weak call to action")

    # Tone: apology strength
    if "sorry" in initial_lower or "apolog" in initial_lower:
        notes_parts.append("Apology present")
    else:
        feedback_parts.append("Include a brief, sincere apology for the delay.")
        notes_parts.append("Apology missing")

    # Length / clarity
    if len(initial_response.strip()) < 150:
        feedback_parts.append("Keep the email short but add one concrete detail (e.g., expected timeline or tracking link placeholder).")
        notes_parts.append("Could be more concrete")

    feedback = " ".join(feedback_parts) if feedback_parts else "The draft is good; make only minor polish if needed (e.g., ensure order reference and one clear next step)."
    notes = "; ".join(notes_parts) if notes_parts else "No issues"

    return feedback, notes


# =============================================================================
# STEP 3: BUILD MODIFIED PROMPT (original + initial + feedback)
# =============================================================================

def build_modified_prompt(user_prompt: str, initial_response: str, feedback: str) -> str:
    """
    Modified prompt so the model "sees" its first attempt and the feedback.
    This is what makes Reflection work in a stateless API: context is in the prompt.
    """
    return f"""Original request:
{user_prompt}

Your previous response:
---
{initial_response}
---

Feedback to apply:
{feedback}

Produce an improved version that addresses the feedback. Keep the same tone and length roughly similar unless the feedback asks for more or less detail."""


# =============================================================================
# STEP 4: GENERATE REVISED RESPONSE (second call)
# =============================================================================

def generate_revised(modified_prompt: str, generate_fn: Optional[Callable[[str], str]] = None) -> str:
    """Second call: modified prompt -> revised response. Return this to the client."""
    if generate_fn is not None:
        return generate_fn(modified_prompt)

    # Simulated revision: add order ref placeholder and clearer next step
    return """Hi,

We're sorry for the delay with your shipment. We know this is frustrating.

Regarding order #[ORDER_ID]: we're working to get your package to you as soon as possible. You can track its status here: [TRACKING_LINK]. If you have any questions, reply to this email or call us at [PHONE].

Thank you for your patience.

Support"""


# =============================================================================
# REFLECTION PIPELINE (orchestrate two calls + evaluator)
# =============================================================================

def run_reflection(
    user_prompt: str,
    generate_fn: Optional[Callable[[str], str]] = None,
    evaluate_fn: Optional[Callable[[str, str], tuple[str, str]]] = None,
) -> ReflectionResult:
    """
    Full Reflection loop: generate -> evaluate -> modified prompt -> revise.
    In production, pass in generate_fn (LLM) and optionally evaluate_fn (e.g., LLM-as-Judge).
    """
    # First call: initial response (not returned to client yet)
    initial_response = generate_initial(user_prompt, generate_fn)

    # Evaluate initial response (LLM or human)
    feedback, notes = evaluate(user_prompt, initial_response, evaluate_fn)

    # Build modified prompt and second call
    modified_prompt = build_modified_prompt(user_prompt, initial_response, feedback)
    revised_response = generate_revised(modified_prompt, generate_fn)

    return ReflectionResult(
        initial_response=initial_response,
        feedback=feedback,
        revised_response=revised_response,
        evaluator_notes=notes,
    )


def run_reflection_multi_round(
    user_prompt: str,
    revision_rounds: int = 1,
    generate_fn: Optional[Callable[[str], str]] = None,
    evaluate_fn: Optional[Callable[[str, str], tuple[str, str]]] = None,
) -> ReflectionResult:
    """
    Generator–critic loop: after the initial generation, repeat evaluate → modified
    prompt → revise for ``revision_rounds`` times (Gulli-style iteration).

    Args:
        user_prompt: Original user request.
        revision_rounds: Number of evaluate+revise cycles after the initial draft (minimum ``1``).
        generate_fn: Optional LLM callable for both initial and revised generations.
        evaluate_fn: Optional evaluator; defaults to rule-based ``evaluate``.

    Returns:
        ``ReflectionResult`` with ``initial_response`` the first draft and
        ``revised_response`` the last draft; ``feedback``/``evaluator_notes`` from
        the final evaluation pass.
    """
    rounds = max(1, revision_rounds)
    current = generate_initial(user_prompt, generate_fn)
    initial_snapshot = current
    last_feedback = ""
    last_notes = ""
    for _ in range(rounds):
        last_feedback, last_notes = evaluate(user_prompt, current, evaluate_fn)
        modified = build_modified_prompt(user_prompt, current, last_feedback)
        current = generate_revised(modified, generate_fn)
    return ReflectionResult(
        initial_response=initial_snapshot,
        feedback=last_feedback,
        revised_response=current,
        evaluator_notes=last_notes,
    )


def _lcel_state_after_initial(state: dict[str, Any]) -> dict[str, Any]:
    """LCEL step: populate ``current`` from ``user_prompt``."""
    user_prompt = str(state["user_prompt"])
    gen_fn = state.get("generate_fn")
    fn = gen_fn if callable(gen_fn) else None
    return {**state, "current": generate_initial(user_prompt, fn)}


def _lcel_state_after_evaluate(state: dict[str, Any]) -> dict[str, Any]:
    """LCEL step: run evaluator; attach ``feedback`` and ``evaluator_notes``."""
    user_prompt = str(state["user_prompt"])
    current = str(state["current"])
    ev_fn = state.get("evaluate_fn")
    efn = ev_fn if callable(ev_fn) else None
    fb, notes = evaluate(user_prompt, current, efn)
    return {**state, "feedback": fb, "evaluator_notes": notes}


def _lcel_state_after_modified(state: dict[str, Any]) -> dict[str, Any]:
    """LCEL step: build modified prompt for the second generation."""
    modified = build_modified_prompt(
        str(state["user_prompt"]),
        str(state["current"]),
        str(state["feedback"]),
    )
    return {**state, "modified_prompt": modified}


def _lcel_state_after_revised(state: dict[str, Any]) -> dict[str, Any]:
    """LCEL step: second generation → ``revised_response``."""
    gen_fn = state.get("generate_fn")
    fn = gen_fn if callable(gen_fn) else None
    revised = generate_revised(str(state["modified_prompt"]), fn)
    return {**state, "revised_response": revised}


def build_reflection_lcel_chain() -> Any:
    """
    Single-pass reflection as LangChain LCEL: initial → evaluate → modify → revise.

    Returns:
        A runnable accepting ``{"user_prompt": str}`` and optional ``generate_fn`` /
        ``evaluate_fn`` callables; output dict includes ``revised_response``.

    Raises:
        RuntimeError: If ``langchain_core`` is not installed.
    """
    if RunnableLambda is None:
        raise RuntimeError(
            "langchain-core is required for build_reflection_lcel_chain(); "
            "install from repo root: pip install -r requirements.txt"
        )
    return (
        RunnableLambda(_lcel_state_after_initial)
        | RunnableLambda(_lcel_state_after_evaluate)
        | RunnableLambda(_lcel_state_after_modified)
        | RunnableLambda(_lcel_state_after_revised)
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    user_prompt = "Draft a short apology email for a delayed shipment."

    print("Reflection — Apology email draft revision")
    print("User request:", user_prompt)
    print()

    result = run_reflection(user_prompt)

    print("--- Initial response (not sent to user) ---")
    print(result.initial_response)
    print()
    print("--- Evaluator feedback ---")
    print(result.feedback)
    print("(Notes:", result.evaluator_notes, ")")
    print()
    print("--- Revised response (returned to client) ---")
    print(result.revised_response)
    print()

    print("--- Multi-round (2× evaluate+revise after initial) ---")
    mr = run_reflection_multi_round(user_prompt, revision_rounds=2)
    print("Final revised (excerpt):", mr.revised_response[:120], "...")
    print()

    try:
        lc = build_reflection_lcel_chain()
        out = lc.invoke({"user_prompt": user_prompt})
        print("--- LCEL chain revised_response (excerpt) ---")
        print(str(out.get("revised_response", ""))[:120], "...")
    except RuntimeError as exc:
        print("LCEL: (skipped)", exc)

    print()
    print("In production: use real LLM for generate_initial and generate_revised,")
    print("and LLM-as-Judge (or human) for evaluate. Reflection works in stateless APIs")
    print("because the modified prompt carries the 'memory' (initial + feedback).")


if __name__ == "__main__":
    main()
