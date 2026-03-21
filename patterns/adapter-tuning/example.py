#!/usr/bin/env python3
"""
Adapter Tuning Pattern - Support Ticket Intent Routing

This example illustrates adapter tuning using a small, runnable setup:
a frozen "foundation" (text representation) plus a small trainable "adapter"
(classifier) on hundreds of input-output pairs. No GPU required.

Real-World Problem:
-------------------
Support teams need to route tickets (billing, technical, sales, etc.) quickly.
Prompt-only or few-shot classification can be brittle. Adapter tuning trains
a small task-specific head on a few hundred labeled tickets while keeping
the base representation fixed — analogous to freezing an LLM and training
only LoRA/adapter layers.

Pattern in code:
- Foundation (frozen): TF-IDF text representation — stand-in for frozen
  pretrained encoder/LLM hidden states.
- Adapter (trained): Small classifier (e.g. linear or MLP) trained on
  (text, intent) pairs. In production this would be LoRA/QLoRA on top
  of a frozen HuggingFace model.

Usage:
    pip install scikit-learn  # if not already installed
    python example.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Project root for shared utilities if needed
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
except ImportError:
    print(
        "This example requires scikit-learn. Install with: pip install scikit-learn",
        file=sys.stderr,
    )
    sys.exit(1)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class TicketExample:
    """Single support ticket with text and intent label."""
    text: str
    intent: str  # billing | technical | sales | general


@dataclass
class AdapterTuningResult:
    """Result of adapter-tuning style training and inference."""
    intent: str
    confidence: float
    all_scores: List[Tuple[str, float]] = field(default_factory=list)


# =============================================================================
# SYNTHETIC TRAINING DATA (few hundred pairs — adapter tuning scale)
# =============================================================================

def _build_ticket_dataset() -> List[TicketExample]:
    """
    Build a small dataset of (ticket text, intent) pairs.
    In production, replace with real labeled tickets (e.g. 200–2000 examples).
    """
    raw: List[Tuple[str, str]] = []

    # Billing
    billing_phrases = [
        "I was charged twice for my subscription",
        "Please cancel my subscription and refund last month",
        "Update my payment method",
        "Why was I charged extra fees",
        "Invoice for last quarter",
        "Wrong amount on my bill",
        "I want to dispute a charge",
        "Payment failed but money was deducted",
        "Need a copy of my invoice",
        "Upgrade my plan and prorate",
    ]
    raw.extend((t, "billing") for t in billing_phrases)

    # Technical
    technical_phrases = [
        "App crashes when I open the dashboard",
        "API returns 500 error",
        "How do I integrate with our SSO",
        "Connection timeout after 30 seconds",
        "Export is failing for large datasets",
        "Webhook not receiving events",
        "Authentication token expired",
        "Slow performance on mobile",
        "Cannot connect to database",
        "Deployment failed on staging",
    ]
    raw.extend((t, "technical") for t in technical_phrases)

    # Sales
    sales_phrases = [
        "I need a quote for enterprise plan",
        "Do you offer volume discounts",
        "Schedule a demo for my team",
        "Comparing your product with competitor X",
        "What is the pricing for 500 seats",
        "Trial extension request",
        "Feature availability in premium tier",
        "Custom SLA options",
        "Multi-year contract pricing",
        "Partner program information",
    ]
    raw.extend((t, "sales") for t in sales_phrases)

    # General
    general_phrases = [
        "Where can I find the documentation",
        "How do I contact support",
        "What are your business hours",
        "Request for account deletion",
        "Feedback on the new UI",
        "Where is the status page",
        "How to report a security issue",
        "Careers at your company",
        "Press or media contact",
        "Partnership inquiry",
    ]
    raw.extend((t, "general") for t in general_phrases)

    # Scale to ~hundreds by repeating with small variations (simulate real variance)
    examples: List[TicketExample] = []
    for text, intent in raw:
        examples.append(TicketExample(text=text, intent=intent))
        # Add slight paraphrases to reach adapter-tuning data scale without real data
        for suffix in ["", " Thanks.", " Can you help?", " Urgent."]:
            if suffix and len(examples) < 400:
                examples.append(TicketExample(text=text + suffix, intent=intent))

    return examples


# =============================================================================
# FOUNDATION (FROZEN) + ADAPTER (TRAINED)
# =============================================================================

class TicketIntentRouter:
    """
    Adapter-tuning style router: frozen foundation (TF-IDF) + trainable adapter (classifier).

    Conceptually:
    - Foundation: fixed text → vector mapping (here TF-IDF). In production:
      frozen HuggingFace encoder/LLM producing hidden states.
    - Adapter: small trainable head that maps foundation output to task labels.
      In production: LoRA/QLoRA layers or a small classification head on [CLS].
    """

    def __init__(self, max_features: int = 2000, adapter_max_iter: int = 500) -> None:
        self.max_features = max_features
        self.adapter_max_iter = adapter_max_iter
        self._pipeline: Pipeline | None = None
        self._intent_order: List[str] = []

    def train(self, examples: List[TicketExample]) -> None:
        """
        Train only the adapter (classifier). Foundation (TF-IDF) is fit once
        and then used as a fixed feature extractor — no further updates.
        """
        texts = [ex.text for ex in examples]
        labels = [ex.intent for ex in examples]

        # Foundation: TF-IDF (frozen after fit — no gradient updates in this demo)
        foundation = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )

        # Adapter: small classifier (the only trainable part in real PEFT)
        adapter = LogisticRegression(
            max_iter=self.adapter_max_iter,
            random_state=42,
            C=0.5,
        )

        # Pipeline: foundation → adapter (foundation fixed during adapter training in spirit)
        self._pipeline = Pipeline(
            [
                ("foundation", foundation),
                ("adapter", adapter),
            ]
        )
        self._pipeline.fit(texts, labels)
        self._intent_order = list(self._pipeline.classes_)

    def predict(self, text: str) -> AdapterTuningResult:
        """Route a single ticket to an intent with confidence."""
        if self._pipeline is None:
            raise RuntimeError("Router not trained. Call train() first.")

        pred = self._pipeline.predict([text])[0]
        probs = self._pipeline.predict_proba([text])[0]
        scores = list(zip(self._intent_order, probs.tolist()))
        scores.sort(key=lambda x: -x[1])

        return AdapterTuningResult(
            intent=pred,
            confidence=float(probs[self._intent_order.index(pred)]),
            all_scores=scores,
        )


# =============================================================================
# MAIN: TRAIN AND RUN
# =============================================================================

def main() -> None:
    # 1. Small dataset (adapter-tuning scale: hundreds of pairs)
    examples = _build_ticket_dataset()
    train_examples, test_examples = train_test_split(
        examples, test_size=0.2, random_state=42, stratify=[ex.intent for ex in examples]
    )

    # 2. Train router (frozen foundation + adapter)
    router = TicketIntentRouter()
    router.train(train_examples)

    # 3. Evaluate on held-out set
    correct = 0
    for ex in test_examples:
        result = router.predict(ex.text)
        if result.intent == ex.intent:
            correct += 1
    accuracy = correct / len(test_examples) if test_examples else 0.0
    print(f"Adapter tuning (frozen foundation + adapter) — Test accuracy: {accuracy:.2%}")
    print()

    # 4. Demo predictions
    demo_texts = [
        "I was charged twice, please refund.",
        "API keeps returning 500.",
        "Can we get a quote for 200 seats?",
        "Where is the documentation?",
    ]
    print("Demo predictions:")
    for text in demo_texts:
        result = router.predict(text)
        print(f"  \"{text[:50]}...\"" if len(text) > 50 else f"  \"{text}\"")
        print(f"    -> {result.intent} (confidence: {result.confidence:.2f})")
        print()

    print(
        "In production: use a real foundation model (e.g. HuggingFace) with frozen weights\n"
        "and train only a PEFT adapter (e.g. LoRA/QLoRA) on your (input, output) pairs."
    )


if __name__ == "__main__":
    main()
