#!/usr/bin/env python3
"""
Routing — classify requests and dispatch to handlers (Agentic Design Patterns)

Demonstrates four router families (mocks / pure-Python math) plus an optional LangChain Core
``RunnableBranch`` composition. Swap mocks for live LLM, embedding APIs, or sklearn.

Reference: Antonio Gulli, Agentic Design Patterns — router agent spec.

Usage:
    python example.py
    # Optional LangChain branch: pip install langchain-core (or repo root requirements.txt)
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from langchain_core.runnables import RunnableBranch, RunnableLambda
except ImportError:  # pragma: no cover - optional when langchain-core not installed
    RunnableBranch = None  # type: ignore[misc,assignment]
    RunnableLambda = None  # type: ignore[misc,assignment]

RouteId = Literal["billing", "technical", "general"]


@dataclass(frozen=True)
class RouteDecision:
    """Result of a routing step: target handler and optional diagnostic metadata."""

    route: RouteId
    confidence: float
    mechanism: str


# --- 1) Rule-based routing -------------------------------------------------

_KEYWORDS: dict[RouteId, tuple[str, ...]] = {
    "billing": ("invoice", "refund", "payment", "charge", "subscription"),
    "technical": ("error", "api", "timeout", "bug", "deploy", "build"),
}


def route_by_rules(text: str) -> RouteDecision:
    """
    Route using ordered keyword checks (stand-in for regex, headers, flags).

    Args:
        text: User or system message text.

    Returns:
        A ``RouteDecision`` with mechanism ``rules``.
    """
    lower = text.lower()
    for route, words in _KEYWORDS.items():
        if any(w in lower for w in words):
            return RouteDecision(route=route, confidence=1.0, mechanism="rules")
    return RouteDecision(route="general", confidence=0.5, mechanism="rules")


# --- 2) Embedding-based routing (cosine to handler prototypes) ------------

# Fixed 4-D embeddings for demo only; production: same model as index / RAG.
_PROTOTYPES: dict[RouteId, list[float]] = {
    "billing": [1.0, 0.2, 0.0, 0.1],
    "technical": [0.1, 1.0, 0.3, 0.0],
    "general": [0.0, 0.1, 0.2, 1.0],
}


def _vec_dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Inner product of two equal-length vectors."""
    return float(sum(x * y for x, y in zip(a, b, strict=True)))


def _vec_norm(a: Sequence[float]) -> float:
    """L2 norm."""
    return float(math.sqrt(sum(x * x for x in a))) or 1.0


def _normalize(v: list[float]) -> list[float]:
    """Return a unit vector (L2), or zeros if norm is zero."""
    n = _vec_norm(v)
    return [x / n for x in v]


def _mock_embed(text: str) -> list[float]:
    """
    Deterministic pseudo-embedding from character histogram buckets (demo only).

    Args:
        text: Input string.

    Returns:
        L2-normalized 4-vector.
    """
    # Cheap hash-like features so "invoice" leans toward billing prototype.
    t = text.lower()
    v = [
        float(sum(1 for c in t if c in "aeiou")),
        float(sum(1 for c in t if c.isdigit())),
        float(len(t) % 17),
        float(hash(t) % 1000) / 1000.0,
    ]
    return _normalize(v)


def route_by_embedding(text: str) -> RouteDecision:
    """
    Route to the handler whose prototype embedding is closest in cosine space.

    Args:
        text: User text (embedded via ``_mock_embed`` in this demo).

    Returns:
        A ``RouteDecision`` with mechanism ``embedding``.
    """
    q = _mock_embed(text)
    best: RouteId = "general"
    best_sim = -1.0
    for route, proto in _PROTOTYPES.items():
        sim = _vec_dot(q, proto) / _vec_norm(proto)
        if sim > best_sim:
            best_sim = sim
            best = route
    conf = max(0.0, min(1.0, (best_sim + 1.0) / 2.0))
    return RouteDecision(route=best, confidence=conf, mechanism="embedding")


# --- 3) ML-style linear classifier (mock weights) ---------------------------

_W: list[list[float]] = [
    [0.9, -0.2, 0.1, 0.0],
    [-0.1, 0.85, 0.2, -0.05],
    [0.0, 0.1, -0.1, 0.95],
]
_LABELS: tuple[RouteId, ...] = ("billing", "technical", "general")


def _matvec_row(mat: list[list[float]], x: Sequence[float]) -> list[float]:
    """Matrix-vector multiply (mat is rows x cols, x length cols)."""
    return [_vec_dot(row, x) for row in mat]


def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax over 1-D logits."""
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def route_by_ml_model(text: str) -> RouteDecision:
    """
    Route using a linear layer + softmax on the mock embedding (stand-in for sklearn / ONNX).

    Args:
        text: User text.

    Returns:
        A ``RouteDecision`` with mechanism ``ml_model``.
    """
    x = _mock_embed(text)
    logits = _matvec_row(_W, x)
    probs = _softmax(logits)
    idx = max(range(len(probs)), key=lambda i: probs[i])
    return RouteDecision(
        route=_LABELS[idx],
        confidence=float(probs[idx]),
        mechanism="ml_model",
    )


# --- 4) LLM-based routing (structured JSON mock) ----------------------------

_ROUTE_JSON = re.compile(r'"route"\s*:\s*"(billing|technical|general)"', re.I)


def route_by_llm(text: str) -> RouteDecision:
    """
    Parse a structured route from a mocked LLM response.

    Args:
        text: User text; mock maps keywords to JSON.

    Returns:
        A ``RouteDecision`` with mechanism ``llm``.
    """
    if any(w in text.lower() for w in _KEYWORDS["billing"]):
        raw = '{"route":"billing","confidence":0.92,"rationale":"payment terms"}'
    elif any(w in text.lower() for w in _KEYWORDS["technical"]):
        raw = '{"route":"technical","confidence":0.88,"rationale":"engineering"}'
    else:
        raw = '{"route":"general","confidence":0.61,"rationale":"ambiguous"}'
    m = _ROUTE_JSON.search(raw)
    route = m.group(1).lower() if m else "general"
    data = json.loads(raw)
    conf = float(data.get("confidence", 0.5))
    return RouteDecision(route=route, confidence=conf, mechanism="llm")


# --- 5) LangChain RunnableBranch (compose rule + default) -------------------

def _lc_state_has_billing_keyword(x: dict[str, Any]) -> bool:
    """Predicate for RunnableBranch: billing keywords in ``text``."""
    t = str(x.get("text", "")).lower()
    return any(w in t for w in _KEYWORDS["billing"])


def _lc_state_has_technical_keyword(x: dict[str, Any]) -> bool:
    """Predicate for RunnableBranch: technical keywords in ``text``."""
    t = str(x.get("text", "")).lower()
    return any(w in t for w in _KEYWORDS["technical"])


def build_langchain_router() -> Any:
    """
    Build a small RunnableBranch that annotates state with ``route`` and ``mechanism``.

    Returns:
        A runnable mapping ``{"text": str}`` to ``dict`` with routing fields added.

    Raises:
        RuntimeError: If ``langchain_core`` is not installed.
    """
    if RunnableBranch is None or RunnableLambda is None:
        raise RuntimeError(
            "langchain-core is required for build_langchain_router(); "
            "install from repo root: pip install -r requirements.txt"
        )
    tag_billing = RunnableLambda(
        lambda x: {**x, "route": "billing", "mechanism": "langchain_branch"}
    )
    tag_technical = RunnableLambda(
        lambda x: {**x, "route": "technical", "mechanism": "langchain_branch"}
    )
    tag_general = RunnableLambda(
        lambda x: {**x, "route": "general", "mechanism": "langchain_branch"}
    )
    return RunnableBranch(
        (_lc_state_has_billing_keyword, tag_billing),
        (_lc_state_has_technical_keyword, tag_technical),
        tag_general,
    )


def cascade_route(text: str) -> RouteDecision:
    """
    Multi-stage: rules first, then LLM only if rules are uncertain.

    Args:
        text: User text.

    Returns:
        Final ``RouteDecision``.
    """
    first = route_by_rules(text)
    if first.confidence >= 1.0:
        return first
    return route_by_llm(text)


def dispatch(decision: RouteDecision, handlers: dict[RouteId, Callable[[], str]]) -> str:
    """
    Invoke the handler callable for the decided route.

    Args:
        decision: Output from any router function.
        handlers: Map route id to a zero-arg handler (mock).

    Returns:
        Handler output string.
    """
    return handlers[decision.route]()


def main() -> None:
    """Run small demos for each router style."""
    sample = "My subscription charge looks wrong on the latest invoice"
    print("Sample:", sample)
    print("  rules:      ", route_by_rules(sample))
    print("  embedding:  ", route_by_embedding(sample))
    print("  ml_model:   ", route_by_ml_model(sample))
    print("  llm (mock): ", route_by_llm(sample))
    print("  cascade:    ", cascade_route(sample))

    try:
        lc = build_langchain_router()
        out = lc.invoke({"text": sample})
        print("  langchain:  ", out)
    except RuntimeError as exc:
        print("  langchain:   (skipped)", exc)

    handlers: dict[RouteId, Callable[[], str]] = {
        "billing": lambda: "BillingHandler: open ticket B-1",
        "technical": lambda: "TechHandler: open ticket T-1",
        "general": lambda: "GeneralHandler: FAQ search",
    }
    print("  dispatch:   ", dispatch(route_by_rules(sample), handlers))


if __name__ == "__main__":
    main()
