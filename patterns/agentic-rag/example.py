#!/usr/bin/env python3
"""
Agentic RAG — chunking, vector similarity (mock), optional LangChain LCEL stub.

Educational only: no real embedding API or vector DB. Maps to Pattern 39 README;
for full pipelines see Pattern 6 (basic-rag) and LangChain / LangGraph tutorials.

Reference: Antonio Gulli, Agentic Design Patterns — rag-retriever agent spec.

Usage:
    python example.py
    # Optional: pip install langchain-core for build_lcel_rag_stub()
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
except ImportError:  # pragma: no cover - optional
    RunnableLambda = None  # type: ignore[misc,assignment]
    RunnablePassthrough = None  # type: ignore[misc,assignment]


def chunk_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """
    Split text into overlapping windows (character-based demo).

    Production systems usually chunk by tokens and respect structure (headings).

    Args:
        text: Full document string.
        max_chars: Maximum characters per chunk.
        overlap: Characters repeated between consecutive chunks.

    Returns:
        Non-empty chunk strings in order.
    """
    t = text.strip()
    if not t:
        return []
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    step = max(1, max_chars - max(0, overlap))
    chunks: list[str] = []
    i = 0
    while i < len(t):
        piece = t[i : i + max_chars]
        if piece:
            chunks.append(piece)
        i += step
    return chunks


def _pseudo_embed(text: str, dim: int = 8) -> list[float]:
    """
    Deterministic fake embedding for demo similarity (not semantic quality).

    Args:
        text: Input string.
        dim: Vector dimension.

    Returns:
        L2-normalized vector of length ``dim``.
    """
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(text[:200]))
    rng = (seed % 997) / 997.0
    raw = [math.sin(rng + j * 0.7) + 0.01 * (hash(text + str(j)) % 17) for j in range(dim)]
    n = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / n for x in raw]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity of two equal-length vectors (not necessarily unit length).

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Similarity in [-1, 1] for typical embeddings.
    """
    if len(a) != len(b) or not a:
        raise ValueError("vectors must be non-empty and equal length")
    dot_ab = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot_ab / (na * nb)


def l2_distance(a: list[float], b: list[float]) -> float:
    """
    Euclidean distance between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Non-negative distance.
    """
    if len(a) != len(b):
        raise ValueError("vectors must be equal length")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def mock_vector_retrieve(
    query: str,
    corpus: dict[str, list[float]],
    *,
    top_k: int = 2,
) -> list[tuple[str, float]]:
    """
    Score corpus keys by cosine similarity to a pseudo-embedding of the query.

    Args:
        query: User question.
        corpus: Mapping chunk_id -> precomputed vector (here: pseudo_embed of chunk text).
        top_k: Number of results.

    Returns:
        List of ``(chunk_id, similarity)`` descending by similarity.
    """
    qv = _pseudo_embed(query)
    scored: list[tuple[str, float]] = []
    for cid, vec in corpus.items():
        scored.append((cid, cosine_similarity(qv, vec)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


class TinyEntityGraph:
    """Minimal undirected graph for Graph RAG illustration (entity co-occurrence)."""

    def __init__(self) -> None:
        self._edges: dict[str, set[str]] = {}

    def add_edge(self, a: str, b: str) -> None:
        """Register an undirected edge between two entity ids."""
        if a not in self._edges:
            self._edges[a] = set()
        if b not in self._edges:
            self._edges[b] = set()
        self._edges[a].add(b)
        self._edges[b].add(a)

    def neighbors(self, entity: str) -> set[str]:
        """Return adjacent entity ids."""
        return set(self._edges.get(entity, ()))


def agentic_rag_stub(query: str, retriever: Any, max_rounds: int = 2) -> dict[str, Any]:
    """
    Simulate multi-step retrieval: refine query with trivial heuristic each round.

    Args:
        query: Initial user query.
        retriever: Callable taking a string, returning list of doc snippets.
        max_rounds: Number of retrieve passes.

    Returns:
        Dict with ``queries`` tried and ``docs`` merged (deduped order preserved).
    """
    seen: list[str] = []
    queries = [query]
    current = query
    for _ in range(max_rounds):
        batch = retriever(current)
        for d in batch:
            if d not in seen:
                seen.append(d)
        current = re.sub(r"\s+", " ", (current + " " + " ".join(batch))).strip()[:120]
        queries.append(current)
    return {"queries": queries, "docs": seen}


def build_lcel_rag_stub() -> Any:
    """
    Minimal LCEL chain: assign retrieved docs to state, then format a prompt string.

    Returns:
        A runnable ``{"question": str} -> {"question", "docs", "prompt"}``.

    Raises:
        RuntimeError: If ``langchain_core`` is not installed.
    """
    if RunnablePassthrough is None or RunnableLambda is None:
        raise RuntimeError(
            "langchain-core is required for build_lcel_rag_stub(); "
            "pip install -r requirements.txt"
        )

    corpus = {
        "c1": _pseudo_embed("OAuth token refresh flow for mobile clients"),
        "c2": _pseudo_embed("Password reset sends email within five minutes"),
    }

    def retrieve(state: dict[str, Any]) -> dict[str, Any]:
        q = str(state["question"])
        hits = mock_vector_retrieve(q, corpus, top_k=2)
        return {**state, "docs": [h[0] for h in hits]}

    def to_prompt(state: dict[str, Any]) -> dict[str, Any]:
        ctx = ", ".join(state.get("docs", []))
        prompt = f"Context ids: {ctx}\nQuestion: {state['question']}\nAnswer concisely."
        return {**state, "prompt": prompt}

    return RunnablePassthrough() | RunnableLambda(retrieve) | RunnableLambda(to_prompt)


def main() -> None:
    """Run chunking, similarity, mock retrieval, graph neighbor demo, optional LCEL."""
    doc = "Paragraph one about OAuth. " * 5 + "Paragraph two about billing. " * 5
    chunks = chunk_text(doc, max_chars=80, overlap=20)
    print("chunks:", len(chunks), "first chunk length:", len(chunks[0]) if chunks else 0)

    u = _pseudo_embed("OAuth security")
    v = _pseudo_embed("OAuth security")
    w = _pseudo_embed("unrelated topic xyz")
    print("cosine same-topic-ish:", round(cosine_similarity(u, v), 4))
    print("cosine different:", round(cosine_similarity(u, w), 4))
    print("l2 distance u vs w:", round(l2_distance(u, w), 4))

    corpus = {f"chunk_{i}": _pseudo_embed(c) for i, c in enumerate(chunks[:5])}
    hits = mock_vector_retrieve("OAuth token", corpus)
    print("mock retrieve:", hits)

    g = TinyEntityGraph()
    g.add_edge("User", "Order")
    g.add_edge("Order", "Payment")
    print("graph neighbors of Order:", g.neighbors("Order"))

    def fake_ret(q: str) -> list[str]:
        return ["doc:A"] if "OAuth" in q else ["doc:B"]

    ar = agentic_rag_stub("Explain OAuth", fake_ret, max_rounds=2)
    print("agentic_rag_stub rounds:", ar["queries"])

    try:
        chain = build_lcel_rag_stub()
        out = chain.invoke({"question": "How do I refresh tokens?"})
        print("LCEL prompt excerpt:", out.get("prompt", "")[:80], "...")
    except RuntimeError as exc:
        print("LCEL:", exc)


if __name__ == "__main__":
    main()
