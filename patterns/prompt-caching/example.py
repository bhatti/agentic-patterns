#!/usr/bin/env python3
"""
Prompt caching — exact memoization vs semantic similarity (toy demo)

Problem: Repeating the same or similar prompts wastes latency and money.

Solution layers:
  - Exact key cache (client-side memoization)
  - Semantic cache (similar text → reuse answer), here using word-vector cosine
  - Provider server-side prefix caching (see README; not simulated here)

This script uses **stdlib only** and a **mock** LLM. Swap ``mock_llm`` for Ollama or a cloud
client; keep the same cache wrappers.

Usage:
    python example.py
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def normalize_text(text: str) -> str:
    """
    Lowercase and strip punctuation for crude token alignment.

    Args:
        text: Raw user prompt.

    Returns:
        Normalized string for tokenization.
    """
    return re.sub(r"[^\w\s]", " ", text.lower())


def word_counts(text: str) -> Counter[str]:
    """
    Build a bag-of-words count vector.

    Args:
        text: Input prompt.

    Returns:
        Mapping of token to count.
    """
    parts = normalize_text(text).split()
    return Counter(parts)


def cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    """
    Compute cosine similarity between two sparse word-count vectors.

    Args:
        a: First count vector.
        b: Second count vector.

    Returns:
        Similarity in ``[0, 1]`` for non-negative counts.
    """
    keys = set(a) | set(b)
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k in keys:
        va = float(a.get(k, 0))
        vb = float(b.get(k, 0))
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def cache_key_exact(prompt: str, model_id: str) -> str:
    """
    Build a stable key for exact-match caching.

    Args:
        prompt: Full prompt text.
        model_id: Model identifier included so different models do not share entries.

    Returns:
        Hex digest used as dict key.
    """
    payload = f"{model_id}\n{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ExactPromptCache:
    """
    In-memory exact-match cache: same model + same prompt → stored completion.
    """

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._store: dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        """
        Look up a cached completion.

        Args:
            key: Key from ``cache_key_exact``.

        Returns:
            Cached text or None.
        """
        return self._store.get(key)

    def set(self, key: str, completion: str) -> None:
        """
        Store a completion.

        Args:
            key: Key from ``cache_key_exact``.
            completion: Model output to reuse later.
        """
        self._store[key] = completion


class SemanticPromptCache:
    """
    Retrieve completions when a new prompt is *similar* to a cached prompt.

    Uses bag-of-words cosine similarity—production systems often use embeddings instead.
    """

    def __init__(self, similarity_threshold: float) -> None:
        """
        Args:
            similarity_threshold: Minimum cosine similarity to accept a hit (0–1).
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        self._threshold = similarity_threshold
        self._entries: list[tuple[str, Counter[str], str]] = []

    def lookup(self, prompt: str) -> Optional[tuple[str, float]]:
        """
        Find the best matching cached prompt if similarity is above threshold.

        Args:
            prompt: New user prompt.

        Returns:
            Tuple of (cached_completion, similarity) or None.
        """
        q = word_counts(prompt)
        best_sim = -1.0
        best_text: Optional[str] = None
        for _orig, vec, completion in self._entries:
            sim = cosine_similarity(q, vec)
            if sim > best_sim:
                best_sim = sim
                best_text = completion
        if best_text is not None and best_sim >= self._threshold:
            return (best_text, best_sim)
        return None

    def remember(self, prompt: str, completion: str) -> None:
        """
        Store a prompt vector and its completion.

        Args:
            prompt: Prompt that was sent to the model.
            completion: Model output to associate.
        """
        self._entries.append((prompt, word_counts(prompt), completion))


def mock_llm(prompt: str, *, sleep_s: float = 0.05) -> str:
    """
    Stand-in for an LLM call: deterministic, slow enough to show cache wins.

    Args:
        prompt: User prompt.
        sleep_s: Simulated network/compute delay.

    Returns:
        Fake completion string.
    """
    time.sleep(sleep_s)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    return f"[mock-llm] response for prompt hash={digest}"


def run_exact_cache_demo(model_id: str) -> None:
    """
    Demonstrate two identical calls: second call hits exact cache.

    Args:
        model_id: Fake model name included in the cache key.
    """
    prompt = "Define prompt caching in one sentence."
    cache = ExactPromptCache()
    key = cache_key_exact(prompt, model_id)

    t0 = time.perf_counter()
    r1 = cache.get(key) or mock_llm(prompt)
    if cache.get(key) is None:
        cache.set(key, r1)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    r2 = cache.get(key) or mock_llm(prompt)
    if cache.get(key) is None:
        cache.set(key, r2)
    t3 = time.perf_counter()

    print("--- Exact cache ---")
    print("  first call (uncached) ms:", round((t1 - t0) * 1000, 2))
    print("  second call (cached) ms:", round((t3 - t2) * 1000, 2))
    print("  responses match:", r1 == r2)


def run_semantic_cache_demo() -> None:
    """
    Show a near-duplicate prompt hitting semantic cache after priming with a similar prompt.
    """
    sem = SemanticPromptCache(similarity_threshold=0.55)
    p1 = "What is retrieval-augmented generation?"
    c1 = mock_llm(p1)
    sem.remember(p1, c1)

    p2 = "Explain retrieval augmented generation briefly."
    hit = sem.lookup(p2)
    if hit:
        completion, sim = hit
        print("\n--- Semantic cache ---")
        print("  primed with:", p1[:50], "...")
        print("  similar query:", p2[:50], "...")
        print("  hit similarity:", round(sim, 3))
        print("  reused completion:", completion[:60], "...")
        return

    c2 = mock_llm(p2)
    sem.remember(p2, c2)
    print("\n--- Semantic cache ---")
    print("  (threshold too strict for this demo pair; stored second call.)")
    print("  completion:", c2[:60], "...")


def langchain_cache_note() -> str:
    """
    Return documentation string for LangChain LLM caching (no import).

    Returns:
        Short usage note.
    """
    return (
        "LangChain: from langchain_core.globals import set_llm_cache\n"
        "from langchain_community.cache import SQLiteCache\n"
        "set_llm_cache(SQLiteCache(database_path='.langchain.db'))"
    )


def main() -> None:
    """Run exact and semantic demos and print a LangChain note."""
    print("Pattern 25: Prompt caching\n")
    run_exact_cache_demo(model_id="mock-model-v1")
    run_semantic_cache_demo()
    print("\n--- Framework note (LangChain) ---")
    print(langchain_cache_note())
    print("\n--- Server-side ---")
    print("  Anthropic / OpenAI: use provider prompt caching APIs for long static prefixes;")
    print("  see patterns/prompt-caching/README.md for links.")


if __name__ == "__main__":
    main()
