#!/usr/bin/env python3
"""
Parallelization — concurrent independent branches then aggregation (Agentic Design Patterns)

Includes:
  1) ``concurrent.futures`` fan-out (stdlib) for CPU/IO-bound style work.
  2) LangChain LCEL: ``RunnableParallel`` chained to a merge runnable (optional).

Reference: Antonio Gulli, Agentic Design Patterns — parallelizer agent spec.

Usage:
    python example.py
    # LCEL section: pip install langchain-core (or repo root requirements.txt)
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from langchain_core.runnables import RunnableLambda, RunnableParallel
except ImportError:  # pragma: no cover - optional when langchain-core not installed
    RunnableLambda = None  # type: ignore[misc,assignment]
    RunnableParallel = None  # type: ignore[misc,assignment]


def mock_research_gather(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Stand-in for parallel source retrieval (search APIs, RAG shards).

    Args:
        payload: Must include ``query`` (str).

    Returns:
        Structured snippets and synthetic source ids.
    """
    q = str(payload["query"])
    return {
        "branch": "gather",
        "snippets": [
            f"[src:wiki] Summary angle for: {q[:40]}",
            f"[src:news] Recent mention: {q[:40]}",
        ],
    }


def mock_analyze(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Stand-in for analytics / feature extraction on the same request.

    Args:
        payload: Must include ``query``.

    Returns:
        Simple numeric/text metrics.
    """
    q = str(payload["query"])
    words = len(q.split())
    return {
        "branch": "analyze",
        "metrics": {"word_count": words, "complexity": "high" if words > 12 else "low"},
    }


def mock_verify(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Stand-in for parallel validation (policy, citations, consistency probes).

    Args:
        payload: Must include ``query``.

    Returns:
        Checklist-style result.
    """
    return {
        "branch": "verify",
        "checks": ["source_diversity", "no_pii_in_query"],
        "passed": True,
    }


def mock_multimodal(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Stand-in for parallel modality pipelines (e.g. image caption + transcript).

    Args:
        payload: Must include ``query``; optional ``modalities``.

    Returns:
        Stub modality channel list.
    """
    modalities = payload.get("modalities") or ["text"]
    return {
        "branch": "multimodal",
        "channels": list(modalities),
        "note": "Swap for vision/audio runnables bound to real encoders",
    }


def run_parallel_stdlib(
    payload: dict[str, Any],
    branches: list[Callable[[dict[str, Any]], dict[str, Any]]],
    max_workers: int = 4,
) -> dict[str, Any]:
    """
    Execute branch callables concurrently and collect results by branch name.

    Args:
        payload: Shared input passed to every branch.
        branches: Independent callables returning dicts with a ``branch`` key.
        max_workers: Upper bound on concurrent threads.

    Returns:
        Mapping ``branch_name -> result``; entries for failed branches include ``error``.
    """
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(fn, payload): fn for fn in branches}
        for fut in as_completed(future_map):
            try:
                out = fut.result()
                key = str(out.get("branch", "unknown"))
                results[key] = out
            except Exception as exc:
                results[f"error_{id(fut)}"] = {"error": repr(exc)}
    return results


def merge_parallel_outputs(parts: dict[str, Any]) -> dict[str, Any]:
    """
    Reduce parallel branch outputs into a single artifact (synthesis stub).

    Args:
        parts: Either stdlib aggregate (branch -> dict) or LCEL parallel dict.

    Returns:
        A compact report dict for downstream prompting.
    """
    return {
        "summary_lines": [
            f"{name}: ok" if isinstance(body, dict) and "error" not in body else f"{name}: failed"
            for name, body in sorted(parts.items())
        ],
        "raw": parts,
    }


def build_lcel_research_pipeline() -> Any:
    """
    Build ``RunnableParallel`` over four independent runnables, then merge.

    Returns:
        A LCEL runnable: ``dict -> dict``.

    Raises:
        RuntimeError: If ``langchain_core`` is not installed.
    """
    if RunnableParallel is None or RunnableLambda is None:
        raise RuntimeError(
            "langchain-core is required for build_lcel_research_pipeline(); "
            "install from repo root: pip install -r requirements.txt"
        )
    return RunnableParallel(
        gather=RunnableLambda(mock_research_gather),
        analyze=RunnableLambda(mock_analyze),
        verify=RunnableLambda(mock_verify),
        multimodal=RunnableLambda(mock_multimodal),
    ) | RunnableLambda(merge_parallel_outputs)


def main() -> None:
    """Demonstrate stdlib parallel execution and optional LCEL."""
    payload: dict[str, Any] = {
        "query": "Compare edge caching vs origin shielding for global video delivery",
        "modalities": ["text", "image_stub"],
    }
    branches = [mock_research_gather, mock_analyze, mock_verify, mock_multimodal]
    stdlib_out = run_parallel_stdlib(payload, branches)
    print("stdlib ThreadPoolExecutor branches:")
    for k, v in sorted(stdlib_out.items()):
        print(f"  {k}: {v}")

    merged = merge_parallel_outputs(stdlib_out)
    print("merged report keys:", list(merged.keys()))

    try:
        lc = build_lcel_research_pipeline()
        lc_out = lc.invoke(payload)
        print("LCEL RunnableParallel | merge:")
        print("  summary_lines:", lc_out.get("summary_lines"))
    except RuntimeError as exc:
        print("LCEL: (skipped)", exc)


if __name__ == "__main__":
    main()
