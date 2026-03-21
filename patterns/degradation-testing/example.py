#!/usr/bin/env python3
"""
Degradation testing — LLM-style metrics (TTFT, EERL, output tok/s, RPS) under load

Problem: Generic load tests miss streaming-specific behavior. This script simulates
streaming inference with asyncio sleeps and reports:

  - TTFT: time until first token
  - EERL: end-to-end request latency (wall time to last token)
  - Output tokens per second: output_tokens / decode_duration
  - Requests per second: completed_requests / total_wall_time

As concurrency increases, latencies typically rise (queueing) — a simple degradation curve.

For real endpoints, use LLMPerf, extend the book's llm_benchmark.py, or LangSmith traces.

Usage:
    python example.py
"""

from __future__ import annotations

import asyncio
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass(frozen=True)
class StreamedRequestMetrics:
    """Per-request measurements aligned with streaming LLM inference."""

    ttft_s: float
    eerl_s: float
    output_tokens: int

    @property
    def decode_duration_s(self) -> float:
        """Duration spent generating tokens after the first token."""
        return max(self.eerl_s - self.ttft_s, 1e-9)

    @property
    def output_tokens_per_second(self) -> float:
        """Approximate generation throughput for this request."""
        return self.output_tokens / self.decode_duration_s


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """
    Nearest-rank percentile for ``q`` in ``[0, 1]``.

    Args:
        sorted_values: Non-decreasing sequence.
        q: Quantile (e.g. 0.95 for p95).

    Returns:
        Value at the given quantile.
    """
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    idx = int(round(q * (len(sorted_values) - 1)))
    return float(sorted_values[idx])


async def simulate_streaming_request(
    rng: random.Random,
    *,
    queue_delay_s: float,
) -> StreamedRequestMetrics:
    """
    Simulate one streaming completion: queue wait, prefill, then decode.

    TTFT is wall time from request start to first token (includes queue + prefill).
    EERL is wall time from request start to last token.

    Args:
        rng: Random source for reproducible jitter.
        queue_delay_s: Simulated time waiting for GPU / batch slot (grows with load).

    Returns:
        Metrics for one synthetic request.
    """
    t0 = time.perf_counter()
    await asyncio.sleep(queue_delay_s)
    prefill_s = rng.uniform(0.002, 0.012)
    await asyncio.sleep(prefill_s)
    t_first = time.perf_counter()
    output_tokens = rng.randint(8, 24)
    per_token = rng.uniform(0.0004, 0.0018)
    for _ in range(output_tokens):
        await asyncio.sleep(per_token)
    t_end = time.perf_counter()
    ttft = t_first - t0
    eerl = t_end - t0
    return StreamedRequestMetrics(ttft_s=ttft, eerl_s=eerl, output_tokens=output_tokens)


async def run_load_level(
    concurrency: int,
    requests: int,
    rng: random.Random,
) -> list[StreamedRequestMetrics]:
    """
    Run ``requests`` simulated calls with at most ``concurrency`` in flight.

    Queue delay increases with concurrency to mimic server-side contention.

    Args:
        concurrency: Max simultaneous requests.
        requests: Total requests to complete.
        rng: Random source.

    Returns:
        List of per-request metrics.
    """
    sem = asyncio.Semaphore(concurrency)
    results: list[StreamedRequestMetrics] = []

    async def one() -> None:
        nonlocal results
        async with sem:
            qdelay = 0.00005 * concurrency * concurrency + rng.uniform(0, 0.002 * concurrency)
            m = await simulate_streaming_request(rng, queue_delay_s=qdelay)
            results.append(m)

    await asyncio.gather(*(one() for _ in range(requests)))
    return results


def summarize(
    metrics: Sequence[StreamedRequestMetrics],
    wall_time_s: float,
) -> dict[str, float]:
    """
    Aggregate TTFT, EERL, tok/s, and RPS.

    Args:
        metrics: Per-request measurements.
        wall_time_s: Wall-clock duration for the whole batch (for RPS).

    Returns:
        Scalar summary fields.
    """
    n = len(metrics)
    if n == 0:
        return {
            "n": 0.0,
            "rps": 0.0,
            "ttft_p50_ms": 0.0,
            "ttft_p95_ms": 0.0,
            "eerl_p50_ms": 0.0,
            "eerl_p95_ms": 0.0,
            "out_tps_mean": 0.0,
        }
    ttfts = sorted(m.ttft_s for m in metrics)
    eerls = sorted(m.eerl_s for m in metrics)
    tps = [m.output_tokens_per_second for m in metrics]
    return {
        "n": float(n),
        "rps": n / wall_time_s if wall_time_s > 0 else 0.0,
        "ttft_p50_ms": percentile(ttfts, 0.50) * 1000,
        "ttft_p95_ms": percentile(ttfts, 0.95) * 1000,
        "eerl_p50_ms": percentile(eerls, 0.50) * 1000,
        "eerl_p95_ms": percentile(eerls, 0.95) * 1000,
        "out_tps_mean": float(statistics.mean(tps)),
    }


async def run_degradation_demo() -> None:
    """Print a table of metrics vs concurrency for a fixed request count."""
    rng = random.Random(42)
    request_count = 16
    levels = [1, 4, 8, 16]
    print("Pattern 27: Degradation testing (synthetic streaming load)\n")
    print(
        f"{'conc':>4}  {'RPS':>8}  {'TTFT p50':>10}  {'TTFT p95':>10}  "
        f"{'EERL p50':>10}  {'EERL p95':>10}  {'out tok/s μ':>12}"
    )
    print(f"{'':>4}  {'':>8}  {'(ms)':>10}  {'(ms)':>10}  {'(ms)':>10}  {'(ms)':>10}  {'':>12}")
    for c in levels:
        t0 = time.perf_counter()
        batch = await run_load_level(c, request_count, rng)
        wall = time.perf_counter() - t0
        s = summarize(batch, wall)
        print(
            f"{c:4d}  {s['rps']:8.2f}  {s['ttft_p50_ms']:10.1f}  {s['ttft_p95_ms']:10.1f}  "
            f"{s['eerl_p50_ms']:10.1f}  {s['eerl_p95_ms']:10.1f}  {s['out_tps_mean']:12.1f}"
        )
    print("\nEERL = end-to-end request latency (submit → last token).")
    print("Tools: LLMPerf (github.com/ray-project/llmperf), LangSmith tracing (docs.smith.langchain.com).")


def main() -> None:
    """Entry point."""
    asyncio.run(run_degradation_demo())


if __name__ == "__main__":
    main()
