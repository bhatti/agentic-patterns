# Pattern 27: Degradation Testing

## Overview

**Degradation testing** (for LLM inference) is **load and performance testing** with **LLM-native metrics**. Generic HTTP benchmarks (p95 latency, errors/sec) miss **prefill vs decode**, **streaming**, and **token throughput**. You need **fine-grained** probes under **rising concurrency** so you see **when** TTFT, end-to-end latency, or tokens/sec **fall below SLOs**—before production traffic does.

## Problem Statement

- **Inference stacks** (vLLM, TGI, cloud APIs) behave differently from **stateless** microservices: **KV cache**, **batching**, and **queueing** interact with **prompt length** and **output length**.
- **Conventional load tests** (e.g. “200 RPS with p95 &lt; 500 ms”) do not separate **time to first token** from **generation speed** or **output tokens per second**.
- **Regulated** or **self-hosted** setups still need **repeatable** benchmarks that match **chat** and **RAG** patterns.

## Solution Overview

### Core metrics (four pillars)

| Metric | Meaning | Why it matters |
|--------|---------|----------------|
| **TTFT** | **Time to first token** — request sent → first streamed (or first) token | Perceived **responsiveness**; dominated by **prefill** and queue wait |
| **EERL** | **End-to-end request latency** — full wall time until **completion** (last token or full body) | Total **user wait**; includes **decode** |
| **Output tokens / second** | **Generation throughput** — output tokens divided by **decode** duration (often \( \text{EERL} - \text{TTFT} \) for streaming) | **Lengthy** answers; batching efficiency |
| **Requests / second** | **Sustained** completed requests per wall-clock second under a given load | **Capacity**; queue saturation |

**Note:** Vendors and tools sometimes use **ITL** (inter-token latency) or **TPOT** (time per output token); they relate to **output tokens / second** (inverse, per token).

### Degradation = SLO breach under stress

Run **stepped load** (e.g. concurrency 1 → 4 → 16 → 64) and plot **p50/p95** of TTFT and EERL, **mean** output tokens/s, and **achieved RPS**. **Degradation** is the **curve**: when **TTFT p95** spikes or **success rate** drops, you have found the **knee** of the system.

### Ecosystem tools

- **[LLMPerf](https://github.com/ray-project/llmperf)** (Ray / Anyscale): Open-source **load testing** for OpenAI-compatible and other APIs; reports **TTFT**, **inter-token latency**, **output throughput**, **requests per minute**, **success rate**—aligned with reproducible **leaderboard** comparisons ([Anyscale overview](https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference)).
- **[LangSmith](https://docs.smith.langchain.com/)**: Tracing and evaluation for LangChain apps; use it to **record** per-run **latency**, token usage, and failures during **soak** tests—not a replacement for dedicated **load generators**, but essential for **debugging** which layer degrades (retrieval vs model vs tool calls).

Reference book code: `generative-ai-design-patterns/examples/27_degradation_testing/` (`llm_benchmark.py`, streaming **TTFT** / **tokens_per_second**, concurrent users).

### High-level flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4f46e5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#4338ca', 'lineColor': '#6366f1', 'secondaryColor': '#10b981', 'tertiaryColor': '#f59e0b'}}}%%
flowchart LR
    L["Load generator"]
    S["Inference endpoint"]
    M["TTFT, EERL, tok/s, RPS"]
    L --> S
    S --> M
```

## Use Cases

- **Pre-production** sign-off for **SLOs** (chat, copilot, RAG)
- **Comparing** two serving configs (batch size, model revision, hardware)

## Implementation Details

- **Warm up** the server before measuring; discard first N requests.
- **Fix** prompt and **max_tokens** when comparing runs; vary **concurrency** as the independent variable.
- **Stream** if production streams—TTFT only applies meaningfully with **streaming**.
- **Record** errors and **timeouts** separately from latency percentiles.

## Constraints & Tradeoffs

**Tradeoffs:** ✅ Insight into real LLM behavior. ⚠️ Benchmarks are **sensitive** to network, rate limits, and **caching**; repeat runs and **isolate** variables.

## References

- Book examples: `examples/27_degradation_testing/` (`llm_benchmark.py`, OpenAI/Qwen variants).
- [LLMPerf (GitHub)](https://github.com/ray-project/llmperf)
- [LangSmith documentation](https://docs.smith.langchain.com/)
- **Pattern 26 (Inference optimization)** — what you tune **after** measuring

## Related Patterns

- **Inference optimization (26)**: continuous batching, speculative decoding—validated with **degradation** curves
- **Prompt caching (25)**: can **inflate** hit-rate metrics—separate **cold** vs **warm** tests
- **Evaluation and monitoring (42)**: **Broader** **observability** **layer**—**degradation** **tests** **feed** **dashboards** **and** **alerts**
