# Pattern 35: Parallelization

## Overview

**Parallelization** runs **independent** subtasks **concurrently**—then **synchronizes** and **aggregates** results. In agentic systems it cuts **wall-clock** latency when work is **embarrassingly parallel**: multi-source **research**, **fan-out** retrieval, **batch** analytics, **multi-critic** verification, or **multimodal** branches (text, vision, audio) that do not depend on each other’s outputs until a **merge** step.

This pattern is catalogued in **Agentic Design Patterns** (*Antonio Gulli*); companion agent spec: `subagents-design-patterns/agents/parallelizer.md`. It complements **prompt chaining** (Pattern 33): **chain** when order matters; **parallelize** when subtasks are **orthogonal**. **Multi-agent** (23) may use parallel **workers**; **routing** (34) may **spawn** parallel paths per intent.

## Problem Statement

- **Sequential** LLM or tool calls for **unrelated** subtasks waste **time** and user patience.
- **Research**, **data** pipelines, and **verification** often need **N** independent lookups or checks before **one** synthesis.
- **Multimodal** products need **parallel** extractors (e.g. **OCR** + **ASR** + **metadata**) feeding a **single** fusion step.

## Solution Overview

1. **Decompose** the task into **independent** units (prove independence; watch **hidden** shared state).
2. **Execute** with **async**, **thread pools**, **process pools**, or orchestration primitives (**LangChain LCEL** `RunnableParallel`, **LangGraph** `Send`/map-reduce).
3. **Bound** concurrency (rate limits, GPU memory, API quotas); **fail** or **partial** aggregate per subtask.
4. **Aggregate**: merge dicts, **rank**, **vote**, or a final **LLM** “synthesize” step over parallel outputs.
5. **Observe**: trace **per-branch** latency and errors (avoid PII in hot-path logs per project norms).

### Use case mapping

| Use case | Parallel branches (examples) | Merge |
|----------|------------------------------|--------|
| **Information gathering / research** | Multiple **retrievers**, **search** APIs, **domain** experts | **Deduplicate**, cite, **synthesize** |
| **Data processing / analysis** | **Partition** rows, **map** stats, **feature** extractions | **Reduce**, join, report |
| **Validation / verification** | **Critic**, **policy** check, **fact** probe, **LLM-as-judge** (17) | **Conjunction** / weighted **vote** |
| **Multimodal processing** | **Vision** caption, **audio** transcript, **text** RAG | **Align** tokens / **fusion** prompt |

### Generative AI Design Patterns (Lakshmanan et al.)

**Deep Search** (12), **RAG** families (6–10), and **multi-agent** (23) often combine **parallel** retrieval or specialists with a **single** consolidation step. **Trustworthy Generation** (11) and **self-check** (31) can run **parallel** checks before release.

## Implementation details

- **LangChain LCEL**: `RunnableParallel({ ... })` fans out the **same** input to named runnables; chain with `|` to a **reducer** runnable. Prefer **immutable** branch outputs for easier testing.
- **LangGraph**: use **parallel** edges or **map** patterns; respect **checkpointing** if branches are long.
- **Dependencies**: only introduce **cross-branch** ordering when the merge step **requires** it (then use **chain** or **barrier**).

## Constraints & Tradeoffs

**Tradeoffs:** ✅ Lower latency for independent work; simpler **per-branch** retries. ⚠️ **Thundering herd** on APIs; **harder** debugging; **false** independence causes **race** bugs; **cost** may **rise** if every branch calls a **frontier** model.

## References

- **Agentic Design Patterns** — *Antonio Gulli*; agent definition: [parallelizer.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/parallelizer.md) (local clone: `subagents-design-patterns/agents/parallelizer.md`).
- **LangChain**: [LCEL — Parallelism](https://python.langchain.com/docs/how_to/parallel/) (`RunnableParallel`).
- **Pattern 12 (Deep Search)**, **Pattern 17 (LLM as Judge)**, **Pattern 23 (Multi-agent)**, **Pattern 33 (Prompt chaining)**, **Pattern 34 (Routing)**

## Related Patterns

- **Prompt chaining (33)**: **Ordered** steps; use when step *n+1* **needs** step *n*.
- **Routing (34)**: May **launch** one of several **parallel** topologies per request.
- **Multi-agent (23)**: **Crews** often mix **parallel** workers and **merge** nodes.
