# Pattern 34: Routing

## Overview

**Routing** sends each **request** to the right **handler** (tool, subgraph, model, queue, or human)—before or instead of a single generic completion. Implementations differ by **signal**: **rules** (fast, explicit), **embeddings** (semantic similarity to labeled exemplars or handler descriptions), **LLM classification** (flexible natural-language policies), or **trained classifiers** (latency/cost tradeoffs, reproducible behavior).

This pattern is catalogued in **Agentic Design Patterns** (*Antonio Gulli*); companion agent spec: `subagents-design-patterns/agents/router.md`. It pairs naturally with **prompt chaining** (Pattern 33), **tool calling** (21), and **multi-agent** collaboration (23): route **first**, then execute the specialized path.

## Problem Statement

- One model or one prompt for **all** intents yields **weak** answers and **unsafe** tool use.
- **Traffic** mixes **billing**, **technical**, **legal**, and **small talk**—each needs different **tools**, **policies**, and **models**.
- You need **predictable** behavior where possible (**rules** / **ML**) and **graceful** handling of **novel** phrasing (**LLM** / **embeddings**).

## Solution Overview

1. **Define handlers** (named targets with schemas, prompts, or subgraph entrypoints).
2. **Choose a router family** (often **combine** them):
   - **Rule-based**: regex, keywords, headers, tenant flags, content type—**O(1)** path, auditable.
   - **Embedding-based**: embed the query (and/or session); **nearest** handler description or **centroid** of exemplars; good for **paraphrase** without an LLM on the hot path.
   - **LLM-based**: structured output (`{ "route": "...", "confidence": 0.0–1.0 }`); **fallback** when rules miss; tune with **few-shot** or **policy** text.
   - **ML model**: classical or small **classifier** on features (embeddings, TF-IDF, metadata); **retrain** on labeled traffic; lowest **inference** cost at scale.
3. **Confidence and fallback**: unknown → **default** handler, **human** queue, or **second-stage** router (e.g. rules → LLM).
4. **Observability**: log **route id** and **reason** (not raw PII in hot paths per project norms).

### Comparison (high level)

| Mechanism | Strengths | Watchouts |
|-----------|-----------|-----------|
| **Rules** | Deterministic, cheap, compliance-friendly | Brittle synonyms; maintenance |
| **Embeddings** | Semantic generalization; no label per utterance if descriptions suffice | Drift; needs good **prototypes** |
| **LLM** | Handles nuance and new intents | Cost/latency; structure with **JSON** / tool schema |
| **ML classifier** | Fast, batch-trainable, SLA-friendly | Cold start; **shift**; explainability |

### Generative AI Design Patterns (Lakshmanan et al.)

**Adapter tuning** (Pattern 15) and **prompt optimization** (Pattern 20) can improve **intent** or **route** quality. **RAG** (6–10) often sits **behind** a route (“docs” vs “SQL” vs “escalation”).

## Implementation details

- **LangChain / LangGraph**: use **`RunnableBranch`**, **conditional edges**, or **supervisor** nodes that emit a **route key**; keep routing **side-effect free** until the chosen branch runs.
- **Multi-stage**: cheap **rules** first, then **embedding** tie-break, then **LLM** only if confidence low.
- **Safety**: route **PII**-heavy traffic to **redaction** pipelines; **blocked** intents to a **refusal** handler.

## Constraints & Tradeoffs

**Tradeoffs:** ✅ Specialization, safer tool use, clearer SLOs per path. ⚠️ **Wrong route** failures; **operational** burden tuning thresholds; **inconsistency** if multiple router layers disagree.

## References

- **Agentic Design Patterns** — *Antonio Gulli*; agent definition: [router.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/router.md) (local clone: `subagents-design-patterns/agents/router.md`).
- **Generative AI Design Patterns** — *Valliappa Lakshmanan* et al.: adapters, evaluation, and composed pipelines.
- **LangChain**: [RunnableBranch](https://python.langchain.com/docs/how_to/branching/) (branching runnables).
- **Pattern 15 (Adapter tuning)**, **Pattern 21 (Tool calling)**, **Pattern 23 (Multi-agent)**, **Pattern 33 (Prompt chaining)**

## Related Patterns

- **Prompt chaining (33)**: Route **selects** which chain or **entry** node runs.
- **Multi-agent (23)**: Route **picks** a **specialist** or **crew** topology.
- **Tool calling (21)**: Route **gates** which tools are **in** context for this turn.
- **Parallelization (35)**: After routing, **fan out** **independent** handler work **concurrently**, then **aggregate**.
- **Prioritization (43)**: **Orders** **work** **within** **or** **across** **queues** **once** **intent** **is** **known**
