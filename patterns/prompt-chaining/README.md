# Pattern 33: Prompt Chaining

## Overview

**Prompt chaining** (also **prompt pipeline**) is **sequential decomposition**: the **output** of step *n* becomes **structured input** to step *n+1*—often the **same** model with **different** system prompts or **tools**, or **different** models per step. It improves **reliability** on complex tasks by giving each call a **single purpose**, enforcing **structured** intermediate representations, and preserving **context** explicitly between stages.

This pattern is catalogued in **Agentic Design Patterns** (*Antonio Gulli*); companion agent spec: `subagents-design-patterns/agents/prompt-chainer.md`. It complements—but is **not** the same as—**Chain-of-Thought** (reasoning *inside* one completion, Pattern 13) or **multi-agent** collaboration (multiple **roles** and tools, Pattern 23).

## Problem Statement

- One long “do everything” prompt yields **vague** or **inconsistent** results.
- **Complex** queries (research → outline → answer), **code** (spec → implement → fix), and **workflows** need **stages** with **verifiable** handoffs.

## Solution Overview

1. **Decompose** the user goal into ordered steps (plan may itself be a first chain step).
2. **Structured output** at each step (JSON, bullet list, typed objects) so the next prompt **parses** reliably.
3. **Context engineering**: pass only **necessary** fields forward; avoid dumping full history every time.
4. **System prompt per step**: narrow **role** and **constraints** per link in the chain.
5. **Tools** at defined steps only (retrieve → then summarize → then cite).
6. **Error handling**: retry or branch when a step fails validation.

**Relation to other patterns**

| Pattern | Distinction |
|---------|-------------|
| **Chain of Thought (13)** | Single response with **explicit** reasoning steps—not necessarily **multiple** API calls. |
| **Multi-agent (23)** | Often **multiple** prompts/systems/**specialists**; chaining can be **one** agent **sequential** or a **subgraph** of a larger crew. |
| **Reflection (18)** | **Two** or **more** passes (generate → critique → refine → iterate); *Gulli* reflector; **LCEL** / LangGraph; chaining generalizes to longer **linear** pipelines. |
| **Template generation (29)** | May **chain** “draft template” → “legal review slot names” → “store”. |

### Generative AI Design Patterns (Lakshmanan et al.)

Earlier chapters in *Generative AI Design Patterns* cover **RAG**, **tool calling**, **multi-agent**, and **safeguards**—chaining is how you **compose** those pieces into **pipelines** (e.g. retrieve → verify → generate).

## Implementation details

- Use **LangGraph**, **LangChain LCEL**, or plain functions with a **TypedDict** / Pydantic state object.
- **Validate** each step’s output (schema, length, banned phrases) before the next.
- **Log** step boundaries for **debugging** (avoid PII in hot paths per project rules).

## Constraints & Tradeoffs

**Tradeoffs:** ✅ Clarity and easier testing per step. ⚠️ **Latency** and **cost** scale with steps; **fragile** if intermediate format drifts.

## References

- **Agentic Design Patterns** — *Antonio Gulli*; agent definition: [prompt-chainer.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/prompt-chainer.md) (local clone: `subagents-design-patterns/agents/prompt-chainer.md`).
- **Generative AI Design Patterns** — *Valliappa Lakshmanan* et al. (O’Reilly): composing retrieval, tools, and agents in **pipelines**.
- **Pattern 13 (CoT)**, **Pattern 23 (Multi-agent)**, **Pattern 21 (Tool calling)**

## Related Patterns

- **Dependency injection (19)**: inject step callables for tests.
- **Guardrails (32)**: run **input/output** scans **between** chain links.
- **Routing (34)**: often **selects** which **chain** or **entry** node runs for this request.
- **Parallelization (35)**: **same** input to **multiple** independent steps **at once**; **chain** when outputs must feed **sequentially**.
- **Exception handling and recovery (37)**: **Per-step** errors, **retries**, **breakers**, **fallbacks** without breaking the whole pipeline.
- **Planning and task decomposition (45)**: **Explicit** **DAG** **of** **tasks** **before** **or** **while** **chaining** **steps**
