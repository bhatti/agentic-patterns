# Pattern 44: Memory Management (Agentic)

## Overview

**Memory management** gives agents **continuity** and **personalization** without stuffing full chat logs into every prompt: **short-term** / **working** **context**, **long-term** **persistent** **stores**, **episodic** **retrieval** (“what we did when”), **procedural** **memory** (how-to, playbooks), and often **semantic** **facts** (stable user/world knowledge). It pairs **selective** **write**, **relevance-ranked** **read**, and **privacy** **controls**.

Companion spec (*Agentic Design Patterns*, *Antonio Gulli*): `subagents-design-patterns/agents/memory-manager.md` ([upstream](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/memory-manager.md)).

## Relationship to Pattern 28

**Pattern 28 (Long-term memory)** in this repo is the **Lakshmanan**-aligned **deep** **implementation** (four memory types, Mem0, extraction). **Pattern 44** is the **Gulli** **agentic** **framing** and **LangGraph** **checkpoint** **pattern**—**cross-use** them; **do** **not** **duplicate** **all** **of** **28** **here**.

## Memory types (quick map)

| Type | Role | Typical mechanism |
|------|------|-------------------|
| **Short-term / working** | **Current** **turn** **focus**, **scratch** **space** | **Sliding** **window**, **summarize**-**then**-**truncate**, **token** **budget** |
| **Contextual memory** | **Session** **or** **task** **scoped** **facts** **injected** **into** **system**/**user** **prefix** | **Per**-**thread** **state** **in** **orchestrator** |
| **Long-term persistent** | **Durable** **across** **sessions** | **DB**, **KV**, **vector** **store** **+** **extractors** |
| **Episodic** | **Time**-**ordered** **events**, **“last** **time…”** | **Append**-**only** **logs**, **timestamps**, **retrieve** **by** **recency** **/** **similarity** |
| **Procedural** | **Skills**, **checklists**, **tool** **recipes** | **Versioned** **docs**, **structured** **steps**, **few**-**shot** **slots** |
| **Semantic** | **Stable** **facts** **(preferences**, **entities)** | **Embedding** **search**, **graph** **+** **RAG** **(Pattern** **39**/**6**) |

## LangGraph: threads and checkpoints

- **`MemorySaver`** (or **Redis**/**Postgres** **savers** **in** **production**) **persists** **graph** **state** **per** **`thread_id`** so **multi**-**turn** **agents** **resume** **without** **you** **manually** **re**-**passing** **full** **history**.
- **State** **schema** **often** **uses** **reducers** **(e.g.** **`operator.add`** **for** **counters**, **`add_messages`** **for** **chat** **lists)** **so** **each** **node** **returns** **deltas**.
- **Combine** **with** **external** **vector** **memory** **(Mem0**, **custom)** **for** **semantic**/**episodic** **retrieval** **nodes** **that** **inject** **facts** **into** **state** **before** **the** **LLM**.

See `example.py`: `build_langgraph_checkpoint_demo()` (optional `langgraph` install).

## Use cases

- **Assistants** **that** **remember** **preferences** **and** **past** **decisions**
- **Support** **bots** **with** **ticket**-**scoped** **context** **+** **account** **history**
- **Agents** **with** **durable** **playbooks** **(procedural)** **and** **audit** **trails** **(episodic)**

## Constraints and tradeoffs

**Tradeoffs:** ✅ **Better** **UX** **and** **fewer** **repeated** **questions**. ⚠️ **PII** **retention**, **staleness**, **retrieval** **noise**; **checkpoint** **size** **and** **migration** **costs**.

## References

- **Agentic Design Patterns** — *Antonio Gulli*; [memory-manager.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/memory-manager.md).
- [LangGraph persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- **Pattern 28 (Long-term memory)**, **Pattern 39 (Agentic RAG)**

## Related patterns

- **Long-term memory (28)**: **Four**-**type** **design** **and** **Mem0**-**style** **pipelines**
- **Agentic RAG (39)**: **Retrieval** **into** **context** **as** **semantic** **memory**
- **Prompt chaining (33)**: **Explicit** **handoffs** **can** **simulate** **short**-**term** **pipelines**
