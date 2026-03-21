# Pattern 39: Agentic RAG (Knowledge Retrieval)

## Overview

**Retrieval-Augmented Generation (RAG)** grounds answers in **external**, often **fresh** knowledge: **retrieve** relevant evidence, then **generate** with that **context**. This entry follows **Agentic Design Patterns** (*Antonio Gulli*); companion agent spec: `subagents-design-patterns/agents/rag-retriever.md` ([upstream](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/rag-retriever.md)).

**Relationship to this repo (Lakshmanan et al.)** — The same idea is already unpacked across **Pattern 6 (Basic RAG)** through **Pattern 12 (Deep Search)** and related retrieval patterns. **Pattern 39** does **not** replace those; it **consolidates** the **agentic** view: **embeddings**, **similarity**, **chunking**, **vector stores**, plus **Graph RAG** and **Agentic RAG** (iterative / tool-using retrieval), with **LangChain**-oriented composition notes.

## Problem Statement

- LLM **weights** are **stale** and **opaque**; organizations need **authoritative**, **up-to-date** **corpora**.
- Raw **prompt stuffing** of whole docs **blows** context limits and **dilutes** relevance.
- **Flat** vector search **misses** **relations** (people, APIs, regulations); **multi-hop** questions need **more** than one retrieval pass.

## Solution Overview

### Core two-stage loop

1. **Index**: load → **chunk** → **embed** (optional but standard) → **store** in a **vector database** or hybrid index.  
2. **Run**: **query** → **retrieve** (similarity search + rerank) → **integrate** chunks into the **prompt** → **generate** (with **citations** where required).

### Embeddings and similarity

- **Embeddings** map text (or images, etc.) to **dense vectors** so **semantic** nearness ≈ **geometric** nearness.
- **Cosine similarity** measures alignment of direction (common for normalized sentence embeddings). ** Euclidean** / **inner product** distance are alternatives depending on index and normalization.
- **Distance** and **similarity** are **linked** (e.g. cosine **distance** = 1 − cosine **similarity** for many setups)—pick **one** convention and **stick** to it in **evals**.

### Chunking

- Split documents by **headings**, **tokens**, or **characters**, with **overlap** to preserve **cross-boundary** context.
- Chunk **size** trades **precision** vs **context** **breadth**; **too small** loses meaning, **too large** adds noise.

### Vector databases

- Store **vectors** + **metadata** (source, ACL, time); support **ANN** (approximate nearest neighbor) **search** at scale.
- Often combined with **BM25** / **keyword** retrieval (**hybrid** retrieval) for **lexical** matches embeddings miss.

### Graph RAG

- Represent **entities** and **relations** (knowledge **graph**); **retrieve** by **traversal**, **community** summaries, or **graph**-augmented prompts—not only vector **k**-NN.
- Useful when answers require **explicit** **relationships** (org charts, **regulatory** **graphs**, **API** **dependency** graphs).

### Agentic RAG

- The **model** or **orchestrator** **decides** **when** and **what** to retrieve (**multi-step**): rewrite **query**, **call** **tools** (Pattern 21), **branch** (Pattern 34), **parallel** fan-out (Pattern 35), **reflect** on gaps (Pattern 18). **LangGraph** fits **cycles**; **LCEL** fits **linear** **retrieve → format → LLM** chains.

### LangChain (examples)

- **LCEL**: `RunnablePassthrough.assign(context=retriever) | prompt | llm` — see `example.py` (`build_lcel_rag_stub`).  
- **LangGraph**: **nodes** for **retrieve**, **grade** **documents**, **rewrite** **query**, **generate**—**conditional** edges for **insufficient** context.  
- **VectorStore** integrations: `similarity_search`, `as_retriever()`, optional **compression** / **rerank** steps.

## Constraints & Tradeoffs

**Tradeoffs:** ✅ **Fresh**, **grounded** answers when sources are good. ⚠️ **Chunking** and **embedding** **drift**; **vector-only** **blind** **spots**; **Graph** **ops** **cost**; **agentic** loops add **latency** and **tokens**.

## References

- **Agentic Design Patterns** — *Antonio Gulli*; [rag-retriever.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/rag-retriever.md).
- [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangGraph RAG examples](https://langchain-ai.github.io/langgraph/tutorials/rag/)
- **Pattern 6 (Basic RAG)** … **Pattern 12 (Deep Search)** — detailed **Lakshmanan**-aligned implementations in this repo.

## Related Patterns

- **Basic RAG (6)**, **Semantic indexing (7)**, **Indexing at scale (8)**, **Index-aware retrieval (9)**, **Node postprocessing (10)**, **Deep Search (12)** — **corpus** and **retrieval** depth.
- **Tool calling (21)**: **Agentic** RAG often **exposes** retrieval as a **tool**.
- **Routing (34)** & **Prompt chaining (33)**: **Orchestrate** **multi-step** retrieval.
