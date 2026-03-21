# Pattern 41: Reasoning Techniques (Agentic)

## Overview

**Reasoning techniques** improve answers on **complex** **Q&A** and **planning** by structuring **inference**: explicit **steps**, **multiple** **hypotheses**, **tools**, **code**, **debate**, or **iterative** **research**. This entry follows **Agentic Design Patterns** (*Antonio Gulli*); companion agent spec: `subagents-design-patterns/agents/reasoning-engine.md` ([upstream](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/reasoning-engine.md)).

**Relationship to this repo** — Many techniques already have **dedicated** **patterns** below. **Pattern 41** is a **map**: **when** to use **which** **family**, and how **advanced** **topics** (e.g. **debates**, **PAL**, **RLVR**, **MASS**) **compose** with **orchestration**.

## Problem Statement

- A **single** **flat** **completion** **fails** on **multi-step** **math**, **planning**, **tool** **use**, or **conflicting** **evidence**.
- Teams need a **shared** **vocabulary** (CoT, ToT, **LATS**, ReAct, **deep** **research**) without **reimplementing** **each** **variant** **from** **scratch**.

## Technique map (concise)

| Technique | Role | Where in this repo |
|-----------|------|-------------------|
| **Chain-of-Thought (CoT)** | Intermediate **reasoning** **steps** in **one** (or **few**) **generations** | **Pattern 13** |
| **Tree of Thoughts (ToT)** | **Search** **over** **multiple** **reasoning** **paths** | **Pattern 14** |
| **Language Agent Tree Search (LATS)** | **Tree** **search** **over** **language**-**modeled** **states** **or** **actions** (expand **nodes**, **evaluate** **candidates**, **prune** **beam** / **MCTS**-**style** **policy**)—often **stronger** **than** **fixed**-**breadth** ToT when **depth** **and** **backtracking** **matter** | Same **family** **as** **Pattern** **14**; **implementation** **is** **graph** **search** **+** **LLM** **as** **generator**/**critic** **(see** **example.py** **stub)** |
| **Self-correction / reflection** | **Critique** → **revise** (often **two+** **calls**) | **Pattern 18** |
| **Program-Aided Language (PAL)** | Model **emits** **executable** **code**; **runtime** **returns** **exact** **results** (math, APIs) | **Pattern 22** (Code execution); pair with **21** for **tool** **orchestration** |
| **ReAct** | Interleaved **Thought** → **Action** (tool) → **Observation** **loops** | **Pattern 21** (Tool calling / agents) |
| **RLVR** (*Reinforcement* **Learning** with **Verifiable** **Rewards*) | **Train** **policies** **using** **checkable** **outcomes** (tests, compilers, theorem checkers)—**orthogonal** **runtime** **pattern**; see **Pattern 36** for **learning** **framing** | Learning stack + **eval** **harnesses** |
| **Chain of Debates (CoD)** | **Multiple** **roles** **argue** **pro/con** **before** **consensus**—**multi-agent** **pattern** | **Pattern 23** (Multi-agent collaboration) |
| **MASS** (and similar **workflow** **optimizers**) | **Optimize** **prompts** **block**-**wise**, **workflow** **topology** (order/branches), or **end**-**to**-**end** **workflow** **objectives**—often **offline** **or** **bandit** **loops** | **Pattern 20** (Prompt optimization); **LangGraph** **graph** **edits** **experimentally** |
| **Deep research** | **Iterative** **retrieval** + **synthesis** **over** **many** **sources** | **Pattern 12** (Deep Search) |

## Use cases

- **Complex** **Q&A** (multi-hop, **numeric**, **policy** **interpretation**)
- **Agents** that **must** **plan**, **use** **tools**, or **verify** **externally**
- **Research** **assistants** with **citations** and **broad** **corpora**

## Constraints & Tradeoffs

**Tradeoffs:** ✅ **Transparency** and **controllability** when **steps** are **explicit**. ⚠️ **Latency** **and** **cost** **scale** with **depth**; **debates** **and** **ToT** **multiply** **tokens**; **PAL** **needs** a **safe** **sandbox**.

## References

- **Agentic Design Patterns** — *Antonio Gulli*; [reasoning-engine.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/reasoning-engine.md).
- **Pattern 12 (Deep Search)**, **13 (CoT)**, **14 (ToT)**, **18 (Reflection)**, **20 (Prompt optimization)**, **21 (Tool calling / ReAct)**, **22 (Code execution / PAL)**, **23 (Multi-agent)**, **36 (Learning and adaptation)**

## Related Patterns

- **Prompt chaining (33)**: **Sequences** **reasoning** **stages** **with** **handoffs**—often **hosts** CoT/ReAct **subgraphs**.
- **Parallelization (35)**: **Parallel** **ToT** **branches** **or** **debate** **rounds** **with** **merge** **policies**.
- **Memory management (44)**: **Long** **reasoning** **traces** **benefit** **from** **tiered** **context** **(working** **+** **summaries)**
