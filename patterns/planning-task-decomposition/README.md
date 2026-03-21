# Pattern 45: Planning and Task Decomposition (Agentic)

## Overview

**Planning** turns a **high-level objective** into a **DAG** (or tree) of **tasks** with **dependencies**, **milestones**, and **replanning** when the world changes. It sits **above** **prompt chaining** (33): you **know** **structure** **before** **or** **while** **executing**, not only **linear** **steps**.

**Gulli** (*Agentic Design Patterns*): [planner.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/planner.md).

## Relationship to other patterns

- **Prompt chaining (33)**: Often **instantiates** **one** **branch** **of** **a** **plan**.
- **Prioritization (43)**: **Orders** **ready** **tasks**; **planning** **decides** **what** **exists** **and** **depends** **on** **what**.
- **Multi-agent (23)**: **Plans** **can** **assign** **subtasks** **to** **roles**.

## References

- **Pattern 33**, **Pattern 43**, **Pattern 23**
