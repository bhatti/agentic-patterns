# Pattern 40: Resource-Aware Optimization

## Overview

**Resource-aware optimization** tunes **agent** and **LLM** systems to **cost**, **latency**, **capacity**, and **reliability** constraints—not just answer quality. It combines **monitoring**, **prediction**, **dynamic** **policies** (model **tier**, **tool** set, **context** **size**), and **graceful** **degradation** when pressure rises or budgets tighten.

This pattern is catalogued in **Agentic Design Patterns** (*Antonio Gulli*); companion agent spec: `subagents-design-patterns/agents/resource-optimizer.md` ([upstream](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/resource-optimizer.md)).

## Problem Statement

- **Frontier** models on **every** request **burn** **budget** and **SLOs**.
- **Latency**-sensitive paths (voice, live support) cannot wait for **long** **chains** or **huge** **contexts**.
- **Clusters** and **API** **quotas** need **awareness** of **parallelism**, **batching**, and **fallback** when **degraded**.

## Solution Overview

### Cost-aware LLM usage

- **Tiered** models: **small** / **fast** for **classification** and **routing**; **large** only for **synthesis** (aligns with **Pattern 34**).
- **Token** **budgets** per **session**, **tenant**, or **request**; **stop** or **summarize** when **exhausted**.
- **Pricing** **hooks**: estimate **cost** from **context** + **expected** **completion** **length** (and **cache** **hits**—**Pattern 25**).

### Latency-sensitive operations

- **Timeouts**, **streaming**, **first-token** **SLOs**; **shorter** **prompts** and **fewer** **round** **trips** for **interactive** **modes**.

### Reliability and fallback

- **Alternate** **models**, **cached** **answers**, **reduced** **tool** **sets**—overlaps **Pattern 37** (exception handling); here the **driver** is **resource** **pressure**, not only **errors**.

### Adaptive allocation

- **Route** **tasks** by **difficulty** **score** or **user** **tier**; **shift** **work** to **cheaper** **queues** under **load**.

### Dynamic model switching

- **Policies** **if** `p95_latency > X` **or** `daily_spend > Y` → **downgrade** **model** **or** **enable** **compression**.

### Adaptive tools

- **Disable** **expensive** **tools** (long **web** **fetch**, **heavy** **code** **run**) under **pressure**; **enable** **read-only** **shortcuts**.

### Contextual pruning and summarization

- **Truncate** **history** with **summarize**-then-**drop**; **retrieve** **only** **top-k** **chunks** (**Pattern 39**); **prompt** **compression** (**Pattern 26**).

### Prediction and planning

- **Forecast** **spend** and **queue** **depth** from **trailing** **metrics**; **autoscale** **workers** or **throttle** **ingress**.

### Parallel and distributed awareness

- **Balance** **fan-out** (**Pattern 35**) with **quota** **caps**; **avoid** **N** **frontier** **calls** when **k** **smaller** **calls** **suffice**.

### Graceful degradation

- **Shorter** **answers**, **fewer** **citations**, **template**-based **replies**, or **async** **“we’ll** **email** **you”** **flows**.

### Generative AI Design Patterns (Lakshmanan et al.)

**Inference optimization** (26), **prompt caching** (25), **small language model** (24), and **degradation testing** (27) are **close** **cousins**—this pattern is the **agentic** **orchestration** **layer** that **decides** **when** to **apply** them.

## Constraints & Tradeoffs

**Tradeoffs:** ✅ **Predictable** **cost** and **latency**. ⚠️ **Quality** **drops** when **degraded**; **mis-tuned** **thresholds** **anger** **users**; **prediction** **errors** **over**- or **under**-**provision**.

## References

- **Agentic Design Patterns** — *Antonio Gulli*; [resource-optimizer.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/resource-optimizer.md).
- **Pattern 24 (Small language model)**, **Pattern 25 (Prompt caching)**, **Pattern 26 (Inference optimization)**, **Pattern 27 (Degradation testing)**, **Pattern 34 (Routing)**, **Pattern 35 (Parallelization)**, **Pattern 37 (Exception handling and recovery)**

## Related Patterns

- **Routing (34)**: **Cost** / **capability** **tiers** **per** **request**.
- **Parallelization (35)**: **Trade** **wall-clock** for **aggregate** **quota** **use**.
- **Exception handling (37)**: **Fallback** when **errors** **or** **budgets** **block** **primary** **path**.
- **Prioritization (43)**: **Schedules** **what** **runs** **next** **under** **budget** **and** **load**
