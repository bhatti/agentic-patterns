# Pattern 43: Prioritization (Agentic)

## Overview

**Prioritization** ranks **competing** **tasks** or **requests** under **limited** **resources**—using **multiple** **criteria** (urgency, business impact, **SLA** **risk**, **effort**, **dependencies**, **security** **severity**) and **recomputing** **order** as **conditions** **change**. In agentic systems it feeds **queues**, **routers**, and **schedulers** so the **right** **work** runs **first**.

Companion agent spec (*Agentic Design Patterns*, *Antonio Gulli*): `subagents-design-patterns/agents/prioritizer.md` ([upstream](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/prioritizer.md)).

## Problem statement

- **FIFO** **starves** **critical** **work**; **single** **metric** **(e.g.** **age)** **misses** **risk** **and** **value**.
- **Cloud** **and** **trading** **environments** **need** **dynamic** **rebalancing**; **support** **and** **security** **need** **severity** **and** **customer** **tier** **awareness**.

## Solution outline

1. **Dimensions**: capture **urgency**, **importance**/**impact**, **effort**/**cost**, **deadline**/**SLA**, **dependencies**, **compliance**/**security** **class**.
2. **Scoring**: **weighted** **sum**, **Eisenhower**-style **matrices**, or **domain** **rules** **(e.g.** **SEV** **levels)**.
3. **Dynamic** **updates**: **re-rank** **on** **new** **events** **(incident** **escalation**, **spot** **price**, **queue** **depth)**.
4. **Resource** **awareness**: tie to **Pattern** **40** **(budgets**, **capacity)**; **avoid** **starvation** **with** **aging** **or** **fairness** **constraints**.
5. **Integration**: **routing** **(34)** **selects** **path**; **prioritization** **orders** **work** **within** **paths** **or** **global** **queues**.

## Use cases

- **Customer** **support**: **tier**, **sentiment**, **outage** **correlation**, **SLA** **clocks**
- **Cloud** **computing**: **job** **scheduling**, **preemption**, **cost** **vs** **deadline**
- **Trading**: **latency**-**sensitive** **orders**, **risk** **limits**, **market** **hours**
- **Security**: **incident** **severity**, **exploit** **in** **the** **wild**, **patch** **windows**

## Constraints and tradeoffs

**Tradeoffs:** ✅ **Alignment** **with** **goals** **and** **risk**. ⚠️ **Gaming** **metrics**; **stale** **scores** **if** **not** **refreshed**; **fairness** **vs** **VIP** **policies**.

## References

- **Agentic Design Patterns** — *Antonio Gulli*; [prioritizer.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/prioritizer.md).

## Related patterns

- **Routing (34)**: **Where** **to** **send** **work**; **prioritization** **orders** **what** **runs** **next**.
- **Resource-aware optimization (40)**: **Capacity** **and** **cost** **constraints** **on** **scheduling**.
- **Human-in-the-loop (38)**: **Escalation** **queues** **often** **priority** **sorted**.
- **Evaluation and monitoring (42)**: **Measure** **whether** **priority** **rules** **hit** **SLOs**.
