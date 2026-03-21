# Pattern 38: Human-in-the-Loop (HITL)

## Overview

**Human-in-the-loop (HITL)** inserts **people** into automated flows where **stakes**, **uncertainty**, or **policy** require **oversight**: the system **proposes** or **pre-classifies**, then **escalates** to a reviewer **based on rules** (confidence thresholds, risk tier, novelty), **structured** review UIs, and **feedback** back into models or playbooks—not ad-hoc “email someone.”

This pattern is catalogued in **Agentic Design Patterns** (*Antonio Gulli*); companion agent spec: `subagents-design-patterns/agents/human-validator.md` ([upstream](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/human-validator.md)).

## Problem Statement

- **Full automation** is unacceptable when errors cause **harm**, **legal** exposure, or **large** financial loss.
- **Model confidence** and **business risk** are **orthogonal**: high logprob ≠ safe to ship.
- You need **clear** **triggers**, **queues**, **SLAs** for reviewers, and **audit** trails—not silent automation.

## Solution Overview

1. **Intervention policy**: when to stop automation—e.g. confidence **below** τ, **high-stakes** domain flag, **first** occurrence of a template, **random** audit sample, **user** complaint channel.
2. **Review surface**: ticket **queue**, **diff** UI (before/after), **rubric** (approve / edit / reject), **time-bounded** escalation.
3. **Feedback loop**: human **labels** and **edits** feed **training** (Pattern 36), **prompt** updates, or **routing** rules (34).
4. **Quality & safety**: **multi-level** review for critical paths (e.g. L1 auto → L2 human → L3 specialist).
5. **Composability**: HITL **wraps** **guardrails** (32); it is not a substitute for **input/output** policy checks.

### Generative AI Design Patterns (Lakshmanan et al.)

**Template generation** (29) often uses **human-approved** templates; **LLM-as-judge** (17) can **triage** what reaches humans. **Trustworthy generation** (11) and **assembled reformat** (30) align with **verify-before-publish** workflows.

## Use Cases

- **Content moderation**: auto scores + **human** appeal and **edge** cases.
- **Autonomous systems** (e.g. driving, robotics): **disengagement**, **OOD** scenes, **geofenced** handoff—illustrative; production stacks add **formal** safety cases.
- **Fraud** and **AML**: model **flags** → **analyst** **investigation** → **feedback** to rules and models.
- **Trading** and **execution**: **pre-trade** **limits**, **human** **approval** above **notional** or **volatility** thresholds.
- **Healthcare** / **legal** assist: **draft** only until **professional** sign-off.

## Constraints & Tradeoffs

**Tradeoffs:** ✅ **Accountability** and **trust** in high-stakes domains. ⚠️ **Latency** and **cost** of staffing; **consistency** across reviewers; **tooling** debt for queues and audit logs.

## References

- **Agentic Design Patterns** — *Antonio Gulli*; [human-validator.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/human-validator.md).
- **Pattern 17 (LLM as Judge)**, **Pattern 29 (Template generation)**, **Pattern 32 (Guardrails)**, **Pattern 37 (Exception handling and recovery)**

## Related Patterns

- **LLM as Judge (17)**: Automated **scoring**; can **feed** HITL **priority** queues.
- **Template generation (29)**: Humans **approve** **templates** before **runtime** fill.
- **Guardrails (32)**: Automated **blocks**; HITL for **gray** zones.
- **Exception handling and recovery (37)**: Escalation paths when **automation** cannot proceed.
