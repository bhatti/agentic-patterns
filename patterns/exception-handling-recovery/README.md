# Pattern 37: Exception Handling and Recovery

## Overview

**Exception handling and recovery** makes **agentic** and **LLM** pipelines **resilient**: **detect** failures (timeouts, rate limits, tool errors, schema violations, policy blocks), **classify** them (transient vs permanent, retriable vs fatal), **handle** them with **retries**, **circuit breakers**, and **fallbacks**, and **recover** without corrupting **state**—including **graceful degradation** (cached answer, smaller model, human queue) when full success is impossible.

This pattern is catalogued in **Agentic Design Patterns** (*Antonio Gulli*); companion agent spec: `subagents-design-patterns/agents/exception-handler.md` ([upstream](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/exception-handler.md)).

## Problem Statement

- **Tool** and **model** calls **fail** unpredictably (network, quota, content filter, malformed JSON).
- Uncaught errors **bubble** to users as 500s or **silent** wrong answers.
- Blind **retries** amplify outages; **no** fallback leaves **no** partial value.
- **Distributed** steps (chains, agents) need **consistent** compensation or **idempotent** retries.

## Solution Overview

1. **Error detection**: structured outcomes (`Result` / `Either`), **timeouts**, **validation** of LLM output (schema, length), **guardrail** signals (Pattern 32), **HTTP** status mapping.
2. **Classification**: **transient** (retry with backoff), **permanent** (fail fast or alternate path), **policy** (user message, no retry), **capacity** (circuit open).
3. **Handling**:
   - **Retry** with **exponential backoff** and **jitter**; cap attempts.
   - **Circuit breaker**: after repeated failures, **trip** open; **half-open** probes before closing.
   - **Fallback**: secondary model, **cached** response, **degraded** tool, static apology template.
4. **Recovery**: preserve or **roll back** durable **state**; **compensating** actions for side effects; **idempotency** keys for retried writes.
5. **Learning**: aggregate error **codes** for dashboards; **not** raw PII in hot-path logs (project norms).

### Generative AI Design Patterns (Lakshmanan et al.)

**Guardrails** (32), **self-check** (31), and **dependency injection** (19) support **detection** and **testability**; this pattern is the **operational** shell around those signals.

## Constraints & Tradeoffs

**Tradeoffs:** ✅ Fewer user-visible failures; clearer **SLAs**. ⚠️ **Retries** increase **latency**; **fallbacks** can **diverge** in quality; **circuit** tuning needs **metrics**.

## References

- **Agentic Design Patterns** — *Antonio Gulli*; [exception-handler.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/exception-handler.md).
- **Pattern 19 (Dependency injection)**, **Pattern 21 (Tool calling)**, **Pattern 32 (Guardrails)**, **Pattern 33 (Prompt chaining)**

## Related Patterns

- **Prompt chaining (33)**: **Per-step** try/catch, **partial** replay from last good state.
- **Tool calling (21)**: **ToolNode** errors → **retry** / **alternate** tool / user-visible **message**.
- **Guardrails (32)**: Policy **violations** are a **class** of errors with **non-retry** handling.
- **Routing (34)**: **Failover** to another **handler** or model **tier**.
- **Human-in-the-loop (38)**: **Escalation** when automation cannot safely decide (complements **retry**/**fallback**).
