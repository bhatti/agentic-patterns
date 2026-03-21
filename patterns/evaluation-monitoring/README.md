# Pattern 42: Evaluation and Monitoring (Agentic)

## Overview

**Evaluation and monitoring** makes agentic and LLM systems **observable** and **accountable**: **latency** and **throughput**, **token** and **cost** **telemetry**, **quality** **scores** (including **LLM-as-judge**), **A/B** **experiments**, **compliance** **evidence**, and **multi-agent** **traces**. It is the **control plane** that feeds **reliability**, **product** **decisions**, and **audits**.

This pattern follows **Agentic Design Patterns** (*Antonio Gulli*); companion agent spec: `subagents-design-patterns/agents/evaluator.md` ([upstream](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/evaluator.md)).

## Relationship to this repo

- **LLM as Judge (17)** — **Automated** **quality** **metrics** **and** **rubrics** **on** **outputs**.
- **Degradation testing (27)** — **Load**/**stepped** **concurrency** **and** **LLM-native** **SLOs**.
- **Prompt optimization (20)** — **Evaluation** **harnesses** **drive** **prompt** **search**.
- **Multi-agent collaboration (23)** — **Per-role** **spans**, **handoff** **metrics**, **end-to-end** **traces**.
- **Resource-aware optimization (40)** — **Consumes** **usage** **and** **latency** **signals** **for** **policies**.

Pattern **42** **coordinates** **these**; it does **not** **replace** **deep** **implementations** **elsewhere**.

## Problem statement

- **Production** **GenAI** **without** **metrics** **is** **un-debuggable** **and** **non-compliant**.
- **Teams** **need** **shared** **dashboards**: **p95** **latency**, **$/request**, **judge** **scores**, **experiment** **lift**, **audit** **logs**.

## Solution outline

1. **Instrument** **every** **LLM** **and** **tool** **call** **with** **trace** **ids**, **timestamps**, **token** **counts**, **model** **id**, **tenant** **id** (avoid **PII** **in** **metric** **labels** **per** **policy**).
2. **Latency** **SLOs** **(TTFT**, **E2E**, **per-step)** **with** **alerts** **on** **regression**.
3. **Cost** **telemetry** **from** **provider** **bills** **or** **estimated** **tokens** **×** **price** **tables**.
4. **Quality** **via** **LLM-as-judge** **(17)**, **human** **labels**, **and** **task-specific** **checks** **(schema**, **regex**, **golden** **sets)**.
5. **A/B** **testing**: **stable** **assignment** **(hash** **user→variant)**, **guardrail** **metrics** **(safety**, **latency)**, **duration** **and** **power** **awareness**.
6. **Compliance**: **immutable** **audit** **trails**, **retention** **policies**, **evidence** **packs** **for** **reviews**.
7. **Multi-agent**: **span** **per** **agent**/**node**, **merge** **points**, **failure** **injection** **tests**.

## Use cases

- **Performance** **tracking** **and** **SRE** **dashboards**
- **A/B** **testing** **prompts**, **models**, **routing** **policies**
- **Compliance** **(finance**, **health**, **EU** **AI** **Act** **readiness)**
- **Enterprise** **governance**: **cost** **centers**, **chargeback**, **SLAs**

## Constraints and tradeoffs

**Tradeoffs:** ✅ **Visibility** **and** **continuous** **improvement**. ⚠️ **Judge** **cost** **and** **bias**; **metric** **overload**; **sampling** **vs** **complete** **coverage**.

## References

- **Agentic Design Patterns** — *Antonio Gulli*; [evaluator.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/evaluator.md).
- OpenTelemetry, LangSmith, Phoenix, vendor **observability** **guides** (integrate per stack).

## Related patterns

- **LLM as Judge (17)**, **Degradation testing (27)**, **Prompt optimization (20)**, **Multi-agent (23)**, **Resource-aware optimization (40)**
