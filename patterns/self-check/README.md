# Pattern 31: Self-Check

## Overview

**Self-check** uses **token-level probabilities** exposed by many inference APIs (**logprobs** / **logits**) to **estimate model confidence** in its own generations. **Low** probability on a span is a **signal** (not a proof) that the model is **uncertain** or **choosing among alternatives**—useful for **flagging** text for **human review**, **RAG re-query**, or **refusal**. It complements **ground-truth** checks (Pattern 30) and **citation**-based RAG (Pattern 11).

## Problem Statement

- LLMs can output **confident-sounding** but **false** or **ungrounded** content.
- Purely **semantic** post-hoc checks miss **calibration** at **generation time** when the API exposes **per-token** scores.

## Solution Overview

### Logits and logprobs

- **Logits** are **unnormalized** scores per vocabulary token before **softmax**: \(z_i\) for token \(i\).
- **Softmax** converts logits to **probabilities**: \(p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}\).
- **Log-probability** (natural log unless API says otherwise): \(\log p_i\). APIs often return **`logprob`** for the **sampled** token (and sometimes **top-k** alternatives with their logprobs).

**Why log domain?** Numerical stability: probabilities multiply along tokens; **log-probs** **add**.

### Using logprobs for self-check

- **Per-token probability** \(p = \exp(\text{logprob})\): very **low** \(p\) suggests the model **did not** strongly prefer that token given the prefix—**candidate** for review.
- **Top logprobs** (if available): a **small gap** between the best and second-best token means **ambiguity**.

**Limitation:** Low probability **does not** mean “wrong relative to the world”—only **relative to the model’s distribution**. A **wrong** fact can still be **high** probability if the model is confidently wrong. Combine with **retrieval**, **tools**, and **Pattern 30** for **facts**.

### Perplexity (sequence-level)

- **Average** **negative log-likelihood** per token: \(\text{NLL} = -\frac{1}{N}\sum_{i=1}^N \log p(w_i)\).
- **Perplexity** \(\text{PPL} = \exp(\text{NLL})\): “effective branching factor” at each step; **higher** → model is **less** confident on average over the sequence (often reported for **eval** sets, not per user response in production).

Book notebooks: `1_hallucination_detection.ipynb`, `2_self_check.ipynb` — parse **OpenAI**-style `logprobs`, **token** probabilities via `exp(logprob)`, **low-confidence** token lists, **response** confidence aggregates.

### High-level flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4f46e5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#4338ca', 'lineColor': '#6366f1', 'secondaryColor': '#10b981', 'tertiaryColor': '#f59e0b'}}}%%
flowchart LR
    G["Generate + logprobs"]
    A["Flag low-confidence"]
    R["Review / RAG / refuse"]
    G --> A
    A --> R
```

## Use Cases

- **Triage** answers in **medical** or **legal** assistants before display
- **Quality** scoring for **batch** generation; **A/B** prompts by **mean** token confidence

## Implementation Details

- Confirm **base** of log (`e` vs `2`) and whether values are **log p** or **logits** in your API.
- **Normalize** by tokenization: **subword** tokens may split words—aggregate to **span** scores if needed.
- **Never** log **PII** in hot paths (per project logging rules).

## Constraints & Tradeoffs

**Tradeoffs:** ✅ Cheap signal when logprobs exist. ⚠️ Not **ground truth**; not all providers expose **logprobs**; **cost** if requesting **top_logprobs** on every token.

## References

- Book: `generative-ai-design-patterns/examples/31_self_check/` (`1_hallucination_detection.ipynb`, `2_self_check.ipynb`)
- **Pattern 1 (Logits masking)**: intervenes **before** sampling; self-check **reads** **after** sampling
- **Pattern 11 (Trustworthy generation)**: citations and guardrails; self-check **augments** risk scoring

## Related Patterns

- **Assembled reformat (30)**: structural **facts**; self-check **scores** **model** confidence on **words**
- **Reflection (18)**: second-pass **text** critique; self-check can **trigger** reflection
