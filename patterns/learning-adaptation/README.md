# Pattern 36: Learning and Adaptation

## Overview

**Learning and adaptation** covers how **agents** and **models** improve from **data** and **feedback**: **supervised** learning (labeled inputs → targets), **unsupervised** structure discovery, **reinforcement** learning (actions → rewards in an environment), **few-shot** / **zero-shot** inference (in-context without weight updates), **online** updates as new data arrives, and **memory-based** learning (retrieve similar past cases to steer behavior—ties to **Pattern 28**).

This pattern is catalogued in **Agentic Design Patterns** (*Antonio Gulli*); companion agent spec: `subagents-design-patterns/agents/learning-adapter.md`. It complements **adapter tuning** (Pattern 15) and **Evol-Instruct** (16): those focus on **efficient** supervised fine-tuning or **instruction** evolution; this pattern foregrounds **RL**, **preference** alignment, and **continuous** adaptation in **deployed** systems.

## Problem Statement

- **Static** prompts and frozen weights **drift** from user needs, fraud patterns, or market regimes.
- **Agent** policies (tool use, trading, robotics) need **updates** from **outcomes**, not only offline batches.
- Teams must choose among **PPO-style** RL pipelines, **preference**-based alignment (**DPO**), **incremental** fine-tunes, and **memory**—each with different **data**, **risk**, and **ops** requirements.

## Solution Overview

### Taxonomy (where each fits)

| Mode | Signal | Typical use |
|------|--------|-------------|
| **Supervised** | Input–output pairs | Classification, adapters, imitation |
| **Unsupervised** | Unlabeled data | Clustering, representation learning, pretraining objectives |
| **Reinforcement learning** | Rewards / returns | Agents in **environments** (games, sims, trading) |
| **Few-shot / zero-shot** | Examples in context (no grad step) | Rapid task adaptation at **inference** |
| **Online learning** | Stream of labeled or bandit feedback | Drift, A/B rewards, incremental updates |
| **Memory-based** | Retrieved episodes / facts | Case-based steering, RAG-style adaptation |

### Proximal Policy Optimization (PPO)

**PPO** is a widely used **policy-gradient** algorithm for training **stochastic** policies (including **LLM** policies acting in discrete token spaces when framed as RL). Core ideas:

1. **Collect trajectories** with the **current** policy (often in parallel workers).
2. **Evaluate** a **surrogate** objective: approximate improvement using **importance sampling** ratios \(r_t(\theta)=\pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)\).
3. **Clip** the ratio to \([1-\varepsilon, 1+\varepsilon]\) so updates stay in a **trust region**—large policy jumps are penalized, improving **stability** vs vanilla policy gradients.
4. **Iterate**: new data under the updated policy → repeat (**continual** improvement when the environment keeps producing new experience).

For **LLM alignment**, a common **two-stage** pipeline is: train a **reward model** from human rankings, then **fine-tune** the language model with **PPO** against that reward (**RLHF** family). That separates **preference modeling** from **policy optimization**.

### Direct Preference Optimization (DPO)

**DPO** aligns the policy **directly** from **preference** pairs \((y_w \succ y_l \mid x)\) without an **explicit reward model** trained as a separate step. It reparameterizes the RL objective so the **policy** update uses **preferred vs dispreferred** responses—typically increasing the relative log-probability of **preferred** completions under the reference model. **Tradeoff**: simpler pipeline and no separate reward net; **data** must be **high-quality** preferences, and hyperparameters (e.g. \(\beta\)) need care.

| | **PPO + reward model (typical RLHF)** | **DPO** |
|---|----------------------------------------|--------|
| Stages | Reward model training → PPO fine-tune | Single objective on preference data |
| Reward net | Explicit | Implicit in preference loss |
| Stability | Clipping + surrogate in PPO | Different mechanics (Bradley–Terry / KL form) |

### Generative AI Design Patterns (Lakshmanan et al.)

**Adapter tuning** (15), **Evol-Instruct** (16), **prompt optimization** (20), and **long-term memory** (28) are adjacent: supervised / evolutionary / bandit-style improvement and **retrieval**-based adaptation.

## Use Cases

- **Trading / execution**: policies updated from PnL or risk-adjusted rewards (simulation + online constraints).
- **Robotics / simulators**: PPO-class algorithms on continuous or discrete action spaces.
- **Fraud / abuse**: online classifiers + human review loops; drift-aware retraining.
- **Recommendations**: bandits and RL-style ranking; A/B tests as feedback.
- **LLM alignment**: RLHF (reward + PPO) or **DPO** / variants on human or AI preference data.

## Constraints & Tradeoffs

**Tradeoffs:** ✅ **PPO** clipping improves **stability** in many domains; **DPO** simplifies **preference** pipelines. ⚠️ **RL** needs **simulation** or **safe** exploration; **reward hacking**; **compute**; **evaluation** of “helpfulness” vs **short-term** metrics. **Online** learning risks **catastrophic** forgetting without **replay** or **regularization**.

## References

- **Agentic Design Patterns** — *Antonio Gulli*; [learning-adapter.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/learning-adapter.md) (local: `subagents-design-patterns/agents/learning-adapter.md`).
- Schulman et al., *Proximal Policy Optimization Algorithms* (arXiv:1707.06347).
- Rafailov et al., *Direct Preference Optimization* (arXiv:2305.18290).
- **Pattern 15 (Adapter tuning)**, **Pattern 16 (Evol-Instruct)**, **Pattern 28 (Long-term memory)**

## Related Patterns

- **Adapter tuning (15)**: Supervised **PEFT**; often a **stage** before or after preference methods.
- **Long-term memory (28)**: **Memory-based** adaptation via **retrieval**, not always gradient updates.
- **Content optimization (5)**: Metric-driven **offline** tuning; overlaps with **bandit** / online ideas.
- **Reflection (18)**: Runtime **critique** loops without necessarily **weight** updates.
