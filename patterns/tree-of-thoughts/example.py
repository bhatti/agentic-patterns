#!/usr/bin/env python3
"""
Tree of Thoughts (ToT) Pattern - Incident Root-Cause Analysis

This example implements ToT for exploring multiple root-cause hypotheses
in parallel, evaluating each path, and summarizing the most likely cause.

Real-World Problem:
-------------------
Incident response requires strategic thinking: there are multiple plausible
causes (DB, cache, dependency, resources). A single linear CoT might fixate
on one path and miss the real cause. ToT explores several hypotheses,
evaluates how promising each direction is, keeps the top K paths (beam search),
and summarizes the best root cause and next steps.

Four components:
1. Thought generation: N possible next hypotheses or checks
2. Path evaluation: Score 0-1 for how promising the path is
3. Beam search: Keep top K states
4. Summary generation: Concise root cause and recommendations

Usage:
    python example.py
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ToTResult:
    """Result of Tree of Thoughts search."""
    problem: str
    solution_summary: str
    best_score: float
    reasoning_path: List[str]
    final_state: str
    stats: Dict[str, Any]


# =============================================================================
# LLM INTERFACE (simulated for demo; replace with Ollama/OpenAI in production)
# =============================================================================

def _simulate_generate_thoughts(state: str, step: int, num_thoughts: int, problem: str) -> List[str]:
    """
    Simulated thought generation: return diverse next hypotheses/checks
    based on problem keywords. In production, call LLM with prompt.
    """
    state_lower = state.lower()
    problem_lower = problem.lower()
    thoughts = []

    # Predefined thought pools for incident-style problems
    if step == 1:
        if "latency" in problem_lower or "slow" in problem_lower:
            thoughts = [
                "Hypothesis: Database connection pool exhausted or slow queries.",
                "Hypothesis: Cache hit rate dropped or cache layer timeout.",
                "Hypothesis: Downstream dependency (API or service) is slow or failing.",
                "Hypothesis: CPU, memory, or thread pool saturation.",
                "Hypothesis: Network or DNS resolution issues.",
            ]
        elif "error" in problem_lower or "failure" in problem_lower:
            thoughts = [
                "Hypothesis: Recent deployment introduced a bug or config change.",
                "Hypothesis: External service or API returned errors.",
                "Hypothesis: Resource limits (disk, memory, connections) exceeded.",
                "Hypothesis: Authentication or authorization failure.",
            ]
        else:
            thoughts = [
                "Check primary component (e.g., database or main service).",
                "Check dependencies and integrations.",
                "Check infrastructure and resources.",
            ]
    else:
        # Deeper steps: more specific follow-ups
        if "database" in state_lower or "db" in state_lower:
            thoughts = [
                "Check connection pool metrics and active connections.",
                "Check slow query log and top queries by latency.",
                "Check replica lag if read replicas are used.",
            ]
        elif "cache" in state_lower:
            thoughts = [
                "Check cache hit rate and eviction rate.",
                "Check cache cluster health and memory.",
                "Check whether a key or namespace was invalidated.",
            ]
        elif "dependency" in state_lower or "api" in state_lower:
            thoughts = [
                "Check dependency latency and error rate.",
                "Check circuit breaker and timeout configuration.",
                "Check for version or contract mismatch.",
            ]
        elif "cpu" in state_lower or "memory" in state_lower or "resource" in state_lower:
            thoughts = [
                "Check process and system CPU/memory over time.",
                "Check thread pool queue depth and rejection rate.",
                "Check for leaks or gradual growth.",
            ]
        else:
            thoughts = [
                "Dig deeper into the most likely component from current state.",
                "Cross-check with logs and metrics for the leading hypothesis.",
                "Consider time correlation with deployments or traffic.",
            ]

    # Return up to num_thoughts, ensure diversity
    return thoughts[:num_thoughts]


def _simulate_evaluate_state(state: str, problem: str) -> float:
    """
    Simulated path evaluation: score 0-1 based on keywords and length.
    In production, call LLM with criteria (correctness, progress, potential).
    """
    state_lower = state.lower()
    problem_lower = problem.lower()
    score = 0.45

    # Reward matching problem keywords (moderate so we need multiple steps to reach 0.9)
    if "latency" in problem_lower or "slow" in problem_lower:
        if "connection pool" in state_lower or "slow query" in state_lower:
            score += 0.12
        if "cache" in state_lower and "hit" in state_lower:
            score += 0.08
        if "dependency" in state_lower or "timeout" in state_lower:
            score += 0.08
    if "error" in problem_lower:
        if "deployment" in state_lower or "config" in state_lower:
            score += 0.1
        if "external" in state_lower or "service" in state_lower:
            score += 0.08

    # Reward depth: more steps = more progress (cap so 3-4 steps needed to approach 0.9)
    steps = state_lower.count("step 1:") + state_lower.count("step 2:") + state_lower.count("step 3:") + state_lower.count("step 4:")
    score += min(0.08 * steps, 0.2)

    return min(0.92, max(0.0, score))


def _simulate_summary(problem: str, final_state: str) -> str:
    """Simulated summary from best path. In production, call LLM."""
    return (
        "Root cause (most likely): Based on the reasoning path, the most promising "
        "direction points to investigating database connection pool and slow queries, "
        "followed by cache and dependency health. Recommended next steps: (1) Check DB "
        "connection pool metrics and slow query log. (2) Check cache hit rate and "
        "dependency latency. (3) Correlate with deployment time and traffic spike."
    )


def invoke_llm(prompt: str, _model: Optional[str] = None) -> str:
    """Use shared Ollama client if available; otherwise return empty for simulated path."""
    try:
        from shared.ollama_client import get_ollama_client
        client = get_ollama_client()
        if client and _model:
            out = client.generate(prompt, model=_model)
            if out:
                return out
    except Exception:
        pass
    return ""


# =============================================================================
# TREE OF THOUGHTS ENGINE
# =============================================================================

class TreeOfThoughts:
    """
    Tree of Thoughts: thought generation, path evaluation, beam search, summary.
    """

    def __init__(
        self,
        num_thoughts_per_step: int = 3,
        max_steps: int = 4,
        beam_width: int = 3,
        score_threshold: float = 0.9,
        verbose: bool = False,
        generate_thoughts_fn: Optional[Callable[..., List[str]]] = None,
        evaluate_fn: Optional[Callable[[str, str], float]] = None,
        summary_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.num_thoughts_per_step = num_thoughts_per_step
        self.max_steps = max_steps
        self.beam_width = beam_width
        self.score_threshold = score_threshold
        self.verbose = verbose
        self.generate_thoughts_fn = generate_thoughts_fn or _simulate_generate_thoughts
        self.evaluate_fn = evaluate_fn or _simulate_evaluate_state
        self.summary_fn = summary_fn or _simulate_summary
        self.call_count = 0

    def generate_thoughts(self, state: str, step: int, problem: str) -> List[str]:
        """Generate N possible next thoughts from current state."""
        self.call_count += 1
        thoughts = self.generate_thoughts_fn(state, step, self.num_thoughts_per_step, problem)
        if self.verbose:
            print(f"  [Step {step}] Generated {len(thoughts)} thoughts")
        return thoughts

    def evaluate_state(self, state: str, problem: str) -> float:
        """Score path promise (0-1)."""
        self.call_count += 1
        return self.evaluate_fn(state, problem)

    def generate_summary(self, problem: str, final_state: str) -> str:
        """Produce concise summary of best path."""
        self.call_count += 1
        return self.summary_fn(problem, final_state)

    def solve(self, problem: str) -> ToTResult:
        """
        Solve using ToT: thought generation → evaluation → beam search → summary.
        """
        start = time.time()
        self.call_count = 0

        initial_state = f"Problem: {problem}\n\nInitial analysis:"
        # Beam: list of (score, state, path, step); we keep highest scores
        beam: List[Tuple[float, str, List[str], int]] = [(0.5, initial_state, [], 0)]
        best_final: List[Tuple[float, str, List[str]]] = []

        for step in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n--- Step {step} ---")

            candidates: List[Tuple[float, str, List[str], int]] = []

            for score, current_state, path, path_step in beam:
                if path_step >= step:
                    candidates.append((score, current_state, path, path_step))
                    continue

                thoughts = self.generate_thoughts(current_state, step, problem)

                for thought in thoughts:
                    new_state = f"{current_state}\n\nStep {step}: {thought}"
                    new_path = path + [f"Step {step}: {thought}"]
                    new_score = self.evaluate_state(new_state, problem)
                    candidates.append((new_score, new_state, new_path, step))

                    if new_score >= self.score_threshold:
                        best_final.append((new_score, new_state, new_path))

            if best_final and max(b[0] for b in best_final) > 0.95:
                break

            # Beam: keep top K by score (descending)
            beam = sorted(candidates, key=lambda x: -x[0])[: self.beam_width]

            if self.verbose and beam:
                for i, (s, st, _, _) in enumerate(beam):
                    last_line = st.strip().split("\n")[-1][:60] if "\n" in st else st[:60]
                    print(f"  Beam {i + 1} (score={s:.2f}): {last_line}...")

        if best_final:
            best_score, best_state, best_path = max(best_final, key=lambda x: x[0])
        else:
            best_score, best_state, best_path, _ = max(beam, key=lambda x: x[0])

        summary = self.generate_summary(problem, best_state)
        elapsed = time.time() - start

        return ToTResult(
            problem=problem,
            solution_summary=summary,
            best_score=best_score,
            reasoning_path=best_path,
            final_state=best_state,
            stats={
                "call_count": self.call_count,
                "elapsed_time_sec": round(elapsed, 2),
                "steps_taken": len(best_path),
            },
        )


# =============================================================================
# INCIDENT ROOT-CAUSE ANALYZER
# =============================================================================

class IncidentRootCauseAnalyzer:
    """
    Uses Tree of Thoughts to explore root-cause hypotheses and summarize
    the most likely cause and next steps.
    """

    def __init__(self, tot: Optional[TreeOfThoughts] = None):
        self.tot = tot or TreeOfThoughts(
            num_thoughts_per_step=3,
            max_steps=4,
            beam_width=3,
            verbose=True,
        )

    def analyze(self, problem: str) -> ToTResult:
        """Run ToT over the incident description and return result."""
        return self.tot.solve(problem)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("TREE OF THOUGHTS PATTERN - INCIDENT ROOT-CAUSE ANALYSIS")
    print("=" * 70)
    print()
    print("Explores multiple root-cause hypotheses, evaluates each path,")
    print("keeps top K (beam search), and summarizes the best conclusion.")
    print()

    problem = """
    Our API latency (p99) has spiked from ~200ms to ~3s over the last 30 minutes.
    Error rate is slightly elevated. We use a relational DB, Redis cache, and
    two downstream services. No deployment in the last 2 hours. Traffic is normal.
    """

    analyzer = IncidentRootCauseAnalyzer()
    result = analyzer.analyze(problem.strip())

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"\nProblem: {result.problem[:200]}...")
    print(f"\nBest path score: {result.best_score:.2f}")
    print("\nReasoning path:")
    for step in result.reasoning_path:
        print(f"  {step}")
    print("\nSolution summary:")
    print(f"  {result.solution_summary}")
    print("\nStats:")
    print(f"  Evaluations/calls: {result.stats['call_count']}")
    print(f"  Time (s): {result.stats['elapsed_time_sec']}")
    print(f"  Steps in path: {result.stats['steps_taken']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
