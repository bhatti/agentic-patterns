#!/usr/bin/env python3
"""
Multi-agent collaboration — hierarchical specialists + merge + adversarial critic

Problem: A single LLM session is a poor fit for long, multi-domain workflows (vendor
security review, incident response, content pipelines). Specialized agents with narrow
prompts and optional different tools reduce cognitive load per role and mirror how teams
work: plan → parallelizable experts → integrate → challenge → finalize.

This example models a *vendor security onboarding* brief. Calls are mocked (no API keys).
Swap ``mock_llm`` bodies for ChatOllama / cloud models in production.

Setup:
    pip install -r patterns/multiagent-collaboration/requirements.txt

Usage:
    python example.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class MultiAgentState(TypedDict, total=False):
    """Shared state for a hierarchical multi-agent run."""

    user_request: str
    task_plan: str
    technical_findings: str
    compliance_findings: str
    merged_brief: str
    critic_review: str
    final_report: str


def mock_orchestrator_plan(user_request: str) -> str:
    """
    Simulate a planner that decomposes the user goal into subtasks.

    In production, use a small, low-temperature model with structured output.

    Args:
        user_request: Natural-language goal from the user.

    Returns:
        A short numbered plan string.
    """
    _ = user_request
    return (
        "1) Technical: architecture, data flows, subprocessors.\n"
        "2) Compliance: GDPR/SOC2 evidence, DPAs, breach notification.\n"
        "3) Merge: single executive brief.\n"
        "4) Adversarial pass: find gaps and overstated claims.\n"
        "5) Finalize: incorporate critic without diluting must-fix items."
    )


def mock_technical_agent(user_request: str) -> str:
    """
    Simulate a technical specialist (threat model, integrations).

    Args:
        user_request: Original user request for context.

    Returns:
        Technical findings as prose.
    """
    return (
        "[Technical] For the stated vendor scope: API is REST over TLS 1.3; "
        "webhooks use HMAC signatures; no customer PII stored at rest in vendor shard "
        "(metadata only). Subprocessors: cloud region EU-West; CDN global. "
        "Gaps: public S3-style URL exposure not ruled out in questionnaire—request "
        "pen-test summary dated within 12 months.\n"
        f"(Context: {user_request[:80]}...)"
    )


def mock_compliance_agent(user_request: str) -> str:
    """
    Simulate a compliance specialist (policy, contracts, regulatory).

    Args:
        user_request: Original user request for context.

    Returns:
        Compliance findings as prose.
    """
    return (
        "[Compliance] DPA on file with SCCs; SOC2 Type II report provided (covers prior "
        "year). GDPR: DSR workflow documented; breach notification clause aligns with "
        "72h where applicable. Open item: subprocessors list must be updated within 30 "
        "days of onboarding—verify process owner.\n"
        f"(Context: {user_request[:80]}...)"
    )


def mock_merge(technical: str, compliance: str) -> str:
    """
    Integrate specialist outputs into one brief (executive summary style).

    Args:
        technical: Output from the technical agent.
        compliance: Output from the compliance agent.

    Returns:
        Merged brief.
    """
    return (
        "## Vendor security brief (draft)\n\n"
        "### Technical posture\n"
        + technical
        + "\n\n### Compliance & privacy\n"
        + compliance
        + "\n\n### Recommendation (pre-review)\n"
        "Proceed to conditional approval pending pen-test artifact and subprocessor update process."
    )


def mock_adversarial_critic(merged_brief: str) -> str:
    """
    Simulate a red-team / critic agent that challenges the merged brief.

    Args:
        merged_brief: Integrated output from specialists.

    Returns:
        Critique listing weaknesses and missing evidence.
    """
    _ = merged_brief
    return (
        "[Critic] Issues: (1) 'No PII at rest' is asserted without citing datastore audit; "
        "demand schema review or attestation. (2) SOC2 scope may exclude webhook "
        "delivery path—confirm boundary. (3) Recommendation is too strong given open pen-test gap; "
        "downgrade to 'pilot with logging' until evidence arrives."
    )


def mock_finalize(merged_brief: str, critic_review: str) -> str:
    """
    Produce the final report by folding critic feedback into the brief.

    Args:
        merged_brief: Pre-critic merged content.
        critic_review: Adversarial review text.

    Returns:
        Final user-facing report.
    """
    return (
        merged_brief
        + "\n\n---\n## Independent review (adversarial)\n"
        + critic_review
        + "\n\n---\n## Final stance\n"
        "**Conditional approval — pilot only.** Address critic items before full production rollout."
    )


def node_plan(state: MultiAgentState) -> MultiAgentState:
    """Orchestrator node: write task plan."""
    req = state.get("user_request", "")
    return {**state, "task_plan": mock_orchestrator_plan(req)}


def node_technical(state: MultiAgentState) -> MultiAgentState:
    """Technical specialist node."""
    req = state.get("user_request", "")
    return {**state, "technical_findings": mock_technical_agent(req)}


def node_compliance(state: MultiAgentState) -> MultiAgentState:
    """Compliance specialist node."""
    req = state.get("user_request", "")
    return {**state, "compliance_findings": mock_compliance_agent(req)}


def node_merge(state: MultiAgentState) -> MultiAgentState:
    """Integrator node: combine specialist outputs."""
    t = state.get("technical_findings", "")
    c = state.get("compliance_findings", "")
    return {**state, "merged_brief": mock_merge(t, c)}


def node_critic(state: MultiAgentState) -> MultiAgentState:
    """Adversarial verifier node."""
    m = state.get("merged_brief", "")
    return {**state, "critic_review": mock_adversarial_critic(m)}


def node_finalize(state: MultiAgentState) -> MultiAgentState:
    """Final merge after critique."""
    m = state.get("merged_brief", "")
    cr = state.get("critic_review", "")
    return {**state, "final_report": mock_finalize(m, cr)}


def build_multiagent_graph() -> Any:
    """
    Build a LangGraph linear pipeline: plan → technical → compliance → merge → critic → finalize.

    Returns:
        Compiled LangGraph application.
    """
    from langgraph.graph import END, StateGraph

    g = StateGraph(MultiAgentState)
    g.add_node("plan", node_plan)
    g.add_node("technical", node_technical)
    g.add_node("compliance", node_compliance)
    g.add_node("merge", node_merge)
    g.add_node("critic", node_critic)
    g.add_node("finalize", node_finalize)
    g.set_entry_point("plan")
    g.add_edge("plan", "technical")
    g.add_edge("technical", "compliance")
    g.add_edge("compliance", "merge")
    g.add_edge("merge", "critic")
    g.add_edge("critic", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def run_sequential_mock(user_request: str) -> MultiAgentState:
    """
    Run the same pipeline without LangGraph (fallback for missing dependency).

    Args:
        user_request: User goal.

    Returns:
        Final multi-agent state.
    """
    s: MultiAgentState = {"user_request": user_request}
    s = node_plan(s)
    s = node_technical(s)
    s = node_compliance(s)
    s = node_merge(s)
    s = node_critic(s)
    s = node_finalize(s)
    return s


def main() -> None:
    """Run the demo and print a short summary and the final report tail."""
    user_request = (
        "Prepare a third-party security assessment for Acme Analytics (EU-hosted SaaS, "
        "CRM integration, webhook callbacks). We need approval for production customer data."
    )
    try:
        app = build_multiagent_graph()
        result: Any = app.invoke({"user_request": user_request})
    except ImportError:
        result = run_sequential_mock(user_request)

    print("Pattern 23: Multi-agent collaboration (hierarchical + critic)")
    print("  scenario: vendor security onboarding brief")
    print("  task_plan (excerpt):", (result.get("task_plan") or "")[:120].replace("\n", " | "), "...")
    print()
    final = result.get("final_report") or ""
    print(final[:1800])
    if len(final) > 1800:
        print("\n... [truncated]")


if __name__ == "__main__":
    main()
