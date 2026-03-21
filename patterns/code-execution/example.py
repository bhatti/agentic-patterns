#!/usr/bin/env python3
"""
Code execution pattern — LLM emits DSL; sandbox runs Graphviz (or Matplotlib fallback)

Problem: Diagrams and plots are not one-shot API calls; the right abstraction is often a
spec (DOT, Vega-Lite, constrained Python). The model generates DSL; a postprocessor runs
it in a sandbox and returns an artifact (PNG).

This example uses a *mock* LLM that returns Graphviz DOT for an enterprise-style service
graph. LangGraph orchestrates: generate_dsl → execute_sandbox → END. If the `dot` binary
is missing, a Matplotlib fallback renders an equivalent *visual* so the pipeline still
produces a file.

Setup:
    pip install -r patterns/code-execution/requirements.txt
    # Optional for native DOT rendering:
    # macOS: brew install graphviz
    # Ubuntu: apt install graphviz

Usage:
    python example.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class CodeExecutionState(TypedDict, total=False):
    """State for the DSL generation and sandbox execution pipeline."""

    user_request: str
    dsl: str
    image_path: str | None
    dsl_path: str | None
    error: str | None
    renderer: str


def mock_llm_generate_dot(user_request: str) -> str:
    """
    Simulate an LLM that returns Graphviz DOT for a service topology.

    In production, replace with ChatOllama / API with low temperature and a system prompt
    that constrains output to valid DOT only.

    Args:
        user_request: Natural-language goal (used only for demo branching).

    Returns:
        A DOT source string.
    """
    _ = user_request.lower()
    return """digraph services {
  rankdir=LR;
  graph [fontname="Helvetica", fontsize=10];
  node [shape=box, style=rounded, fontname="Helvetica"];
  edge [fontname="Helvetica", fontsize=9];

  client [label="Client"];
  gateway [label="API Gateway"];
  orders [label="Orders Svc"];
  pay [label="Payments Svc"];
  db [label="Orders DB", shape=cylinder];

  client -> gateway [label="HTTPS"];
  gateway -> orders [label="REST"];
  gateway -> pay [label="REST"];
  orders -> db [label="SQL"];
}
"""


def _write_dot_and_run_graphviz(dsl: str, out_dir: Path) -> tuple[str | None, str | None]:
    """
    Write DOT to a file and invoke `dot -Tpng`. Returns (png_path, error_message).

    Args:
        dsl: Graphviz DOT source.
        out_dir: Directory for temporary artifacts.

    Returns:
        Tuple of output PNG path or None, and error string or None.
    """
    dot_bin = shutil.which("dot")
    if not dot_bin:
        return None, "Graphviz `dot` not found on PATH; install graphviz or use Matplotlib fallback."

    dot_path = out_dir / "graph.dot"
    png_path = out_dir / "graph.png"
    dot_path.write_text(dsl, encoding="utf-8")
    try:
        proc = subprocess.run(
            [dot_bin, "-Tpng", "-o", str(png_path), str(dot_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "Graphviz timed out after 30 seconds."

    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
        return None, "Graphviz failed: " + err

    if not png_path.is_file():
        return None, "Graphviz did not produce output PNG."

    return str(png_path.resolve()), None


def _render_matplotlib_fallback(out_dir: Path) -> str:
    """
    Draw a simple service diagram with Matplotlib when Graphviz is unavailable.

    Args:
        out_dir: Directory for the output PNG.

    Returns:
        Absolute path to the written PNG.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, label: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#4338ca",
            facecolor="#eef2ff",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    box(0.3, 1.0, 1.2, 0.7, "Client")
    box(2.2, 1.0, 1.4, 0.7, "API\nGateway")
    box(4.2, 1.6, 1.3, 0.7, "Orders\nSvc")
    box(4.2, 0.4, 1.3, 0.7, "Payments\nSvc")
    box(6.5, 1.0, 1.1, 0.9, "Orders\nDB",)

    arr = FancyArrowPatch(
        (1.5, 1.35),
        (2.2, 1.35),
        arrowstyle="-|>",
        mutation_scale=12,
        color="#6366f1",
        linewidth=1.2,
    )
    ax.add_patch(arr)
    ax.add_patch(
        FancyArrowPatch((3.6, 1.55), (4.2, 1.85), arrowstyle="-|>", mutation_scale=10, color="#6366f1")
    )
    ax.add_patch(
        FancyArrowPatch((3.6, 1.15), (4.2, 0.75), arrowstyle="-|>", mutation_scale=10, color="#6366f1")
    )
    ax.add_patch(
        FancyArrowPatch((5.5, 1.35), (6.5, 1.35), arrowstyle="-|>", mutation_scale=10, color="#6366f1")
    )
    ax.text(1.85, 1.55, "HTTPS", fontsize=7, color="#64748b")
    ax.text(5.0, 2.05, "REST", fontsize=7, color="#64748b")
    ax.text(5.0, 0.95, "REST", fontsize=7, color="#64748b")
    ax.text(6.0, 1.55, "SQL", fontsize=7, color="#64748b")

    out_path = out_dir / "graph_fallback.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_path.resolve())


def execute_in_sandbox(state: CodeExecutionState) -> CodeExecutionState:
    """
    Run DOT through Graphviz, or fall back to Matplotlib if `dot` is missing.

    Args:
        state: Must contain ``dsl``.

    Returns:
        Updated state with ``image_path``, ``renderer``, and optional ``error``.
    """
    dsl = state.get("dsl") or ""
    persistent = Path(__file__).resolve().parent / "_output"
    persistent.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="code_exec_") as tmp:
        out_dir = Path(tmp)
        png, err = _write_dot_and_run_graphviz(dsl, out_dir)
        if png:
            dest = persistent / "graph.png"
            shutil.copy2(png, dest)
            return {
                **state,
                "image_path": str(dest),
                "dsl_path": None,
                "error": None,
                "renderer": "graphviz",
            }

        dest_fb = persistent / "graph_fallback.png"
        dot_saved = persistent / "graph.dot"
        dot_saved.write_text(dsl, encoding="utf-8")
        try:
            fb_path = _render_matplotlib_fallback(out_dir)
            shutil.copy2(fb_path, dest_fb)
            return {
                **state,
                "image_path": str(dest_fb),
                "dsl_path": str(dot_saved),
                "error": err,
                "renderer": "matplotlib",
            }
        except Exception as exc:  # noqa: BLE001 — demo surfaces any matplotlib issue
            return {
                **state,
                "image_path": None,
                "dsl_path": str(dot_saved),
                "error": (
                    (err or "")
                    + " | matplotlib fallback failed: "
                    + str(exc)
                    + ". DOT saved; install deps for PNG fallback."
                ),
                "renderer": "none",
            }


def node_generate_dsl(state: CodeExecutionState) -> CodeExecutionState:
    """LangGraph node: fill ``dsl`` from the mock LLM."""
    req = state.get("user_request", "")
    return {**state, "dsl": mock_llm_generate_dot(req)}


def run_pipeline_langgraph(user_request: str) -> CodeExecutionState:
    """
    Run the generate → sandbox pipeline using LangGraph.

    Args:
        user_request: User goal string.

    Returns:
        Final state after execution.
    """
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(CodeExecutionState)
    workflow.add_node("generate_dsl", node_generate_dsl)
    workflow.add_node("sandbox", execute_in_sandbox)
    workflow.set_entry_point("generate_dsl")
    workflow.add_edge("generate_dsl", "sandbox")
    workflow.add_edge("sandbox", END)
    app = workflow.compile()
    initial: CodeExecutionState = {"user_request": user_request}
    result: Any = app.invoke(initial)
    return result  # type: ignore[return-value]


def run_pipeline_direct(user_request: str) -> CodeExecutionState:
    """Same pipeline without LangGraph (for environments without langgraph)."""
    s: CodeExecutionState = {"user_request": user_request}
    s = node_generate_dsl(s)
    return execute_in_sandbox(s)


def main() -> None:
    """Entry point: run pipeline and print paths and renderer."""
    request = "Show our API gateway, orders and payments services, and orders database."
    try:
        final = run_pipeline_langgraph(request)
    except ImportError:
        final = run_pipeline_direct(request)

    print("Pattern 22: Code execution (DSL + sandbox)")
    print("  user_request:", request[:60] + "...")
    print("  dsl (first 200 chars):", (final.get("dsl") or "")[:200].replace("\n", " ") + "...")
    print("  renderer:", final.get("renderer"))
    print("  image_path:", final.get("image_path"))
    if final.get("dsl_path"):
        print("  dsl_path:", final.get("dsl_path"))
    if final.get("error"):
        print("  note:", final.get("error"))


if __name__ == "__main__":
    main()
