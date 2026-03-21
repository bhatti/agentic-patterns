#!/usr/bin/env python3
"""
Tool Calling Pattern — LangGraph agent with enterprise-style tools

Problem: RAG injects knowledge, but LLMs cannot call your OMS, FAQ search, or
booking APIs. Tool calling lets the model emit structured tool calls; LangGraph
runs a ReAct-style loop: assistant -> ToolNode -> assistant until done.

This example uses LangGraph (StateGraph + ToolNode) like the book reference,
but with local-first tools that mock an order-management system and internal
FAQ — no Google Maps or external keys required.

Setup (use a virtual environment):
    pip install -r requirements.txt
    ollama pull llama3.2
    ollama serve   # default :11434

Usage:
    python example.py
    # Optional: OLLAMA_MODEL=qwen2.5 OLLAMA_BASE_URL=http://localhost:11434
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _run_langgraph_demo() -> None:
    """Build and run the LangGraph agent with Ollama + tools."""
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "Install dependencies: pip install -r patterns/tool-calling/requirements.txt"
        ) from exc

    # ------------------------------------------------------------------
    # Tools: stand-ins for enterprise APIs (OMS + knowledge search)
    # ------------------------------------------------------------------
    @tool
    def lookup_order_status(order_id: str) -> str:
        """
        Look up shipping and payment status for a customer order in the order management system.
        Use when the user mentions an order number or wants delivery status.
        """
        oid = order_id.strip().upper()
        if oid.endswith("123") or oid == "ORD-123":
            return (
                '{"order_id":"ORD-123","status":"shipped","carrier":"ExampleShip",'
                '"tracking":"1Z999","eta_days":2}'
            )
        if "DELAY" in oid or "DELAY" in order_id.upper():
            return '{"order_id":"' + oid + '","status":"delayed","reason":"weather at hub","eta_days":5}'
        return '{"order_id":"' + oid + '","status":"processing","eta_days":3}'

    @tool
    def search_internal_faq(query: str) -> str:
        """
        Search the internal FAQ for return policy, shipping times, and warranty rules.
        Use for policy questions that do not require a specific order id.
        """
        q = query.lower()
        if "return" in q or "refund" in q:
            return (
                "FAQ: Returns accepted within 30 days if unopened; opened items may be eligible for "
                "store credit. Refunds to original payment within 5-10 business days."
            )
        if "ship" in q or "delivery" in q:
            return (
                "FAQ: Standard shipping 3-5 business days; express 1-2 days. International varies by region."
            )
        return "FAQ: No exact match; escalate to human support for account-specific policy."

    tools = [lookup_order_status, search_internal_faq]

    model_name = os.environ.get("OLLAMA_MODEL", "llama3.2")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(model=model_name, base_url=base_url, temperature=0)
    model = llm.bind_tools(tools)

    def assistant_next_node(state: MessagesState):
        messages = state["messages"]
        last = messages[-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("assistant", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("assistant")
    workflow.add_conditional_edges("assistant", assistant_next_node)
    workflow.add_edge("tools", "assistant")

    app = workflow.compile()

    system_hint = (
        "You are a helpful support assistant. Use tools when you need live order data or "
        "internal FAQ policy. Do not invent order status or policy details. "
        "After tools return, reply concisely in plain English."
    )

    questions = [
        "What is the status of order ORD-123?",
        "What is your return policy for opened items?",
    ]

    for question in questions:
        print("=" * 60)
        print("Question:", question)
        print("-" * 60)
        final = app.invoke(
            {"messages": [HumanMessage(content=f"{system_hint}\n\nUser: {question}")]}
        )
        for m in final["messages"]:
            role = getattr(m, "type", None) or m.__class__.__name__
            content = getattr(m, "content", "")
            tcalls = getattr(m, "tool_calls", None)
            if tcalls:
                print(role, "tool_calls:", tcalls)
            elif content:
                print(role, ":", content[:500] + ("..." if len(str(content)) > 500 else ""))
        print()


def _run_fallback_demo() -> None:
    """Explain the flow when LangGraph / LangChain are not installed."""
    print("Tool Calling — LangGraph pattern (offline walkthrough)")
    print()
    print("Install dependencies in a venv, then re-run:")
    print("  pip install -r patterns/tool-calling/requirements.txt")
    print("  ollama pull llama3.2")
    print()
    print("Graph: assistant (LLM + bind_tools) -> [tools if tool_calls] -> assistant -> END")
    print("Tools mock enterprise APIs: lookup_order_status, search_internal_faq")
    print()
    print("Example ReAct trace (conceptual):")
    print("  1. User: What is the status of order ORD-123?")
    print("  2. Model: tool_calls=[lookup_order_status({order_id: ORD-123})]")
    print("  3. ToolNode: returns JSON with status shipped, tracking ...")
    print("  4. Model: final answer summarizing shipment status")
    print()
    print("MCP: expose the same tools via Model Context Protocol for IDE/agents;")
    print("see generative-ai-design-patterns/examples/21_tool_calling/mcp/")


def main() -> None:
    try:
        _run_langgraph_demo()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        print()
        _run_fallback_demo()
        sys.exit(1)
    except Exception as e:
        print(f"LangGraph demo failed: {e}", file=sys.stderr)
        print("Ensure Ollama is running and a tool-capable model is pulled (e.g. llama3.2).", file=sys.stderr)
        print()
        _run_fallback_demo()
        sys.exit(1)


if __name__ == "__main__":
    main()
