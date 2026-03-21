#!/usr/bin/env python3
"""MCP-style message stubs (no network). Usage: python example.py"""

from __future__ import annotations

import json
from typing import Any


def list_tools_request() -> dict[str, Any]:
    """JSON-RPC shaped request listing tools (illustrative)."""
    return {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


def tools_list_response(tool_names: list[str]) -> dict[str, Any]:
    """Mock successful tools/list payload."""
    tools = [{"name": n, "description": f"tool {n}"} for n in tool_names]
    return {"jsonrpc": "2.0", "id": 1, "result": {"tools": tools}}


def call_tool_request(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """JSON-RPC tools/call style body."""
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def main() -> None:
    print(json.dumps(list_tools_request()))
    print(json.dumps(tools_list_response(["read_file", "grep"])))
    print(json.dumps(call_tool_request("grep", {"pattern": "TODO"})))


if __name__ == "__main__":
    main()
