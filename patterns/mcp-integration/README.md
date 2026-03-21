# Pattern 47: MCP Integration (Agentic)

## Overview

**Model Context Protocol (MCP)** standardizes how **hosts** (IDEs, agents) **discover**, **authenticate**, and **call** **tools** **and** **resources** **exposed** **by** **servers**—**JSON-RPC**-style **messages**, **capability** **negotiation**, and **modular** **composition** **of** **external** **systems**.

**Gulli**: [mcp-integrator.md](https://github.com/anti-achismo-social-club/subagents-design-patterns/blob/main/agents/mcp-integrator.md).

## Relationship to Pattern 21

**Tool calling (21)** is the **model** **API** **pattern**; **MCP** is a **wire** **and** **lifecycle** **standard** **for** **tool** **servers**. **Use** **together**: MCP **servers** **behind** **the** **same** **agent** **loop**.

## References

- [MCP specification](https://modelcontextprotocol.io/)
- **Pattern 21 (Tool calling)**
