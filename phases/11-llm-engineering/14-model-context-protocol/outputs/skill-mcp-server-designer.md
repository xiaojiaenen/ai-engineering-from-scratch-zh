---
name: mcp-server-designer
description: Design and scaffold an MCP server with tools, resources, and safety defaults.
description-zh: # MCP Server Scaffold with Tools, Resources & Safety Defaults

## Project Structure

```
mcp-server/
├── pyproject.toml
├── config/
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── server.py              # Main MCP server entry point
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── defaults.py        # Safety policies & guardrails
│   │   ├── rate_limiter.py    # Rate limiting
│   │   └── validator.py       # Input/output validation
│   ├── tools/
│
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

Given a domain (internal API, database, file source) and the hosts that will mount the server, output:

1. Primitive map. Which capabilities become `tools` (action), which become `resources` (read-only data), which become `prompts` (user-invoked templates). One line per primitive.
2. Auth plan. Stdio (trusted local), streamable HTTP with API key, or OAuth 2.1 with PKCE. Pick and justify.
3. Schema draft. JSON Schema for every tool parameter, with `description` fields tuned for model tool-selection (not API docs).
4. Destructive-action list. Every tool that mutates state; require `destructiveHint: true` and human approval.
5. Test plan. Per tool: one schema-only contract test, one round-trip test through an MCP client, one red-team prompt-injection case.

Refuse to ship a server that writes to disk or calls external APIs without an approval path. Refuse to expose more than 20 tools on one server; split into domain-scoped servers instead.
