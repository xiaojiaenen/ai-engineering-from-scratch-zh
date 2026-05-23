---
name: elicitation-form-designer
description: Design the elicitation form schema and message template for a tool that needs mid-call user confirmation or disambiguation.
description-zh: # Elicitation Form Schema & Message Template

## Overview

This design covers a **mid-call elicitation** pattern: a tool pauses execution to ask the user for confirmation or disambiguation, then resumes once the user responds.

---

## 1. Elicitation Form Schema (JSON Schema)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://example.com/schemas/elicitation-form.json",
  "title": "ElicitationForm",
  "description": "Schema for mid-call user confirmation or disambiguation requests.",
  "type": "object",
  "required": ["elicitationId", "triggerTool", "prompt", "responseType
version: 1.0.0
phase: 13
lesson: 12
tags: [mcp, elicitation, user-input, forms]
---

Given a tool whose behavior may require mid-call user input, design the elicitation schema and message.

Produce:

1. Trigger condition. State the exact input or ambiguity that should cause the tool to call `elicitation/create`.
2. Message template. One sentence the host shows the user. Plain, specific, free of jargon.
3. Schema. Flat JSON Schema with typed properties and the `enum` list (for disambiguation) or `boolean` (for confirmation). Do not nest.
4. Branch handling. Map `accept` / `decline` / `cancel` to tool behaviors.
5. Rate-limit rule. Cap elicitations per tool invocation; never elicit inside a loop.

Hard rejects:
- Any schema that nests objects. Elicitation v1 is flat.
- Any elicitation used to pad a missing argument the LLM could have asked for in prose.
- Any high-frequency elicitation (more than once per tool call).

Refusal rules:
- If the tool is read-only and low-risk, refuse to elicit and just return the result.
- If the tool is destructive and the host supports `destructiveHint` annotations, suggest using annotations and letting the client handle confirmation natively.
- If the need is an OAuth sign-in, recommend URL-mode elicitation and flag the SEP-1036 drift risk.

Output: a one-page design with trigger condition, message template, schema, branch handling, rate-limit rule, and a note on whether form mode or URL mode fits better.
