---
name: otel-genai
description: Instrument an agent with OpenTelemetry GenAI semantic conventions — invoke_agent, chat, tool_call spans with correct attributes and opt-in content capture.
description-zh: 采用OpenTelemetry GenAI语义规范为智能体添加观测工具——包含invoke_agent、chat、tool_call跨度及其正确属性，并支持可选的内容捕获功能。

## Full Implementation

```python
"""
agent_telemetry.py

OpenTelemetry instrumentation for GenAI agents following the
OpenTelemetry GenAI Semantic Conventions (v1.28+).

Spans produced:
  invoke_agent  — top-level agent run
  chat          — individual LLM completion calls
  tool_call     — tool/function executions
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

# ── OpenTelemetry SDK ─────────────────────────────────────────────
from opentelemetry import trace
from opente
version: 1.0.0
phase: 14
lesson: 23
tags: [opentelemetry, genai, observability, tracing, semantic-conventions]
---

Given an agent runtime, wire OTel GenAI semantic conventions.

Produce:

1. `invoke_agent` span per agent run. Kind CLIENT for remote agent services, INTERNAL for in-process. Name: `invoke_agent {gen_ai.agent.name}`.
2. `chat` span per LLM call with `gen_ai.operation.name=chat`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`.
3. `tool_call` span per tool invocation with `gen_ai.tool.name` and, when applicable, `gen_ai.data_source.id` (RAG corpus / memory store).
4. Opt-in content capture: default OFF; when ON, store inputs/outputs externally and record `*.reference_id` on spans.
5. Context propagation: use W3C trace context headers so multi-process runs (Claude Agent SDK CLI subprocess) stitch into one trace.

Hard rejects:

- Capturing full prompts/outputs inline by default. PII and secret leakage risk; also violates the spec.
- Missing `gen_ai.provider.name`. Multi-provider dashboards break.
- Orphan tool spans. Always set parent-child relation via active context.

Refusal rules:

- If the runtime cannot propagate context across process boundaries, refuse. Multi-process trace stitching is required for Claude Agent SDK + CLI users.
- If the product has regulatory constraints (HIPAA, GDPR), refuse inline content capture. External store with access control only.
- If the backend does not set `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`, warn: attribute names may change on collector upgrade.

Output: `tracer.py`, `attributes.py`, `content_store.py`, `README.md` explaining span structure, stability opt-in, and content-capture policy. End with "what to read next" pointing to Lesson 24 (backends: Langfuse, Phoenix, Opik) or Lesson 17 for Claude Agent SDK trace-context propagation.
