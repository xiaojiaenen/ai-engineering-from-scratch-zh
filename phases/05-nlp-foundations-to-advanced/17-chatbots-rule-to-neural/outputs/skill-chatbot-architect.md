---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
description-zh: # Chatbot Stack Design Framework

Since no specific use case was provided, I'll use a **Customer Support Chatbot for an E-commerce Platform** as an illustrative example. This framework can be adapted to any use case.

---

## 1. Define the Use Case

| Attribute | Details |
|---|---|
| **Goal** | Automate 70% of customer support queries |
| **Users** | E-commerce shoppers |
| **Channels** | Website, Mobile App, WhatsApp, Facebook Messenger |
| **Key Tasks** | Order tracking, returns/refunds, FAQs, escalation to human agents |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────┐
│                  CHANNELS                        │
│  (Web Widget, Mobile App, WhatsApp, Messenger)   │
└────────────────
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
