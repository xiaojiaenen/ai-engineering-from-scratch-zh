---
name: gateway-bootstrap
description: Produce a gateway configuration spec given users, backends, and compliance constraints.
description-zh: # Gateway Configuration Specification

## 1. Overview

This document defines the configuration for an API Gateway integrating **users**, **backends**, and **compliance constraints**.

---

## 2. Users

| User ID | Role | Auth Method | Rate Limit | Scope |
|---------|------|-------------|------------|-------|
| `u-001` | Admin | OAuth 2.0 + MFA | 1000 req/min | Full CRUD |
| `u-002` | Partner | API Key + TLS Mutual Auth | 500 req/min | Read + Write |
| `u-003` | Internal Service | mTLS Certificate | 5000 req/min | Full CRUD |
| `u-004` | External Consumer | JWT (RS256) | 200 req/min | Read-only |

### Authentication Policies
```yaml
authentication:
  methods:
    - type: oauth2
      issuer: "https://auth.example.com"
      jwks_uri: "https://auth.example
version: 1.0.0
phase: 13
lesson: 17
tags: [mcp, gateway, rbac, audit, policy]
---

Given an enterprise MCP plan (users, backends, compliance constraints), produce the gateway configuration spec.

Produce:

1. Backend list. Each with its registry (Official / Glama / custom), canonical name (reverse-DNS), pinned description hashes.
2. User list. Each with a role and allowed-tool set.
3. RBAC matrix. One row per user x backend-tool, with allow/deny.
4. Rate limits. Per-user burst and sustained limits; per-tool limits for expensive tools.
5. Audit plan. Log destination (file, OpenTelemetry, SIEM), retention, fields captured.

Hard rejects:
- Any backend not in the Official Registry without explicit admin approval.
- Any RBAC rule allowing all users all tools. Privilege explosion.
- Any audit plan without immutable storage. Compliance fail.

Refusal rules:
- If a developer population exceeds 100 without any roles defined, refuse to bootstrap and require at least three roles.
- If the plan does not identify an OAuth 2.1 identity provider, refuse and recommend adopting Keycloak or Auth0 first.
- If any backend uses stdio, refuse to proxy it through the HTTP gateway; stdio servers run per-developer locally.

Output: a one-page config document with backend list, user list, RBAC matrix, rate limits, and audit plan. End with the single policy rule the team should implement first.
