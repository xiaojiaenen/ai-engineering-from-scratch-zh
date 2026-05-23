# 综合项目 13 — 带注册表和治理功能的 MCP 服务器

> 模型上下文协议在 2026 年不再是未来，而成为了默认的工具使用规范。Anthropic、OpenAI、Google 以及每个主要的 IDE 都配备了 MCP 客户端。Pinterest 发布了其内部的 MCP 服务器生态系统。AAIF 注册表在 `.well-known` 规范化了能力元数据。AWS ECS 发布了参考无状态部署。Block 的 goose-agent 将同一协议嵌入了一个托管助手。2026 年的生产形态是：StreamableHTTP 传输、OAuth 2.1 作用域、OPA 策略网关，以及一个能让平台团队发现、验证和启用服务器的注册表。端到端地构建它。

**类型：** 综合项目
**语言：** Python (服务器，通过 FastMCP) 或 TypeScript (@modelcontextprotocol/sdk)，Go (注册表服务)
**先决条件：** 第 11 阶段 (LLM 工程)，第 13 阶段 (工具与 MCP)，第 14 阶段 (智能体)，第 17 阶段 (基础设施)，第 18 阶段 (安全)
**涉及阶段：** P11 · P13 · P14 · P17 · P18
**时间：** 25 小时

## 问题

MCP 成为了工具使用的通用语言。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 以及每一个托管的智能体现在都消费 MCP 服务器。生产环境的挑战不在于编写服务器（FastMCP 让这变得简单），而在于如何以满足企业要求的方式大规模部署它们：每租户的 OAuth 作用域、对破坏性工具的 OPA 策略、StreamableHTTP 的无状态扩展、用于发现的注册表、每次工具调用的审计日志。Pinterest 的内部 MCP 生态系统和 AAIF 注册表规范设定了 2026 年的标准。

你将构建一个暴露 10 个内部工具（Postgres 只读查询、S3 列表、Jira、Linear、Datadog 等）的 MCP 服务器，一个供平台发现的注册表 UI，以及一个用于破坏性工具的人工审批门控。负载测试将展示 StreamableHTTP 的水平扩展能力。审计跟踪将满足企业安全审查的要求。

## 概念

MCP 2026 修订版将 StreamableHTTP 定为默认传输协议。与早期的 stdio 和 SSE 模式不同，StreamableHTTP 默认是无状态的：单个 HTTP 端点接受 JSON-RPC 请求，流式传输响应，并支持用于通知的长连接。无状态意味着可以在负载均衡器后面进行水平扩展。

授权采用 OAuth 2.1，带有按工具划分的作用域。一个 token 携带诸如 `jira:read`、`s3:list`、`postgres:query:readonly` 等作用域。MCP 服务器在工具调用时检查作用域，而不仅仅是在会话开始时。对于高风险工具，服务器会拒绝任何作用域未在最近 N 分钟内提升至 `approved:by:human` 的调用——该提升来自一张 Slack 审批卡片。

注册表是一个独立的服务。每个 MCP 服务器都暴露一个 `.well-known/mcp-capabilities` 文档，包含其工具清单、传输 URL 和认证要求。注册表轮询、验证并索引这些信息。平台团队使用注册表 UI 来查看有哪些工具可用、它们需要哪些作用域，以及哪些团队拥有它们。

## 架构

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## 技术栈

- 服务器框架：FastMCP (Python) 或 `@modelcontextprotocol/sdk` (TypeScript)
- 传输：基于 HTTPS 的 StreamableHTTP (无状态)
- 认证：OAuth 2.1，通过 SPIFFE / SPIRE 实现工作负载身份验证
- 策略：OPA / Rego 规则应用于每个工具；每个请求调用策略决策服务
- 注册表：自托管，消费 `.well-known/mcp-capabilities` 清单
- 人工审批：用于破坏性工具的 Slack 交互式消息
- 部署：AWS ECS Fargate 或 Fly.io，每个租户一个服务器或共享服务器并通过租户作用域隔离
- 审计：每个租户一个追加写入的 JSONL 日志桶，包含每次调用的血缘信息

## 构建步骤

1.  **工具接口。** 暴露 10 个内部工具：Postgres 只读查询、S3 列出对象、Jira 搜索/获取、Linear 搜索/获取、Datadog 指标查询、PagerDuty 值班查询、GitHub 只读、Notion 搜索、Slack 搜索、Salesforce 读取。每个工具都有类型化的 schema 和作用域标签。

2.  **FastMCP 服务器。** 挂载这些工具。配置 StreamableHTTP 传输。添加用于 OAuth token 内省和作用域强制执行的中间件。

3.  **OPA 策略。** 为每个工具编写 Rego 策略：哪些作用域允许调用、适用哪些 PII 脱敏规则、应用哪些载荷大小限制。在每次工具调用时调用决策服务。

4.  **注册表服务。** 独立的 Go 或 TS 服务，轮询已注册服务器的 `.well-known/mcp-capabilities`，使用 JSON Schema 进行验证，并暴露一个提供列表/搜索/验证/启用-禁用功能的 UI。

5.  **能力清单。** 每个服务器暴露 `.well-known/mcp-capabilities`，包含：工具列表、认证要求、传输 URL、所属团队、SLO。

6.  **破坏性工具分离。** 会改变状态的工具（如 Jira 创建、Linear 创建、Postgres 写入）位于第二个 MCP 服务器上，具有更严格的认证流程：token 必须在 15 分钟内通过 Slack 卡片提升并持有 `approved:by:human` 作用域。

7.  **审计日志。** 每个租户一个追加写入的 JSONL 日志：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前通过 Presidio 进行 PII 脱敏。

8.  **负载测试。** 100 个并发客户端通过 StreamableHTTP 连接。通过添加第二个副本展示水平扩展能力；展示负载均衡器在没有会话粘性的情况下重新分配流量。

9.  **一致性测试。** 针对两个服务器运行官方 MCP 一致性测试套件。通过所有必需部分。

## 使用说明

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 交付成果

`outputs/skill-mcp-server.md` 描述了交付物。一个生产级的 MCP 服务器 + 注册表 + 审计层，用于内部工具，具备 OAuth 2.1 作用域和 OPA 网关。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 规范符合性 | StreamableHTTP + 能力清单通过 MCP 一致性测试 |
| 20 | 安全性 | 作用域强制执行、OPA 覆盖每个工具、密钥管理规范 |
| 20 | 可观测性 | 每个工具调用的审计日志，带 PII 脱敏 |
| 20 | 扩展性 | 100 客户端负载测试水平扩展演示 |
| 15 | 注册表用户体验 | 发现 / 验证 / 启用-禁用 工作流 |
| **100** | | |

## 练习

1.  添加一个新工具（Confluence 搜索）。通过注册表验证流程发布它，而无需修改核心服务器。

2.  编写一个 OPA 策略，用于脱敏包含名为 `email`、`ssn` 或 `phone` 列的 Postgres 查询结果。使用探测查询进行测试。

3.  对比 StreamableHTTP 与 stdio 在本地环境下的延迟。报告每次调用的 p50/p95 数据。

4.  实现每租户配额：每个租户每个工具每分钟最多 N 次调用。通过第二个 OPA 规则强制执行。

5.  从 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 运行 MCP 一致性测试套件，并修复所有失败项。

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|-----------------|------------------------|
| StreamableHTTP | "2026 MCP 传输" | 无状态 HTTP + 流式传输；取代了网络化服务器的 SSE + stdio |
| 能力清单 | "知名文档" | `.well-known/mcp-capabilities`，包含工具列表、认证、传输 URL |
| OPA / Rego | "策略引擎" | 开放策略代理，用于根据外部规则授权工具调用 |
| 作用域提升 | "人工批准" | 通过 Slack 审批授予的短期作用域，破坏性工具必需 |
| 注册表 | "工具发现" | 从能力清单中索引 MCP 服务器的服务 |
| 工作负载身份 | "SPIFFE / SPIRE" | 用于 OAuth token 签发的密码学服务身份 |
| 一致性测试套件 | "规范测试" | 用于 StreamableHTTP + 工具清单正确性的官方 MCP 测试套件 |

## 延伸阅读

- [模型上下文协议 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据、注册表
- [AAIF MCP 注册表规范](https://github.com/modelcontextprotocol/registry) — 2026 年注册表规范
- [AWS ECS 参考部署](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — 参考生产部署
- [Pinterest 内部 MCP 生态系统](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — 参考内部部署
- [Block `goose` MCP 使用模式](https://block.github.io/goose/) — 参考智能体消费模式
- [FastMCP](https://github.com/jlowin/fastmcp) — Python 服务器框架
- [开放策略代理](https://www.openpolicyagent.org/) — 策略引擎参考
- [SPIFFE / SPIRE](https://spiffe.io) — 工作负载身份参考