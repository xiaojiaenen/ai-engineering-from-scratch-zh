# 毕业项目 — 构建完整的工具生态系统

> 第13阶段教授了每个部分。本毕业项目将它们整合成一个生产级系统：一个包含工具、资源、提示、任务和UI的MCP服务器；边缘的OAuth 2.1；RBAC网关；多服务器客户端；A2A子代理调用；OTel追踪到收集器；CI中的工具投毒检测；以及一个AGENTS.md + SKILL.md包。学完之后，您将能为每个架构决策辩护。

**类型：** 构建
**语言：** Python（标准库，端到端生态系统工具链）
**前置要求：** 第13阶段 · 01至21
**时间：** 约120分钟

## 学习目标

- 组合一个暴露工具、资源、提示以及一个`ui://`应用的任务的MCP服务器。
- 用强制执行RBAC和固定哈希的OAuth 2.1网关前置该服务器。
- 编写一个使用OTel GenAI属性进行端到端追踪的多服务器客户端。
- 将部分工作负载委托给A2A子代理；验证不透明性得到保持。
- 使用AGENTS.md + SKILL.md打包整个堆栈，以便其他代理可以驱动它。

## 问题描述

交付"研究与报告"系统：

- 用户提问："总结2026年关于代理协议的三篇引用最多的arXiv论文。"
- 系统：通过MCP搜索arXiv；通过A2A将论文总结委托给专门的写作代理；聚合结果；将交互式报告渲染为MCP Apps `ui://`资源；将每一步记录到OTel。

第13阶段的所有原语都会出现。这不是一个玩具——2026年由Anthropic（Claude Research产品）、OpenAI（带有Apps SDK的GPT）以及第三方交付的生产研究助手系统正是这种形态。

## 核心概念

### 架构

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### 追踪层次

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

一个追踪ID。每个跨度都具有正确的`gen_ai.*`属性。

### 安全态势

- OAuth 2.1 + PKCE，资源指示器将受众固定到网关。
- 网关持有上游凭据；用户永远看不到它们。
- RBAC：`alice` 拥有 `research:read`，`research:write`，可以调用所有工具。`bob` 拥有 `research:read`，不能调用 `generate_report`。
- 固定的描述符清单：任何工具哈希发生变更的服务器都将被丢弃。
- 双重规则审计：没有工具将不受信任的输入、敏感数据和后果性操作组合在一起。

### 渲染

最终的`generate_report`任务返回内容块加上一个`ui://report/current`资源。客户端的宿主（Claude Desktop等）在沙盒iframe中渲染交互式仪表板。该仪表板包含排序的论文列表、引用计数，以及一个为用户点击的任何论文调用`host.callTool('summarize_paper', {arxiv_id})`的按钮。

### 打包

整个系统作为如下结构交付：

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

用户通过`docker compose up`部署。Claude Code、Cursor、Codex和opencode的用户可以通过调用`run-research`技能来驱动系统。

### 第13阶段各课的贡献

| 课程 | 毕业项目使用的部分 |
|--------|------------------------|
| 01-05 | 工具接口、提供者可移植性、并行调用、模式、linting |
| 06-10 | MCP原语、服务器、客户端、传输、资源 + 提示 |
| 11-14 | 采样、根 + 引导、异步任务、`ui://`应用 |
| 15-17 | 工具投毒、OAuth 2.1、网关 + 注册表 |
| 18 | A2A子代理委托 |
| 19 | OTel GenAI追踪 |
| 20 | 面向LLM层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 使用它

`code/main.py` 将前几课的模式整合成一个可运行的演示。全部使用标准库，全部在进程内，以便您可以端到端地阅读它。它为研究与报告场景运行完整的流程：与网关握手，模拟的OAuth 2.1，合并的tools/list，将generate_report作为任务，对写作代理的A2A调用，返回的ui://资源，发出的OTel跨度。

需要关注的地方：

- 跨每个跃点的单个追踪ID。
- 网关策略阻止第二个用户进行写入。
- 任务生命周期从进行中 → 已完成，并返回文本和ui://内容。
- A2A调用的内部状态对协调器是不透明的。
- AGENTS.md和SKILL.md是另一个代理重现工作流所需的唯一文件。

## 交付它

本课程产出`outputs/skill-ecosystem-blueprint.md`。给定一个产品需求（研究、总结、自动化），该技能将产生完整的架构：使用哪些MCP原语，哪些网关控制，哪些A2A调用，哪些遥测，哪些打包。

## 练习

1. 运行`code/main.py`。注意单一的追踪ID以及跨度如何嵌套。计算演示涉及了第13阶段的多少原语。
2. 扩展演示：添加第二个后端MCP服务器（例如`bibliography`），并确认网关将其工具合并到同一命名空间。
3. 用一个在子进程中运行的真实代理替换假的A2A写作代理。使用第19课的工具链。
4. 在协调器和LLM之间的路由网关中添加PII脱敏步骤。确认用户查询中的电子邮件地址被清除。
5. 为维护此系统的队友编写一个AGENTS.md。它应该在五分钟内读完，并为他们在Cursor或Codex中驱动毕业项目提供所需的一切信息。

## 关键术语

| 术语 | 人们说什么 | 实际含义 |
|------|----------------|------------------------|
| 毕业项目 | "第13阶段集成演示" | 使用每个原语的端到端系统 |
| 研究与报告 | "该场景" | 搜索、总结、渲染模式 |
| 生态系统 | "所有部分组合在一起" | 服务器 + 客户端 + 网关 + 子代理 + 遥测 + 打包 |
| 追踪层次 | "单一追踪ID" | 每个跃点的跨度共享该追踪；通过跨度ID确定父子关系 |
| 网关颁发的令牌 | "传递性认证" | 客户端只看到网关的令牌；网关持有上游凭据 |
| 合并的命名空间 | "所有工具在一个扁平列表中" | 在网关处进行多服务器合并，冲突时添加前缀 |
| 不透明边界 | "A2A调用隐藏内部细节" | 子代理的推理对协调器不可见 |
| 三层堆栈 | "AGENTS.md + SKILL.md + MCP" | 项目上下文 + 工作流 + 工具 |
| 纵深防御 | "多层安全" | 固定哈希、OAuth、RBAC、双重规则、审计日志 |
| 规范符合矩阵 | "我们交付了规范所要求的" | 将交付物映射到2025-11-25要求的检查清单 |

## 扩展阅读

- [MCP — 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 综合参考
- [MCP博客 — 2026年路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议的发展方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考
- [OpenTelemetry — GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范的追踪约定
- [Anthropic — Claude Agent SDK 概述](https://code.claude.com/docs/en/agent-sdk/overview) — 生产级代理运行时模式