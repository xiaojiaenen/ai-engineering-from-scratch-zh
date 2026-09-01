# A2A —— Agent-to-Agent 协议

> 谷歌于 2025 年 4 月宣布 A2A；截至 2026 年 4 月，规范已发布至 https://a2a-protocol.org/latest/specification/，并获得 150+ 个组织支持。A2A 是 MCP（第 13 课）的水平互补协议：MCP 是纵向的（agent ↔ 工具），A2A 是对等协议（agent ↔ agent）。它定义了 Agent Card（发现）、带产物（text、结构化数据、video）的任务、不透明的任务生命周期以及认证。生产系统越来越倾向于将 MCP 与 A2A 配对使用。谷歌云在 2025-2026 年间将 A2A 支持集成到了 Vertex AI Agent Builder 中。

**类型：** 学习 + 构建
**语言：** Python（stdlib、`http.server`、`json`）
**前置知识：** Phase 16 · 04（Primitive Model）
**时间：** 约 75 分钟

## 问题

你的 agent 需要调用另一个系统的 agent。如何实现？你可以暴露一个 HTTP 端点，定义一个定制化的 JSON Schema，并祈祷对方能识别它。每一对 agent 都会形成一次定制集成。

A2A 正是这种调用的通用传输协议。标准化的发现机制、标准化的任务模型、标准化的传输、标准化的产物。类似于 HTTP+REST，但以 agent 作为一等公民。

## 概念

### 四大要素

**Agent Card**。位于 `/.well-known/agent.json` 的 JSON 文档，描述 agent：名称、技能、端点、支持的模态、认证要求。通过读取 Card 实现发现。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task。** 工作单元。一个具有生命周期的异步、有状态对象：`submitted → working → completed / failed / canceled`。客户端发送任务，轮询或订阅更新。

**Artifact。** 任务产生的结果类型。文本、结构化 JSON、图像、视频、音频。产物是类型化的，使得不同模态成为一等公民。

**不透明生命周期。** A2A 不规定远程 agent *如何* 解决问题。客户端只看到状态转换和产物；实现方可以自由使用任何框架。

### MCP/A2A 分工

- **MCP**（第 13 课）：agent ↔ 工具。agent 通过 JSON-RPC 读写工具服务器。默认为无状态。
- **A2A**：agent ↔ agent。对等协议；双方都是具有自身推理能力的 agent。

生产级多 agent 系统同时使用两者。一个 A2A 对端会调用其侧的 MCP 工具。这种分工使两类关注点保持清晰。

### 发现流程

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或使用流式传输：订阅 `/tasks/{id}/events` 以获取推送更新（SSE）。

### 认证

A2A 支持三种常见模式：

- **Bearer token** —— OAuth2 或opaque token。
- **mTLS** —— 双向 TLS；组织间相互证明身份。
- **签名请求** —— 对 payload 的 HMAC。

认证在 Agent Card 中声明；客户端发现并遵守。

### 截至 2026 年 4 月的 150+ 组织支持

企业采用推动了 A2A 的规模化。核心要点：A2A 成为企业 agent 系统跨越信任边界的标准方式。谷歌云推出了 Vertex AI Agent Builder 的 A2A 支持；Microsoft Agent Framework 支持 A2A；大多数主流框架（LangGraph、CrewAI、AutoGen）都提供了 A2A 适配器。

### A2A 的优势场景

- **跨组织调用。** A 公司的 agent 调用 B 公司的 agent。没有 A2A 时，每一对都是定制化契约。
- **异构框架。** LangGraph agent 调用 CrewAI agent 调用自定义 Python agent。A2A 进行标准化。
- **类型化产物。** 视频结果、结构化 JSON、音频——全部作为一等公民。
- **长运行任务。** 不透明生命周期 + 轮询使长达数小时的任务变得简单。

### A2A 的不足之处

- **延迟敏感的微调用。** A2A 的生命周期是异步的。亚毫秒级的 agent 到 agent 调用不适合；应使用直接 RPC。
- **紧耦合的同进程 agent。** 如果两个 agent 运行在同一个 Python 进程中，A2A 的 HTTP 往返属于杀鸡用牛刀。
- **小型团队。** 规范开销是真实存在的；仅内部使用的 agent 可能不需要这种正式性。

### A2A vs ACP、ANP、NLIP

2024-2026 年间涌现了若干相关规范：

- **ACP**（IBM/Linux Foundation）—— A2A 的前身，范围更窄。
- **ANP**（Agent Network Protocol）—— 强调对等发现，以去中心化为首要目标。
- **NLIP**（Ecma 自然语言交互协议，2025 年 12 月标准化）—— 自然语言内容类型。

截至 2026 年 4 月，A2A 是最被广泛采用的对等协议。参见 arXiv:2505.02279（Liu 等，"Agent Interoperability Protocols 综述"）了解对比。

```figure
sw-agent-card-discovery
```

## 构建它

`code/main.py` 使用 `http.server` 和 JSON 实现了一个最小化的 A2A 服务端和客户端。服务端：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 时返回产物。

客户端：

- 获取 Agent Card，
- 提交任务，
- 轮询直至完成，
- 读取产物。

运行：

```
python3 code/main.py
```

脚本在后台线程启动服务端，然后对其运行客户端。你会看到完整的流程：发现、提交、轮询、产物。

## 使用它

`outputs/skill-a2a-integrator.md` 设计了一个 A2A 集成方案：Agent Card 内容、任务 Schema、认证选择、流式传输与轮询的权衡。

## 交付

检查清单：

- **锁定规范版本。** A2A 仍在演进；Agent Card 应声明协议版本。
- **幂等任务创建。** 重复提交（网络重试）应产生一个任务。
- **产物 Schema。** 声明 agent 返回的形状；消费者应进行验证。
- **速率限制 + 认证。** A2A 面向公网；应用标准 Web 安全实践。
- **失败任务的死信队列。** 随时间检查模式，识别 recurring 失败类型。

## 练习

1. 运行 `code/main.py`。确认客户端发现服务端并收到正确的产物。
2. 为服务端添加第二个技能（如"summarize"）。更新 Agent Card。编写一个根据任务类型选择技能的客户端。
3. 实现 SSE 流式端点：`/tasks/{id}/events`，用于发射状态变更。客户端需要做什么不同的处理？
4. 阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。找出规范中强制要求但本示例未实现的三个功能。
5. 比较 A2A（Agent Card 发现）与 MCP（通过 `listTools` 的服务器端能力列示）。自描述 agent 与能力探测之间的权衡是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| A2A | "Agent-to-agent" | 用于 agent 跨系统调用的对等协议。谷歌 2025 年推出。 |
| Agent Card | "agent 的名片" | 位于 `/.well-known/agent.json` 的 JSON，描述技能、端点、认证。 |
| Task | "工作单元" | 具有生命周期的异步有状态对象；完成后产生产物。 |
| Artifact | "结果" | 类型化输出：文本、结构化 JSON、图像、视频、音频。一等媒体类型。 |
| Opaque lifecycle | "如何解决是 agent 自己的事" | 客户端看到状态转换；服务端可自由选择框架/工具。 |
| Discovery | "寻找 agent" | `GET /.well-known/agent.json` 返回 Card。 |
| MCP vs A2A | "工具 vs 对等体" | MCP：纵向 agent ↔ 工具。A2A：横向 agent ↔ agent。 |
| ACP / ANP / NLIP | "兄弟协议" | 相邻规范；A2A 是 2026 年采用最广的。 |

## 延伸阅读

- [A2A 规范](https://a2a-protocol.org/latest/specification/) —— 权威规范
- [谷歌开发者博客 —— A2A 公告](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) —— 2025 年 4 月发布帖
- [A2A GitHub 仓库](https://github.com/a2aproject/A2A) —— 参考实现与 SDK
- [Liu 等 —— Agent Interoperability Protocols 综述](https://arxiv.org/html/2505.02279v1) —— MCP、ACP、A2A、ANP 对比
