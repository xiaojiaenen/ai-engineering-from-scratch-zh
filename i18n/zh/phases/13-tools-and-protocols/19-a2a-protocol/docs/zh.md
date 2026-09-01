# A2A — 智能体间协议（Agent-to-Agent Protocol）

> MCP 是智能体对工具（agent-to-tool）。A2A（Agent2Agent）是智能体对智能体——一个开放协议，让基于不同框架构建的"不透明"智能体能够协作。2025年4月由 Google 发布，2025年6月捐赠给 Linux Foundation，2026年4月达到 v1.0 版本，拥有包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow 在内的 150+ 支持方。它吸收了 IBM 的 ACP 并新增了 AP2 支付扩展。本教程将介绍 Agent Card、Task 生命周期以及两种传输绑定方式。

**类型：** 构建
**语言：** Python（标准库，Agent Card + Task 驱动）
**前置条件：** Phase 13 · 06（MCP 基础），Phase 13 · 08（MCP 客户端）
**时间：** 约 75 分钟

## 学习目标

- 区分智能体对工具（MCP）与智能体对智能体（A2A）的用例。
- 在 `/.well-known/agent.json` 发布 Agent Card，包含技能和端点元数据。
- 掌握 Task 生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带 Parts（文本、文件、数据）的 Messages 和作为输出的 Artifacts。

## 问题

一个客服智能体需要委派报告撰写工作给一个专门的写作智能体。A2A 之前的选项：

- 自定义 REST API。可用，但每对组合都是临时方案。
- 共享代码库。要求两个智能体运行相同的框架。
- MCP。不适用：MCP 用于调用工具，而非两个智能体在保留各自内部推理不透明性的前提下协作。

A2A 填补了这个空白。它将交互建模为一个智能体向另一个发送 Task，带有生命周期、消息和工件。被调用智能体的内部状态保持不透明——调用方只能看到 task 状态转换和最终输出。

A2A 是"让跨框架的智能体互相通信"的协议。它不替代 MCP；两者是互补关系。

## 概念

### Agent Card

每个 A2A 合规的智能体在 `/.well-known/agent.json` 发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现基于 URL：获取卡片，了解 A2A 端点的 URL，枚举技能。

### 签名 Agent Card（AP2）

AP2 扩展（2025年9月）为 Agent Card 添加加密签名。发布者用 JWT 签署自己的卡片；消费者进行验证。防止身份冒充。

### Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (通过消息循环)
```

客户端通过 `tasks/send` 发起。被调用智能体转换状态；客户端通过 SSE 订阅状态更新或轮询。

### Messages 和 Parts

一条消息携带一个或多个 Parts：

- `text` — 纯文本内容。
- `file` — base64 编码的二进制数据，带 mimeType。
- `data` — 类型化的 JSON 负载（对被调用智能体的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

输出是 Artifacts，而非原始字符串。Artifact 是一个命名、类型化的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以分块流式传输。调用方负责累积。

### 两种传输绑定

1. **JSON-RPC over HTTP。** `/a2a` 端点，POST 用于请求，SSE 用于可选的流式传输。默认绑定。
2. **gRPC。** 适用于 gRPC 作为原生传输的企业环境。

两种绑定承载相同的逻辑消息结构。

### 不透明性保护

一个关键设计原则：被调用智能体的内部状态是不透明的。调用方只看到 task 状态和 artifacts。被调用智能体的思维链、工具调用、子智能体委派——全部不可见。这与 MCP 不同，MCP 中工具调用是透明的。

理由：A2A 使竞争对手能够协作而无需透露内部细节。A2A 可以是"调用这个客服智能体"，而调用方无需了解该智能体如何提供服务。

### 时间线

- **2025-04-09。** Google 发布 A2A。
- **2025-06-23。** 捐赠给 Linux Foundation。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（Agent Payments）发布。
- **2026-04。** v1.0 发布，拥有 150+ 支持组织。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|-----------|-----|-----|
| 用例 | 智能体对工具 | 智能体对智能体 |
| 透明度 | 透明的工具调用 | 不透明的内部推理 |
| 典型调用方 | Agent 运行时 | 另一个智能体 |
| 状态 | 工具调用结果 | 带生命周期的 Task |
| 授权 | OAuth 2.1（Phase 13 · 16） | JWT 签名的 Agent Card（AP2） |
| 传输 | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

当你想调用特定工具时使用 MCP。当你想将整个任务委托给另一个智能体时使用 A2A。许多生产系统同时使用两者：智能体用 MCP 作为其工具层，用 A2A 作为其协作层。

```figure
a2a-task-lifecycle
```

## 使用

`code/main.py` 实现了一个最小化的 A2A 驱动：一个研究智能体发布其卡片，一个写作智能体接收包含 PDF 和文本指令的 `tasks/send`，经历 working → input_required → working → completed 状态转换，并返回文本 artifact。全标准库实现；使用内存传输以聚焦于消息结构。

重点关注：

- Agent Card 的 JSON 结构。
- Task id 分配与状态转换。
- 混合类型 Parts 的 Messages。
- 任务中途的 input-required 分支。
- 完成时的 Artifact 返回。

## 交付

本教程产出 `outputs/skill-a2a-agent-spec.md`。给定一个新智能体（应能被其他智能体调用），该技能会生成 Agent Card JSON、技能 schema 和端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整的 Task 生命周期，包括被调用智能体请求澄清的 input-required 暂停阶段。

2. 添加签名 Agent Card。对卡片的规范 JSON 使用 HMAC 签名。编写验证器，确认其对篡改的卡片验证失败。

3. 实现任务流式传输：写作智能体通过 SSE 发出三个增量 artifact 块，调用方进行累积。

4. 设计一个封装 MCP 服务器的 A2A 智能体。将每个 MCP 工具映射到 A2A 技能。注意权衡——哪些不透明性会丧失？

5. 阅读 A2A v1.0 公告，找出截至 2026年4月尚无任何框架实现的特性。（提示：它与多跳任务委派相关。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent protocol" | 面向不透明智能体协作的开放协议 |
| Agent Card | "`.well-known/agent.json`" | 描述智能体技能和端点的已发布元数据 |
| Skill | "一个可调用单元" | 智能体支持的命名操作（类比 MCP 工具） |
| Task | "委派单位" | 带生命周期和最终 artifact 的工作项 |
| Message | "Task 输入" | 携带 Parts（文本、文件、数据） |
| Part | "类型化数据块" | `text` / `file` / `data` 消息元素 |
| Artifact | "Task 输出" | 命名、类型化的完成输出 |
| AP2 | "Agent Payments Protocol" | 用于信任和支付的签名 Agent Card 扩展 |
| Opacity | "黑盒协作" | 被调用智能体的内部对被调用方隐藏 |
| Input-required | "Task 暂停" | 智能体需要更多信息时的生命周期状态 |

## 进一步阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A 规范原文
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 引用实现和 SDK
- [Linux Foundation — A2A 发布新闻稿](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025年6月治理移交
- [Google Cloud — A2A 协议升级](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图和合作伙伴动态
- [Google Dev — A2A 1.0 里程碑](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 发布说明和向后兼容指南
