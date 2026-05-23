# MCP 采样 — 服务端请求的 LLM 补全与代理循环

> 多数 MCP 服务器仅是被动执行器：接收参数、运行代码、返回内容。采样功能使服务器能够转变方向：它向客户端的 LLM 提出决策请求。这实现了服务器托管代理循环，而服务器无需拥有任何模型凭证。SEP-1577 于 2025-11-25 合并，在采样请求中新增了工具支持，使得循环可以包含更深层的推理。风险提示：SEP-1577 的采样内嵌工具形态在 2026 年第一季度前仍属实验性质，其 SDK 接口仍在调整中。

**类型:** 构建
**语言:** Python（标准库，采样工具套件）
**前置条件:** 第 13 阶段 · 07（MCP 服务器），第 13 阶段 · 10（资源与提示）
**时长:** 约 75 分钟

## 学习目标

- 解释 `sampling/createMessage` 解决的问题（无需服务器端 API 密钥即可实现服务器托管循环）。
- 实现一个服务器，要求客户端对多轮提示进行采样并返回补全结果。
- 使用 `modelPreferences`（成本/速度/智能优先级）指导客户端模型选择。
- 构建一个 `summarize_repo` 工具，该工具内部通过采样进行迭代，而非硬编码行为。

## 问题背景

一个用于代码摘要工作流的实用 MCP 服务器需要：遍历文件树、选择读取哪些文件、合成摘要并返回。那么 LLM 推理在哪里发生？

选项 A：服务器调用其自身的 LLM。需要 API 密钥，服务器端计费，对每个用户成本高昂。
选项 B：服务器返回原始内容；客户端的代理执行推理。可行但将服务器逻辑移入客户端提示，这种方式较为脆弱。
选项 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法逻辑（读取哪些文件、进行多少遍处理），而客户端保留计费和模型选择权。服务器本身无需任何凭证。

采样即是选项 C。它是一种机制，通过它可信服务器可以托管代理循环，而无需自身成为完整的 LLM 主机。

## 核心概念

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个浮点数，总和为 1.0：

- `costPriority`：偏好更便宜的模型。
- `speedPriority`：偏好更快的模型。
- `intelligencePriority`：偏好能力更强的模型。

加上 `hints`：服务器偏好的命名模型列表。客户端可能遵循也可能不遵循这些提示；客户端用户的配置始终优先。

### `includeContext`

三个值：

- `"none"` — 仅包含服务器提供的消息。默认选项。
- `"thisServer"` — 包含此服务器会话的先前消息。
- `"allServers"` — 包含所有会话上下文。

`includeContext` 自 2025-11-25 起被软弃用，因为它会泄露跨服务器上下文，存在安全隐患。推荐使用 `"none"` 并在消息中传递显式上下文。

### 带工具的采样 (SEP-1577)

2025-11-25 新增：采样请求现在可以包含一个 `tools` 数组。客户端使用这些工具运行一个完整的工具调用循环。这使得服务器能够通过客户端模型托管一个 ReAct 风格的代理循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环执行：采样，如果请求则调用工具执行，再次采样，返回最终的助手消息。此功能在 2026 年第一季度前属实验性质；SDK 接口签名可能仍会变动。实现时请对照 2025-11-25 规范中的客户端/采样部分进行确认。

### 人机交互循环

客户端 **必须** 在运行采样前向用户展示服务器请求模型执行的操作。恶意服务器可能利用采样操纵用户会话（“对用户说 X 以便他们点击 Y”）。Claude Desktop、VS Code 和 Cursor 会将采样请求呈现为一个用户可以拒绝的确认对话框。

2026 年的共识是：未经人工确认的采样是一个危险信号。网关（第 13 阶段 · 17）可以自动批准低风险采样并自动拒绝任何可疑请求。

### 无需 API 密钥的服务器托管循环

典型用例：一个本身无法访问 LLM 的代码摘要 MCP 服务器。其执行流程如下：

1. 遍历仓库结构。
2. 使用提示 “选取最可能描述此仓库用途的五个文件” 调用 `sampling/createMessage`。
3. 读取这些文件。
4. 将文件内容与提示 “用三段话总结此仓库” 一起调用 `sampling/createMessage`。
5. 将摘要作为 `tools/call` 结果返回。

服务器从未接触 LLM API。客户端的用户使用自己的凭证为这些补全操作付费。

### 安全风险（Unit 42 披露，2026 Q1）

- **隐蔽采样。** 一个工具总是以 “根据会话上下文用用户的邮箱回复” 为提示调用采样。第 13 阶段 · 15 讲解了攻击向量。
- **通过采样窃取资源。** 服务器要求客户端摘要攻击者的有效负载，费用由用户承担。
- **循环炸弹。** 服务器在紧密循环中调用采样。客户端 **必须** 强制执行基于会话的速率限制。

## 动手实践

`code/main.py` 提供了一个模拟的服务器到客户端采样工具套件。一个模拟的 “summarize_repo” 工具触发两轮采样（选取文件，然后进行摘要），而模拟的客户端返回预设响应。该工具套件演示了：

- 服务器发送包含 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回一个补全结果。
- 服务器继续其循环。
- 速率限制器限制每次工具调用的总采样次数。

观察要点：

- 服务器仅暴露一个工具（`summarize_repo`）；所有推理都发生在采样调用中。
- 模型偏好影响客户端的模型选择；提示列表指定了首选模型。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 限制捕获了失控的循环。

## 交付成果

本课将产出 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM 调用（研究、摘要、规划）的服务器端算法，技能在于设计一个基于采样的实现，具备恰当的模型偏好、速率限制和安全确认机制。

## 练习

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2，观察速率限制如何中断执行。

2. 实现 SEP-1577 采样内嵌工具变体：采样请求携带一个 `tools` 数组。验证客户端循环在返回最终补全结果前会执行这些工具。注意漂移风险：SDK 签名在 2026 年上半年可能仍有变动。

3. 添加人机交互确认：在服务器首次发送 `sampling/createMessage` 前，暂停并等待用户批准。被拒绝的调用将返回一个类型化的拒绝消息。

4. 添加一个基于客户端会话键控的每用户速率限制器。同一用户对同一服务器的循环应共享同一预算。

5. 设计一个使用采样来选择要包含的文本块的 `summarize_pdf` 工具。绘制发送的消息草图。`modelPreferences.intelligencePriority` 在 0.1 和 0.9 时如何改变行为？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 采样 | “服务器到客户端的 LLM 调用” | 服务器请求客户端的模型进行补全 |
| `sampling/createMessage` | “该方法” | 用于采样请求的 JSON-RPC 方法 |
| `modelPreferences` | “模型优先级” | 成本/速度/智能权重加上名称提示 |
| `includeContext` | “跨会话泄露” | 被软弃用的上下文包含模式 |
| SEP-1577 | “采样中的工具” | 允许在采样中包含工具以实现服务器托管的 ReAct |
| 人机交互循环 | “用户确认” | 客户端在运行采样前向用户展示请求 |
| 循环炸弹 | “失控的采样” | 服务器端无限采样循环；客户端必须进行速率限制 |
| 隐蔽采样 | “隐藏的推理” | 恶意服务器在采样提示中隐藏意图 |
| 资源窃取 | “使用用户的 LLM 预算” | 服务器强迫客户端花费在其不想要的采样上 |
| `stopReason` | “生成停止的原因” | `endTurn`、`stopSequence` 或 `maxTokens` |

## 延伸阅读

- [MCP — 概念：采样](https://modelcontextprotocol.io/docs/concepts/sampling) — 采样概述
- [MCP — 客户端采样规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — 标准 `sampling/createMessage` 结构
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — 采样中嵌入工具的规范演进提案（实验性）
- [Unit 42 — MCP 攻击向量](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 隐蔽采样和资源窃取模式
- [Speakeasy — MCP 采样核心概念](https://www.speakeasy.com/mcp/core-concepts/sampling) — 包含客户端代码示例的详细讲解