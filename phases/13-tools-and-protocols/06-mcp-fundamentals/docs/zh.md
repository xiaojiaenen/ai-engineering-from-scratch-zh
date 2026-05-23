# MCP 基础 — 原语、生命周期、JSON-RPC 基础

> 在MCP之前，每一次集成都是独特的。模型上下文协议最初由Anthropic于2024年11月发布，现由Linux基金会下的代理人工智能基金会管理，它标准化了发现和调用过程，使得任何客户端都能与任何服务器通信。2025-11-25规范定义了六个原语（三个服务器端，三个客户端）、一个三阶段生命周期和JSON-RPC 2.0传输格式。掌握这些内容，本章其余的MCP部分就变得易于理解。

**类型：** 学习
**语言：** Python (标准库，JSON-RPC解析器)
**前置条件：** 阶段13 · 01至05（工具接口与函数调用）
**时间：** 约45分钟

## 学习目标

- 列出所有六个MCP原语（服务器端的工具、资源、提示；客户端的根、采样、引出）并各举一个用例。
- 探讨三阶段生命周期（初始化、操作、关闭），并说明每个阶段谁发送哪条消息。
- 解析和生成JSON-RPC 2.0请求、响应和通知信封。
- 解释 `initialize` 处的能力协商是什么，以及没有它会导致什么问题。

## 问题背景

在MCP之前，每个使用工具的智能体都有自己的协议。Cursor有一个形似MCP但不兼容的工具系统。Claude Desktop搭载了另一个不同的系统。VS Code的Copilot扩展则有第三种。一个构建了“Postgres查询”工具的团队需要将同一个工具实现三次，每次针对不同主机的API。重用它意味着复制代码。

结果就是定制集成的寒武纪式爆发，生态系统发展速度触顶。

MCP通过标准化传输格式解决了这个问题。单个MCP服务器可以在所有MCP客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，截至2026年4月已有300多个客户端。每月1.1亿次SDK下载。10,000多个公共服务器。Linux基金会于2025年12月在新的代理人工智能基金会下接管了管理权。

本阶段使用的规范修订版是**2025-11-25**。它添加了异步任务（SEP-1686）、URL模式引出（SEP-1036）、带工具的采样（SEP-1577）、增量范围同意（SEP-835）以及OAuth 2.1资源指示符语义。阶段13 · 09至16涵盖了这些扩展。本课程止于基础部分。

## 核心概念

### 三个服务器端原语

1.  **工具。** 可调用的操作。与阶段13 · 01中的四步循环相同。
2.  **资源。** 暴露的数据。只读内容，可通过URI寻址：`file:///path`、`db://query/...`、自定义方案。
3.  **提示。** 可重用的模板。宿主UI中的斜杠命令；服务器提供模板，客户端填充参数。

### 三个客户端原语

4.  **根。** 服务器被允许访问的URI集合。客户端声明它们；服务器遵守。
5.  **采样。** 服务器请求客户端的模型执行补全。实现了无需服务器端API密钥的服务器托管智能体循环。
6.  **引出。** 服务器在操作过程中请求客户端的用户提供结构化输入。表单或URL（SEP-1036）。

MCP中的每一项能力都恰好属于这六个之一。阶段13 · 10至14会深入探讨每一个。

### 传输格式：JSON-RPC 2.0

每条消息都是一个包含以下字段的JSON对象：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` — 没有 `id`，不期望响应。

基础规范约有15个方法，按原语分组。重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`，`tools/call`
- `resources/list`，`resources/read`，`resources/subscribe`
- `prompts/list`，`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`，`notifications/resources/updated`，`notifications/progress`

### 三阶段生命周期

**阶段1：初始化。**

客户端发送包含其 `capabilities` 和 `clientInfo` 的 `initialize`。服务器用其自身的 `capabilities`、`serverInfo` 以及它所遵循的规范版本来响应。当客户端消化了响应后，会发送 `notifications/initialized`。此后，双方可以根据协商的能力发送请求。

**阶段2：操作。**

双向进行。客户端调用 `tools/list` 进行发现，然后调用 `tools/call` 进行调用。如果服务器声明了该能力，则可以发送 `sampling/createMessage`。当其工具集发生变化时，服务器可以发送 `notifications/tools/list_changed`。当用户更改根范围时，客户端可以发送 `notifications/roots/list_changed`。

**阶段3：关闭。**

任一方关闭传输连接。MCP中没有结构化的关闭方法；传输层（stdio或可流式HTTP，阶段13 · 09）携带连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 是合同。一个服务器的示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自己的能力来同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端没有声明 `sampling`，服务器不得调用 `sampling/createMessage`。对称地：如果服务器没有声明 `resources.subscribe`，客户端不得尝试订阅。

这正是防止生态系统漂移的原因。一个不支持采样的客户端仍然是有效的MCP客户端；一个不调用 `sampling` 的服务器仍然是有效的MCP服务器。它们只是不一起使用该功能。

### 结构化内容与错误形式

`tools/call` 返回一个包含类型化块的 `content` 数组：`text`、`image`、`resource`。阶段13 · 14添加了MCP应用程序（`ui://` 交互式UI）到该列表中。

错误使用JSON-RPC错误代码。规范定义的补充有：`-32002` "资源未找到"，`-32603` "内部错误"，以及作为 `error.data` 的MCP特定错误数据。

### 客户端能力与工具调用细节

一个常见的混淆点：`capabilities.tools` 指的是客户端是否支持工具列表更改通知。客户端**是否**会调用特定工具是运行时由其模型驱动的决策，而不是一个能力标志。能力标志是规范层面的合同。模型的选择是正交的。

### 为什么是JSON-RPC而不是REST？

JSON-RPC 2.0（2010年）是一种轻量级双向协议。REST是客户端发起的。MCP需要服务器发起的消息（采样、通知），因此具有对称请求/响应形式的JSON-RPC是一个自然的选择。JSON-RPC也可以干净地组合到stdio和WebSocket/可流式HTTP上，而无需重新发明HTTP的请求形式。

## 动手实践

`code/main.py` 提供了一个最小的JSON-RPC 2.0解析器和生成器，然后手动演示 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。没有真正的传输层；只有消息形式。将其与“进一步阅读”中链接的规范进行比较以验证每个信封。

需要关注的地方：

- `initialize` 双向声明能力；响应包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目包含 `name`、`description`、`inputSchema`。
- `tools/call` 使用了 `params.name` 和 `params.arguments`。
- 响应 `content` 是一个 `{type, text}` 块的数组。

## 提交成果

本课程产出 `outputs/skill-mcp-handshake-tracer.md`。给定一个MCP客户端-服务器交互的pcap风格记录，该技能可以为每条消息标注它属于哪个原语、哪个生命周期阶段以及依赖哪个能力。

## 练习

1.  运行 `code/main.py`。找出能力协商发生的那一行，并描述如果服务器没有声明 `tools.listChanged` 会发生什么变化。

2.  扩展解析器以处理 `notifications/progress`。消息形式：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在一个长时间运行的 `tools/call` 过程中发出它，并确认客户端处理程序会显示一个进度条。

3.  从头到尾阅读MCP 2025-11-25规范——整个文档约80页。找出大多数服务器**不需要**的一个能力标志。提示：它与资源订阅相关。

4.  在纸上勾勒出一个假设的“定时任务”功能会属于哪个原语。（提示：服务器希望客户端在预定时间调用它。目前六个原语都不合适。）MCP的2026路线图中有一份关于此的SEP草案。

5.  解析来自GitHub上一个公开MCP服务器的一个会话日志。统计请求、响应和通知消息的数量。计算生命周期流量与操作流量的比例。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| MCP | "模型上下文协议" | 用于模型到工具发现和调用的开放协议 |
| 服务器端原语 | "服务器暴露什么" | 工具（操作）、资源（数据）、提示（模板） |
| 客户端原语 | "客户端让服务器使用什么" | 根（范围）、采样（LLM回调）、引出（用户输入） |
| JSON-RPC 2.0 | "传输格式" | 对称的请求/响应/通知信封 |
| `initialize` 握手 | "能力协商" | 第一对消息；服务器和客户端声明它们支持的功能 |
| `tools/list` | "发现" | 客户端向服务器请求其当前工具集 |
| `tools/call` | "调用" | 客户端请求服务器使用参数执行一个工具 |
| `notifications/*_changed` | "变更事件" | 服务器告知客户端其原语列表已更改 |
| 内容块 | "类型化结果" | 工具结果中的 `{type: "text" | "image" | "resource" | "ui_resource"}` |
| SEP | "规范演进提案" | 命名的草案提案（例如，SEP-1686用于异步任务） |

## 进一步阅读

- [模型上下文协议 — 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 规范权威文档
- [模型上下文协议 — 架构概念](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语心智模型
- [Anthropic — 介绍模型上下文协议](https://www.anthropic.com/news/model-context-protocol) — 2024年11月发布文章
- [MCP博客 — MCP一周年](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾及2025-11-25规范变更
- [WorkOS — MCP 2025-11-25规范更新](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835和1724的摘要