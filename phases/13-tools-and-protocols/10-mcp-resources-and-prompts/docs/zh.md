# MCP 资源与提示 —— 超越工具的上下文暴露

> 工具获得了 MCP 90% 的关注。另外两个服务器原语解决不同的问题：资源暴露数据用于读取；提示暴露可复用的模板作为斜杠命令。许多服务器应该使用资源，而不是将读取包装在工具中；使用提示，而不是在客户端提示中硬编码工作流程。本课将阐述决策规则，并解析 `resources/*` 和 `prompts/*` 消息。

**类型：** 构建
**语言：** Python（标准库，资源 + 提示处理器）
**前置条件：** 阶段 13 · 07 (MCP 服务器)
**时间：** 约 45 分钟

## 学习目标

- 针对给定领域，决定将某个能力暴露为工具、资源还是提示。
- 实现 `resources/list`、`resources/read`、`resources/subscribe` 并处理 `notifications/resources/updated`。
- 实现 `prompts/list` 和带参数模板的 `prompts/get`。
- 识别宿主何时将提示作为斜杠命令显示，何时作为自动注入的上下文。

## 问题

一个笔记应用的朴素 MCP 服务器将所有内容暴露为工具：`notes_read`、`notes_list`、`notes_search`。这将每次数据访问都包装在模型驱动的工具调用中。后果：

- 模型必须对每个可能受益于上下文的查询决定是否调用 `notes_read`。
- 只读内容无法被订阅或流式传输到宿主的侧边栏。
- 客户端 UI（如 Claude Desktop 的资源附件面板、Cursor 的“包含文件”选择器）无法呈现该数据。

正确的划分方式：将数据暴露为资源；将变更性或计算性操作暴露为工具；将可复用的多步骤工作流暴露为提示。每种原语都有其用户体验特性和访问模式。

## 概念

### 工具、资源与提示 —— 决策规则

| 能力 | 原语 |
|------------|-----------|
| 用户想要搜索、过滤或转换数据 | 工具 |
| 用户希望宿主将此数据作为上下文包含 | 资源 |
| 用户想要一个可以重新运行的模板化工作流 | 提示 |

指导原则：如果模型在每个相关查询中调用它都有益，则它是一个工具。如果用户将其附加到对话中会受益，则它是一个资源。如果整个多步骤工作流是用户希望复用的单元，则它是一个提示。

### 资源

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接受 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的内容：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义方案）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 支持文本和二进制。二进制使用 `blob` 作为 base64 编码的字符串，加上一个 `mimeType`。

### 资源订阅

在能力中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。当资源发生变化时，服务器发送 `notifications/resources/updated {uri}`。客户端重新读取。

用例：一个笔记服务器，其资源是磁盘上的文件；文件监视器触发更新通知；当在宿主外部编辑文件时，Claude Desktop 会将该文件重新拉入上下文。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 让你暴露一个参数化的 URI 模式：`notes://{id}`，其中 `id` 作为补全目标。客户端可以在资源选择器中自动补全 ID。

### 提示

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接受 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

提示是一个模板，它填充为宿主馈送其模型的消息列表。例如，一个 `code_review` 提示接受一个 `file_path` 参数，并返回一个三消息序列：一个系统消息、一个包含文件正文的用户消息，以及一个带有推理模板的助手启动消息。

### 宿主与提示

Claude Desktop、VS Code 和 Cursor 在聊天 UI 中将提示作为斜杠命令暴露。用户输入 `/code_review` 并从表单中选择参数。服务器的提示是“用户快捷方式”与“发送给模型的完整提示”之间的契约。

并非所有客户端都支持提示——请检查能力协商。一个声明了提示能力但客户端不支持提示的服务器，将不会看到斜杠命令。

### “列表已变更”通知

当集合发生变更时，资源和提示都会发出 `notifications/list_changed`。一个刚导入了 20 条新笔记的笔记服务器会发出 `notifications/resources/list_changed`；客户端会重新调用 `resources/list` 以获取新增内容。

### 内容类型约定

对于文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
对于二进制：`image/png`、`application/pdf`，加上 `blob` 字段。
对于 MCP 应用（第 14 课）：`text/html;profile=mcp-app` 在 `ui://` URI 中。

### 动态资源

资源 URI 不必对应于静态文件。`notes://recent` 每次读取都可以返回最新的五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可以自由地动态计算内容。

规则：如果客户端可以通过 URI 缓存，则 URI 必须稳定。如果计算是一次性的，URI 应包含时间戳或 nonce，以使客户端缓存不会过期。

### 订阅与轮询

支持订阅的客户端通过 `notifications/resources/updated` 接收服务器推送。预订阅客户端或不支持它的宿主通过重新读取来轮询。两者都符合规范。服务器的能力声明告诉客户端它支持哪种方式。

订阅的代价：服务器上需要维护每个会话的状态（谁订阅了什么）。保持订阅集有界；断开连接的客户端应超时。

### 提示与系统提示

MCP 中的提示不是系统提示。宿主的系统提示（其自身的操作指令）和 MCP 提示（由用户调用的服务器提供的模板）并存。一个行为良好的客户端从不允​​许服务器提示覆盖其自身的系统提示；它将它们分层。

## 使用它

`code/main.py` 扩展了第 07 课的笔记服务器，增加了：

- 每个笔记的资源（`notes://note-1` 等），支持 `resources/subscribe`。
- 一个渲染为三消息模板的 `review_note` 提示。
- 一个文件监视器模拟，当笔记被修改时发出 `notifications/resources/updated`。
- 一个始终返回最新五条笔记的 `notes://recent` 动态资源。

运行演示以查看完整流程。

## 交付

本课产生 `outputs/skill-primitive-splitter.md`。给定一个提议的 MCP 服务器，该技能将每个能力分类为工具/资源/提示，并附带理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发一个笔记编辑并验证 `notifications/resources/updated` 事件是否触发。
2. 添加一个 `resources/list_changed` 发射器：当创建新笔记时，发送通知以便客户端重新发现。
3. 为 GitHub MCP 服务器设计三个提示：`summarize_pr`、`triage_issue`、`release_notes`。每个都带有参数模式。提示正文应无需进一步编辑即可运行。
4. 选取第 07 课服务器中的一个现有工具，并判断它应该保留为工具还是应该拆分为资源加工具对。用一句话说明理由。
5. 阅读规范的 `server/resources` 和 `server/prompts` 部分。识别 `resources/read` 中很少被填充但规范支持的字段。提示：查看资源内容上的 `_meta`。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| 资源 | "暴露的数据" | 宿主可以读取的 URI 可寻址内容 |
| 资源 URI | "数据指针" | 方案前缀标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | "监视变更" | 客户端选择加入的、针对特定 URI 的服务器推送更新 |
| `notifications/resources/updated` | "资源已变更" | 通知客户端已订阅的资源有新内容 |
| 资源模板 | "参数化 URI" | 带补全提示的 URI 模式，供宿主选择器使用 |
| 提示 | "斜杠命令模板" | 带有参数槽的命名多消息模板 |
| 提示参数 | "模板输入" | 渲染前宿主收集的类型化参数 |
| `prompts/get` | "渲染模板" | 服务器返回填充后的消息列表 |
| 内容块 | "类型化块" | `{type: text | image | resource | ui_resource}` |
| 斜杠命令 UX | "用户快捷方式" | 宿主将提示作为以 `/` 开头的命令呈现 |

## 扩展阅读

- [MCP — 概念：资源](https://modelcontextprotocol.io/docs/concepts/resources) — 资源 URI、订阅和模板
- [MCP — 概念：提示](https://modelcontextprotocol.io/docs/concepts/prompts) — 提示模板和斜杠命令集成
- [MCP — 服务器资源规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整的 `resources/*` 消息参考
- [MCP — 服务器提示规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整的 `prompts/*` 消息参考
- [MCP — 协议信息网站：资源](https://modelcontextprotocol.info/docs/concepts/resources/) — 对官方文档进行扩展的社区指南