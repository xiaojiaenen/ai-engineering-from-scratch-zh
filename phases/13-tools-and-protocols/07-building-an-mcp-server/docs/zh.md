# 构建 MCP 服务器 — Python + TypeScript SDK

> 多数 MCP 教程仅展示标准输入输出（stdio）的 "Hello World" 示例。而真正的服务器需要暴露工具、资源与提示，处理能力协商，发出结构化错误，并且在两个 SDK 之间行为一致。本课程将从头到尾构建一个笔记服务器：使用标准库的 stdio 传输、JSON-RPC 调度、三种服务器原语，以及一种纯函数风格的代码，这种代码可以直接迁移到 Python SDK 的 FastMCP 或者当你进阶时迁移到 TypeScript SDK。

**类型：** 构建
**语言：** Python（标准库，stdio MCP 服务器）
**先决条件：** 阶段 13 · 06（MCP 基础）
**时间：** 约 75 分钟

## 学习目标

- 实现 `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, 和 `prompts/get` 方法。
- 编写一个调度循环，从 stdin 读取 JSON-RPC 消息并向 stdout 写入响应。
- 根据 JSON-RPC 2.0 规范和 MCP 附加的错误码，发出结构化错误响应。
- 将标准库实现迁移到 FastMCP（Python SDK）或 TypeScript SDK，而无需重写工具逻辑。

## 问题背景

在能够使用远程传输（阶段 13 · 09）或认证层（阶段 13 · 16）之前，你需要一个干净的本地服务器。本地指的是标准输入输出（stdio）：服务器由客户端作为子进程启动，消息通过 stdin/stdout 以换行符分隔进行传输。

2025-11-25 规范规定，stdio 消息编码为 JSON 对象，并使用明确的 `\n` 分隔符。这里没有 SSE；SSE 是旧的远程模式，将在 2026 年中期被淘汰（Atlassian 的 Rovo MCP 服务器已于 2026 年 6 月 30 日弃用；Keboola 于 2026 年 4 月 1 日弃用）。对于 stdio，每行一个 JSON 对象就是完整的线路格式。

笔记服务器是一个很好的示例，因为它涵盖了所有三种服务器原语。工具执行变更操作（`notes_create`）。资源暴露数据（`notes://{id}`）。提示提供模板（`review_note`）。本课程的结构可以推广到任何领域。

## 概念详解

### 调度循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何非 JSON-RPC 封装的内容。调试日志应输出到 stderr。
- 每个请求必须匹配一个带有相同 `id` 的响应。
- 通知不应被回复。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

仅声明你支持的功能。客户端依赖能力集来决定启用哪些功能。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，其中每个条目具有 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

内容块有类型。最常见的是：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形式。协议级错误（未知方法、参数错误）是 JSON-RPC 错误。工具级错误（调用有效但工具失败）则作为 `{content: [...], isError: true}` 返回。这样模型可以在其上下文中看到失败信息。

### 实现资源

资源在设计上是只读的。`resources/list` 返回一个清单；`resources/read` 返回具体内容。URI 可以是 `file://...`、`http://...` 或自定义方案，例如 `notes://`。

当你将数据作为资源（而不是工具）暴露时：

- 模型不会“调用”它；客户端可以根据用户请求将其注入上下文。
- 订阅使服务器能够在资源更改时推送更新（阶段 13 · 10）。
- 阶段 13 · 14 通过 `ui://` 扩展了这一点，用于交互式资源。

### 实现提示

提示是带有命名参数的模板。主机将其显示为斜杠命令。一个 `review_note` 提示可能接受一个 `note_id` 参数，并生成一个多消息提示模板，客户端会将其提供给其模型。

### Stdio 传输的细节

- 换行符分隔的 JSON。没有长度前缀的帧。
- 不要缓冲。每次写入后执行 `sys.stdout.flush()`。
- 客户端控制生命周期。当 stdin 关闭（EOF）时，干净地退出。
- 不要静默处理 SIGPIPE；记录日志并退出。

### 注解

每个工具都可以携带 `annotations` 来描述安全属性：

- `readOnlyHint: true` — 纯读取，安全可重试。
- `destructiveHint: true` — 不可逆副作用；客户端应确认。
- `idempotentHint: true` — 相同输入产生相同输出。
- `openWorldHint: true` — 与外部系统交互。

客户端使用这些来决定用户体验（确认对话框、状态指示器）和路由（阶段 13 · 17）。

### 迁移路径

`code/main.py` 中的标准库服务器约有 180 行代码。FastMCP（Python）将同样的逻辑折叠为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 具有等效的结构。当你准备好时，迁移路径是即插即用的；概念（能力、调度、内容块）是相同的。

## 使用说明

`code/main.py` 是一个完整的笔记 MCP 服务器，基于 stdio，仅使用标准库。它处理三个工具（`notes_list`、`notes_search`、`notes_create`）的 `initialize`、`tools/list`、`tools/call`，每篇笔记的 `resources/list` 和 `resources/read`，以及一个 `review_note` 提示。你可以通过管道传递 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

需要关注的地方：

- 调度器是一个由方法名作为键的 `dict[str, Callable]`。
- 每个工具执行器返回一个内容块列表，而不是一个裸字符串。
- 当执行器抛出异常时，`isError: true` 会被设置。

## 成果产出

本课程生成 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该技能可以搭建一个具有正确工具/资源/提示划分和 SDK 迁移路径的 MCP 服务器。

## 练习

1.  运行 `code/main.py` 并用手工构建的 JSON-RPC 消息驱动它。先执行 `notes_create`，然后执行 `resources/read` 以检索新笔记。

2.  添加一个带有 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端是否会显示确认对话框（这需要一个真实的主机；Claude Desktop 可以工作）。

3.  实现 `resources/subscribe`，以便服务器在每次修改笔记时推送 `notifications/resources/updated`。添加一个保活任务。

4.  将服务器移植到 FastMCP。Python 文件应缩减到 80 行以下。线路行为必须相同；使用相同的 JSON-RPC 测试套件进行验证。

5.  阅读规范中的 `server/tools` 部分，并找出一个本课程服务器中未实现的工具定义字段。（提示：有几个；选一个并添加它。）

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| MCP 服务器 | "暴露工具的东西" | 通过 stdio 或 HTTP 通信，遵循 MCP JSON-RPC 协议的进程 |
| stdio 传输 | "子进程模型" | 服务器由客户端启动；通过 stdin/stdout 通信 |
| 调度器 | "方法路由器" | JSON-RPC 方法名到处理函数的映射 |
| 内容块 | "工具结果块" | 工具响应 `content` 数组中的类型化元素 |
| `isError` | "工具级失败" | 表示工具失败；区别于 JSON-RPC 错误 |
| 注解 | "安全提示" | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | "Python SDK" | 基于装饰器的 MCP 协议上层框架 |
| 资源 URI | "可寻址数据" | 标识资源的 `file://`、`db://` 或自定义方案 |
| 提示模板 | "斜杠命令简介" | 服务器提供的模板，带有参数槽位供主机 UI 使用 |
| 能力声明 | "功能开关" | 在 `initialize` 中声明的每种原语的标志 |

## 扩展阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 参考 Python 实现
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 并行的 TypeScript 实现
- [FastMCP — 服务器框架](https://gofastmcp.com/) — 面向 MCP 服务器的装饰器风格 Python API
- [MCP — 快速入门服务器指南](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端教程
- [MCP — 服务器工具规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — 工具/* 消息的完整参考