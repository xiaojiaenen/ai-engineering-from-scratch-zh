# MCP 基础：无状态请求与 JSON-RPC

> 现代 MCP 没有握手，也没有协议会话。每个请求都必须携带足够的元数据，以便独立理解、授权、路由和重试。

**类型：** 学习
**语言：** Python
**前置条件：** 第 13 阶段，课程 01 至 05
**时间：** 约 55 分钟

## 学习目标

- 区分 MCP 的服务端原语与其客户端功能。
- 为 MCP `2026-07-28` 构建有效的 JSON-RPC 2.0 请求和响应。
- 在每个请求中附带协议版本、客户端能力和客户端身份。
- 使用 `server/discover` 并处理 `UnsupportedProtocolVersionError`，无需握手。
- 跟踪一个独立请求从验证到完整结果的完整流程。

## 问题所在

一个 MCP 服务端可以在同一个进程或 HTTP worker 上接收来自不同客户端的两个连续请求，它们具有不同的能力。如果服务端记住了前一个请求所声明的内容，就可能导致权限判断错误或返回错误的 wire shape。

MCP `2026-07-28` 消除了这种歧义。协议核心是无状态的。服务端必须根据当前请求来决定如何处理它，而不是依据连接历史。

这改变了心智模型。旧的顺序是：先建立连接，再做握手，最后执行操作。现代的顺序更简单：

1. 客户端发送一个自描述的请求。
2. 服务端验证该请求的版本和能力。
3. 服务端处理方法。
4. 服务端返回一个类型化的结果或 JSON-RPC 错误。

下一个请求从开始重复同样的过程。

## 概念

### 服务端原语

MCP 服务端暴露三个主要原语：

1. **工具（Tools）** 是由模型控制的操作，通过 `tools/list` 发现，通过 `tools/call` 调用。
2. **资源（Resources）** 是 URI 寻址的数据，通过 `resources/list` 发现，通过 `resources/read` 读取。
3. **提示词（Prompts）** 是可重用的模板，通过 `prompts/list` 发现，通过 `prompts/get` 渲染。

Roots、采样和日志记录在 `2026-07-28` 模式图中仍保留用于兼容性，但已标记弃用。新的实现应使用显式的工具或资源输入来处理 roots，使用直接模型提供商 API 来处理采样，使用 stderr 或 OpenTelemetry 来处理日志。Elicitation 仍可通过多轮往返请求实现，即服务端返回一个输入请求，客户端重试原始操作。现代服务端从不发起独立的 JSON-RPC 请求。

### JSON-RPC 信封

MCP 使用 JSON-RPC 2.0：

- 请求：`{jsonrpc, id, method, params}`
- 响应：`{jsonrpc, id, result}` 或 `{jsonrpc, id, error}`
- 通知：`{jsonrpc, method, params}`，没有 `id`

请求的 `id` 用于关联一个响应。它不会创建协议会话。

### 必需的请求元数据

每个现代请求都在 `params` 内携带一个 `_meta` 对象：

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    }
  }
}
```

协议版本和客户端能力是必需的。客户端身份是推荐项。它是自我报告的显示和调试数据，而非安全凭据。

服务端不得从早期请求、stdio 进程、HTTP 连接或传输层头部中推断这些值中的任何一个。

### 完整结果与服务端身份

每个成功的现代结果都包含 `resultType`。普通最终结果使用 `"complete"`。服务端还应在结果元数据中标识自身：

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "resultType": "complete",
    "tools": [],
    "ttlMs": 30000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "notes-server",
        "version": "1.0.0"
      }
    }
  }
}
```

`tools/list`、`resources/list`、`prompts/list`、`resources/templates/list`、`resources/read` 和 `server/discover` 是可缓存的结果。它们包含 `ttlMs` 和 `cacheScope`。一个安全的默认值是 `ttlMs: 0` 和 `cacheScope: "private"`。列表项应具有确定性顺序，以便等效响应产生稳定的缓存键和稳定的模型上下文。

### 无握手的发现

每个现代服务端都必须实现 `server/discover`。客户端可以在调用其他方法之前调用它，以获取：

- `supportedVersions`
- 服务端 `capabilities`
- 可选的使用说明 `instructions`
- 结果 `_meta` 中的服务端身份
- 缓存提示

发现是有用的，但它不是一个门控。客户端可以先发送 `tools/list`，因为该请求已经携带了它的协议版本和能力。

如果请求的版本不受支持，服务端返回 JSON-RPC 代码 `-32022`，内容为：

```json
{
  "requested": "2027-01-01",
  "supported": ["2026-07-28"]
}
```

客户端选择一个双方都支持的现代版本，并使用新的 JSON-RPC 请求 id 重试。

### 一个请求的生命周期

按以下顺序跟踪一个现代请求：

1. 解析一个 JSON-RPC 信封。
2. 确认 `jsonrpc` 为 `"2.0"`，存在 `id`，`method` 是字符串，`params` 是对象。
3. 要求在 `params._meta` 中存在版本字符串和能力对象；格式错误或缺失的元数据返回 `-32602`。
4. 在 HTTP 边界处，将版本、方法和适用的名称头部与请求体进行比较。即使其中一个版本值不受支持，不匹配也会返回 `-32020`。
5. 建立匹配后，拒绝已匹配但版本不支持的请求，返回 `-32022`。
6. 检查所需的能力，然后按 `method` 路由并验证方法特定的参数。
7. 在处理器运行之前，对具体操作进行认证和授权。
8. 返回带有服务端身份的完整结果。
9. 遗忘请求范围的协议元数据。

该顺序防止两个组件解读不同的调用。网关不能在授权 `Mcp-Name: notes.read` 的同时，让源端执行 `params.name: notes.delete`。它还使格式错误的输入、头部混淆、版本协商、能力失败、授权失败和处理器失败成为独立的证据。

关闭 stdin 或 HTTP 响应会结束传输活动。它不会终止协议会话，因为现代 MCP 没有协议会话。

### 显式遗留兼容

版本至 `2025-11-25` 使用 `initialize`、`notifications/initialized`、连接范围的能力，以及在早期 Streamable HTTP 上可选的协议会话。当一个双时代客户端与旧服务端通信时，这些行为仍然相关。

保持时代分离。现代请求由必需的每请求元数据标识。遗留连接仅通过文档记录的回退路径选择。不要将 `initialize` 作为 `2026-07-28` 服务端的默认操作。

因此，"无状态"具有特定时代的含义。在 `2026-07-28` 中，它是一个协议不变量：每个普通请求都是独立可解释的，不存在 MCP 会话。在 `2025-11-25` 之前的版本中，初始化和协商的能力属于连接，因此兼容适配器可以保留该遗留连接状态。双时代实现不是一个宽松的有限状态机。它是在一个隔离的遗留适配器旁运行的无状态现代核心，在任一解析器运行之前有明确的选路决策。

这两种含义都不禁止持久的应用状态。工作流、任务或草稿可以生活在共享存储中一个不透明句柄之后。客户端将该句柄作为普通输入发送，每个副本对其使用进行认证和授权。协议上下文不得泄漏到该存储中以替代被移除的会话。

```figure
mcp-tool-call
```

## 实践使用

`code/main.py` 在没有框架的情况下构建、验证、跟踪和分发现代 MCP 消息。运行：

```bash
python3 code/main.py
python3 -m unittest discover code/tests -v
```

在输出中关注三个不变量：

- 每个请求都重复其 `_meta` 字段。
- 每个成功的结果都是 `resultType: "complete"` 并包含服务端身份。
- 列表结果是确定性排序的，并具有显式的缓存提示。

## 交付使用

本课交付 `outputs/skill-mcp-handshake-tracer.md`。历史文件名保持稳定，但该工件现在是一个无状态请求追踪器。它独立审计每条消息，仅在真正存在时才标记遗留握手流量。

## 练习

1. 将一个请求的协议版本改为 `2027-01-01`。确认错误代码为 `-32022`，且数据中公布了支持的版本。
2. 从第二个请求中移除 `io.modelcontextprotocol/clientCapabilities`。确认服务端不会重用第一个请求的能力。
3. 反转内存中的工具注册表。确认 `tools/list` 仍返回相同的确定性顺序。
4. 将 `cacheScope` 从 `public` 改为 `private`。解释在每种情况下哪些授权上下文可以重用该响应。
5. 添加一个可选的 `clientInfo` 省略测试。该请求应保持有效，因为客户端身份是推荐项，而非必需项。

## 关键术语

| 术语 | 含义 |
|------|------|
| 无状态协议 | 每个请求都提供解释它所需的元数据 |
| 请求元数据 | 版本、客户端能力和推荐的客户端身份，位于 `params._meta` 中 |
| `server/discover` | 用于版本、能力、说明和身份的必要服务端方法 |
| `resultType` | 每个成功的现代结果上的判别字段 |
| 可缓存结果 | 包含必需 `ttlMs` 和 `cacheScope` 提示的结果 |
| 协议时代 | 现代每请求元数据或遗留的连接范围初始化 |
| 传输生命周期 | 进程、连接或响应流的寿命，而非协议会话状态 |
| `-32022` | 不支持的协议版本错误，包含请求版本和支持版本 |

## 延伸阅读

- [MCP 架构](https://modelcontextprotocol.io/specification/2026-07-28/architecture)
- [MCP 基础协议](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP 服务端发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP 2026-07-28 变更日志](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
