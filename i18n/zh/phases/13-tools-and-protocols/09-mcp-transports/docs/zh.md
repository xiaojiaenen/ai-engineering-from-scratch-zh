# MCP 传输层：stdio 和无状态 Streamable HTTP

> 传输层承载 MCP 消息。它不提供缺失的协议状态。在 `2026-07-28` 版本中，本地 stdio 和远程 Streamable HTTP 都携带自描述请求。

**类型：** 学习
**语言：** Python
**前置知识：** 第13阶段，课程 07 和 08
**预计时长：** 约65分钟

## 学习目标

- 为本地子进程选择 stdio，为网络服务选择 Streamable HTTP。
- 实现现代单端点、仅 POST 的 Streamable HTTP 协议。
- 镜像并验证 MCP 版本、方法和名称头与 JSON-RPC 请求体的一致性。
- 正确交付请求级 SSE 和长期 `subscriptions/listen` 流。
- 从基于会话和遗留 HTTP+SSE 的部署迁移，而不将遗留行为呈现为现代。

## 问题背景

早期 Streamable HTTP 修订版本将协议协商与连接和会话行为混合在一起。服务器可以创建 `Mcp-Session-Id`、暴露独立的 GET 流、接受 DELETE 用于会话终止，以及使用 `Last-Event-ID` 恢复 SSE。

MCP `2026-07-28` 从现代协议中移除了这些机制。每个请求都可以到达任何健康的 worker，因为其协议版本和客户端能力信息包含在请求体中。HTTP 头镜像了部分字段用于路由和策略，但服务器在执行前会验证这些头与请求体的一致性。

这使得系统更容易扩展、更容易推理。这也意味着，如果一个服务器仍然教授 2025 传输协议作为现代协议，那就是在教授错误的故障和安全模型。

## 概念解析

### stdio

stdio 绑定用于客户端启动的子进程：

- 客户端每行写入一个 UTF-8 JSON-RPC 消息到 stdin。
- 服务端每行写入一个 UTF-8 JSON-RPC 消息到 stdout。
- 服务端将诊断信息写入 stderr。
- 服务端在 stdin EOF 时立即退出。
- 每个现代请求都在 `params._meta` 中携带版本和客户端能力信息。

进程可以存活很多次调用，但它不是现代协议的会话。如果进程意外退出，进行中的请求将丢失。重启进程、重新发现、重新列出工具、重新打开订阅，并用新的请求 ID 重试安全操作。

### 2026-07-28 的 Streamable HTTP

现代服务器暴露一个 MCP 端点，例如 `/mcp`，只接受 POST。

每个 JSON-RPC 请求或通知都是一个独立的 HTTP POST。请求体包含一个 JSON-RPC 消息。客户端不会向服务器发送 JSON-RPC 响应。

对于请求，服务器返回以下之一：

- `Content-Type: application/json` 包含一个 JSON-RPC 响应；或
- `Content-Type: text/event-stream` 包含与该请求相关的通知，后跟最终的 JSON-RPC 响应。

对于已接受的通知，服务器返回 `202 Accepted` 且无响应体。

客户端声明支持两种响应类型：

```http
Accept: application/json, text/event-stream
```

### 仅 POST 就是仅 POST

现代 Streamable HTTP 没有独立的 GET 流，也没有 DELETE 会话端点。

- `GET /mcp` 返回 `405 Method Not Allowed`。
- `DELETE /mcp` 返回 `405 Method Not Allowed`。
- `Mcp-Session-Id` 被忽略，既不创建也不回显。
- `Last-Event-ID` 被忽略，因为现代流不可恢复。

如果请求级流在最终响应前中断，客户端就丢失了该进行中的请求。当重试安全时，可以发起一个新请求并携带新的 JSON-RPC id。不得尝试流恢复。

### 源验证

服务器验证传入连接的 `Origin` 头以防止 DNS 重新绑定攻击。如果该头存在且不在显式允许列表中，返回 `403 Forbidden`。非浏览器客户端可能省略 `Origin`，官方传输规则允许这种情况。

本地服务器应绑定到 `127.0.0.1`，而不是所有接口。网络服务仍需在每个请求上进行认证和授权。源验证不是认证。

在规范化配置后使用精确的源匹配。像 `origin.startswith("https://trusted.example")` 这样的前缀检查是不安全的，因为它们可能接受攻击者控制的域名后缀。

### 必需的 HTTP 元数据头

每个现代 POST 请求都包含：

```http
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes_search
```

头规则：

- `MCP-Protocol-Version` 是必需的，且必须等于 `params._meta.io.modelcontextprotocol/protocolVersion`。
- `Mcp-Method` 是必需的，且必须等于 JSON-RPC 的 `method`。
- `Mcp-Name` 对 `tools/call`、`resources/read` 和 `prompts/get` 是必需的。
- `Mcp-Name` 等于 `params.name`，或对 `resources/read` 使用 `params.uri`。
- 头值区分大小写，即使头名本身不区分大小写。

不安全或非 ASCII 的 `Mcp-Name` 值使用精确的 UTF-8 Base64 哨兵格式：

```text
=?base64?{Base64EncodedValue}?=
```

服务器在比较前先解码该值，再与请求体进行比较。

缺失、格式错误或镜像不一致的头返回 HTTP `400` 和 JSON-RPC 错误码 `-32020`。如果头与请求体就服务器不支持的版本达成一致，返回 HTTP `400` 和 `-32022`，并附带精确的错误数据，如 `{"supported":["2026-07-28"],"requested":"2027-01-01"}`。

未知的现代方法返回 HTTP `404` 和 JSON-RPC 错误码 `-32601`。JSON-RPC 请求体很重要，因为双时代客户端用它来区分现代错误和遗留端点未命中。

### 请求级 SSE

服务器可以为某个长时间运行的请求选择 SSE：

```text
POST tools/call id=41
  <- 与 id=41 相关的 notifications/progress
  <- 与 id=41 相关的 notifications/progress
  <- JSON-RPC 响应 id=41
流关闭
```

服务器不得在此流上发送独立的 JSON-RPC 请求。采样、elicitation 和 roots 交互使用多轮往返请求结果。关闭响应流即取消该请求。

不要为回放大厦添加 SSE 事件 id。`Last-Event-ID` 恢复不是现代修订版的一部分。

### 长期变更使用 subscriptions/listen

变更通知使用客户端发起的请求，而不是独立的 GET：

```json
{
  "jsonrpc": "2.0",
  "id": "listen-1",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true,
      "resourceSubscriptions": ["notes://note-1"]
    },
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

POST 响应是一个长期 SSE 流。其第一个协议消息是 `notifications/subscriptions/acknowledged`。确认消息、每个变更通知和最终结果都在 `_meta` 中携带 `io.modelcontextprotocol/subscriptionId`，其值等于 listen 请求 id。服务器可以发送 SSE 注释作为保活消息。当流断开时，客户端重新发起 `subscriptions/listen` 并携带新的请求 id，同时重新获取受影响的数据。

`resources/subscribe` 和 `resources/unsubscribe` 属于遗留时代。不得在现代连接中使用它们。

### 显式应用状态

移除协议会话并不意味着禁止有状态的工作流程。服务器可以创建一个不透明的状态句柄，并将其作为普通工具结果返回。客户端在后续调用中将其作为显式参数传递。

将句柄绑定到认证主体、使其不可猜测、设置过期时间，并对每次使用进行授权检查。这使得状态在应用层可见，而不是隐藏在传输层连接亲和力中。

隐藏副本状态导致的故障是这样的：

1. 请求 A 到达副本 1 并在该进程的内存中创建草稿。
2. 响应不返回草稿句柄，因为实现假设连接能标识该草稿。
3. 请求 B 是一个新的 POST，到达副本 2。
4. 副本 2 有有效的协议元数据，但无法命名或加载该草稿，导致工作流失败或读取错误的本地对象。
5. 粘性路由看起来能修复症状，直到重启、部署、重新调度或故障转移移动下一个请求。

正确的边界有两个部分。协议上下文保留在每个请求中。持久化的应用状态存放在共享存储中，由服务器创建句柄并返回给客户端。下一次调用提供该句柄，任何副本都能加载相同的记录，授权将该记录绑定到认证主体和租户。副本内存可以缓存一条记录，但不能是唯一保证正确性所需的副本。

根据生命周期选择状态机制。请求局部变量可以服务单次调用。短寿命的多轮往返续传可以使用完整性保护的 `requestState`。草稿或持久化任务需要显式句柄加上共享持久化、过期、并发控制和幂等性。这些对象都不是 MCP 协议会话。

### HTTP 双时代兼容性

支持现代和遗留服务器的客户端首先尝试现代 POST。如果收到 HTTP `400`、`404` 或 `405`，它就检查响应体：

- 识别的现代 JSON-RPC 错误证明服务器是现代的。修正请求或重试已宣告的版本。不得降级。
- 空响应体或无法识别的响应可能表示遗留的 HTTP+SSE 服务器。只有这时才尝试旧的 GET 端点，并期望收到其遗留的 `endpoint` 事件。

服务器可以在迁移期间同时支持两个时代，通过将现代元数据路由到现代 POST-only 实现，同时为旧客户端保留单独的遗留端点。永远不要将遗留的 GET、DELETE、会话 id 或回放行为描述为 `2026-07-28` 的一部分。

```figure
tp-transport-handshake
```

## 使用它

`code/main.py` 使用 Python 标准库实现了有限的现代 Streamable HTTP 服务器。它验证 Origin 和镜像头、忽略已移除的会话头、为正常调用返回 JSON，并演示了有限的 `subscriptions/listen` SSE 流。

```bash
cd code
python3 main.py --probe
python3 -m unittest discover tests -v
```

探针检查：

- 无效 Origin 被拒绝；
- 无需 session id 即可发现成功；
- `Mcp-Session-Id` 和 `Last-Event-ID` 被忽略；
- 头不匹配返回 `-32020`；
- 不支持的版本返回 `-32022` 及精确的 `supported` 和 `requested` 数据；
- 已接受的无 id 通知返回 HTTP `202` 且无响应体；
- GET 和 DELETE 返回 `405`；
- `subscriptions/listen` 是一个 POST 响应流，其确认消息、通知和最终结果都携带其 subscription id。

## 交付它

本课程交付 `outputs/skill-mcp-transport-migrator.md`。它移除了现代协议会话、添加了头-体验证、用 `subscriptions/listen` 替换了独立的 GET，并保持任何遗留桥接代码可见且分离。

## 练习

1. 从 POST 中移除 `Mcp-Method`。确认 HTTP `400` 和错误 `-32020`。
2. 发送匹配的头和请求体版本 `2027-01-01`。确认 HTTP `400`、错误 `-32022` 及精确数据 `{"supported":["2026-07-28"],"requested":"2027-01-01"}`。
3. 为非 ASCII 资源 URI 发送 Base64 哨兵 `Mcp-Name`。确认解码后的值与 `params.uri` 进行比较。
4. 在有限 listen 流的最终响应前中断它。用新的 JSON-RPC id 重新发起它并重新获取工具。
5. 在 ping 工具中添加显式工作流句柄。在不使用连接亲和力的情况下将其绑定到认证主体。

## 关键术语

| 术语 | 含义 |
|------|---------|
| stdio | 通过客户端启动的子进程传输的换行分隔 JSON-RPC |
| Streamable HTTP | 每个现代消息都是独立 POST 的单端点 |
| 请求级 SSE | 包含相关通知和最终响应的 POST 响应流 |
| `subscriptions/listen` | 用于 opted-in 变更通知的长期 POST 请求 |
| 头不匹配 | 镜像头与请求体不一致时返回 HTTP `400` 和 JSON-RPC `-32020` |
| 源验证 | 传入连接的 DNS 重新绑定防御机制，而非认证 |
| 显式状态句柄 | 作为普通参数传递的应用令牌，而非隐藏的会话状态 |
| 遗留桥接 | 仅为兼容性保留的早期时代行为 |

## 延伸阅读

- [MCP 传输概述](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [MCP stdio 传输](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP 订阅](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 变更日志](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
