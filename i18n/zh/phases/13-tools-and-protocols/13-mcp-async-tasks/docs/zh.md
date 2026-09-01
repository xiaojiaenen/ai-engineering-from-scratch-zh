# MCP Tasks 扩展：在状态核心上的持久化工作

> 状态无涉 MCP 并不意味着每个操作必须在一次请求中完成。官方 Tasks 扩展为长时间运行的工作提供了显式的持久化句柄。服务器可以从 `tools/call` 返回该句柄，任何实例都可以响应 `tasks/get`，客户端输入通过 `tasks/update` 到达，无需恢复协议会话。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 13 · 09（传输层）、Phase 13 · 11（状态无涉 MRTR）、Phase 13 · 12（elicitation）
**预计时间：** ~90 分钟

## 学习目标

- 区分状态无涉协议传输与持久化应用任务状态。
- 在每次请求的 capabilities 和 `server/discover` 中协商 `io.modelcontextprotocol/tasks` 扩展。
- 仅在持久化创建后才返回服务器驱动的 `CreateTaskResult`，且 `resultType: "task"`。
- 通过 `tasks/get` 轮询，通过 `tasks/update` 提供任务输入，并通过 `tasks/cancel` 请求协作式取消。
- 移除旧的 `tasks/status`、`tasks/result` 和 `tasks/list` 假设。
- 通过在 POST 响应 SSE 流上使用 `subscriptions/listen` 订阅可选的任务通知。
- 正确建模任务过期、重启恢复、输入键去重和执行错误。

## 为什么 Tasks 是扩展

Tasks 首次作为实验性核心功能出现在 2025-11-25 版本。2026 年 7 月的重新设计将其移入官方 `io.modelcontextprotocol/tasks` 扩展，使客户端和服务器可以选择加入额外的生命周期，而不会为所有人扩展核心协议。

尽管它是 Tasks 当前的官方归属，扩展规范仍然是草稿版本。固定你的 SDK 支持的扩展版本，运行合规性场景，并将线适配器与你的 worker 和存储域隔离。

当操作具有以下一个或多个特性时使用任务：

- 可能超出普通请求超时。
- worker 队列或外部作业系统已拥有执行权。
- 客户端需要在自身重启后恢复。
- 操作在执行过程中暂停等待用户或模型输入。
- 取消和持久化结果检索是产品需求。

不要为廉价确定性查找创建任务。句柄、持久化、轮询、过期和取消都是真实的复杂性。

## 状态无涉核心，状态ful 应用

MCP 2026-07-28 移除了 `initialize`、`notifications/initialized`、协议会话和 `Mcp-Session-Id`。这并不禁止有状态的产品。

任务 id 是显式的应用状态：

- 服务器在返回前持久化它。
- 客户端可以存储它并在重启后再次轮询。
- 该 id 可以路由到由同一持久存储支持的任意副本。
- 每次任务方法都检查授权。
- 过期和删除由任务字段定义，而非传输生命周期。

这在操作上与附加到连接的隐藏状态不同。

保持四个生命周期分离：

| 状态 | 生命周期 | 归属位置 |
|---|---|---|
| 协议元数据 | 一次请求 | `params._meta`，每次调用时重新验证 |
| 传输工作 | 一次 stdio 请求或 HTTP 响应 | 带有限制期限的在进行中协调器 |
| MRTR 延续 | 一次重试序列 | 完整性保护的 `requestState`，以及按需的重放控制 |
| 持久任务 | 跨请求、副本、重启和重连 | 通过授权 `taskId` 键控的共享应用存储 |

将任务记录移入进程内存并不会使 MCP 有状态。它会使应用不可靠。协议保持状态无涉，但后续路由到另一副本的 `tasks/get` 无法恢复该记录。在返回句柄前持久化，然后让每个任务方法在租户和主体检查下解析同一共享记录。

## 能力协商

客户端在每次符合条件的请求上声明支持：

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {
        "io.modelcontextprotocol/tasks": {}
      }
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "lesson-client",
      "version": "1.0.0"
    }
  }
}
```

服务器从 `server/discover` 返回精确的 `supportedVersions`、capabilities、`ttlMs` 和 `cacheScope`，capabilities 下包含同一扩展。因为声明了工具，它也实现了必需的 `tools/list`。该结果返回确定性的 `generate_report` 描述符、有效的 object `inputSchema`、`resultType: "complete"`、服务器身份元数据和公开缓存提示。

来自未声明该扩展的客户端的任务方法返回 `-32021`（缺少必需的客户端能力），其中 `data.requiredCapabilities` 设置为 `{"extensions":{"io.modelcontextprotocol/tasks":{}}}`。不支持的协议字符串返回 `-32022`，附带精确的 `supported` 和 `requested` 数据；缺失或非字符串版本返回 `-32602`。

没有 JSON-RPC `id` 的包是通知。接收方可以处理它，但不会发出 JSON-RPC 结果或错误。Streamable HTTP 适配器对接受的通知返回 `202 Accepted`，无正文。

目前，只有 `tools/call` 支持任务增强执行。设计你的内部抽象，使未来的请求类型不需要重写存储。

## 服务器驱动的任务创建

旧的客户端标志 `params._meta.task.required` 已移除。客户端声明扩展支持，然后服务器决定特定的 `tools/call` 是否成为任务。

请求：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "generate_report",
    "arguments": {"size": "large"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

响应：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "task",
    "taskId": "tsk_786512e29e0d",
    "status": "working",
    "statusMessage": "Preparing report outline.",
    "createdAt": "2026-08-21T10:30:00Z",
    "lastUpdatedAt": "2026-08-21T10:30:00Z",
    "ttlMs": 900000,
    "pollIntervalMs": 1000
  }
}
```

在针对该 id 的 `tasks/get` 可以解析之前，服务器不得返回此句柄。在最终一致存储中，在回答前等待读可见性。否则客户端可能收到看起来有效的 id，随即得到"未找到"。

任务响应在某种意义上是未经请求的，因为客户端不请求任务模式。但它并非未经协商：当前请求仍必须声明扩展。

## 任务形态

每个任务包含：

- `taskId`：稳定的服务器生成标识符；
- `status`：`working`、`input_required`、`completed`、`cancelled` 或 `failed`；
- `createdAt` 和 `lastUpdatedAt`：ISO 8601 时间戳；
- `ttlMs`：从创建起的过期时长，或 `null` 表示无 advertised 限制；
- 可选 `pollIntervalMs`：服务器的当前最小建议轮询间隔；
- 可选 `statusMessage`：面向用户或模型的上下文。

状态特定的字段仅在相关时出现：

- `input_required` 包含 `inputRequests`。
- `completed` 包含原始请求的 `result` 形态。
- `failed` 包含 JSON-RPC `error` 对象。

客户端应遵守 `pollIntervalMs`。服务器可能对更激进的轮询进行速率限制，并可能在任务生命周期内更改间隔。

## 通过 `tasks/get` 轮询

客户端请求当前快照：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/get
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tasks/get",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

`tasks/get` 本身已完成，因此其结果始终具有 `resultType: "complete"`。嵌套任务仍可能有 `status: "working"` 或 `status: "input_required"`。

此区别防止了常见的解析器 bug：

```text
result.resultType = complete    表示 tasks/get RPC 已完成
result.status = working         表示所表示的作业仍在运行
```

没有 `tasks/result` 调用。当任务完成时，下一个 `tasks/get` 响应在 `result` 下内联原始 `CallToolResult`：

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "completed",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:34:12Z",
  "ttlMs": 900000,
  "result": {
    "resultType": "complete",
    "content": [
      {"type": "text", "text": "Generated large report with approved outline."}
    ],
    "structuredContent": {"size": "large", "approved": true},
    "isError": false,
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "tasks-demo",
        "version": "1.0.0"
      }
    }
  },
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "tasks-demo",
      "version": "1.0.0"
    }
  }
}
```

外层 `resultType` 表示 `tasks/get` RPC 已完成。嵌套的 `result.resultType` 表示原始工具调用已完成。该嵌套鉴别器是必需的。嵌套的 `CallToolResult` 还应携带其自己的 `io.modelcontextprotocol/serverInfo`；本课程包含它而非存储未类型化的载荷。

没有 `tasks/list`。无会话服务器无法安全地推断哪些任务属于连接作用域列表。需要历史记录的应用应公开带有显式过滤和所有权规则的授权域工具。

## 任务执行期间的输入

任务输入和核心 MRTR 看起来相似，但使用不同的延续。

### 任务创建前的输入

从原始 `tools/call` 返回核心 `resultType: "input_required"`。客户端兑现它并重试该原始调用。仅在那些同步 MRTR 轮完成后才创建任务。

### 任务创建后的输入

将任务设置为 `input_required`。`tasks/get` 暴露待处理的 `inputRequests`，客户端通过 `tasks/update` 发送响应。客户端不重试原始 `tools/call`。

快照：

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "input_required",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:31:00Z",
  "ttlMs": 900000,
  "inputRequests": {
    "approve_outline": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Approve the generated report outline?",
        "requestedSchema": {
          "type": "object",
          "properties": {"approved": {"type": "boolean"}},
          "required": ["approved"]
        }
      }
    }
  }
}
```

更新：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/update
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tasks/update",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "inputResponses": {
      "approve_outline": {
        "action": "accept",
        "content": {"approved": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

成功响应是一个空确认加上 `resultType: "complete"`。状态变更可能是最终一致的，因此客户端继续轮询或监听。

每个 `inputRequests` 键在整个任务生命周期中必须唯一。重复的 `tasks/get` 快照可能显示相同的待处理键；客户端对 UI 去重，服务器忽略未知、已取代或已兑现的键的响应。部分更新可能使任务保持 `input_required`，直到所有必需键都已回答。

## 取消是协作式的

`tasks/cancel` 发出意图信号并返回空完成确认。该确认并不保证 worker 已停止。工作可能首先完成、忽略取消或稍后转换。

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/cancel
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tasks/cancel",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

对于所有三个任务方法，`Mcp-Name` 镜像 `params.taskId`。它不重复 JSON-RPC 方法名。`code/main.py` 在 `make_http_request` 中集中了此规则。

课程 worker 立即尊重取消，使重复调用幂等。生产客户端仍必须将取消视为协作式，而非从确认推断最终任务状态。

不要使用 `notifications/cancelled` 来取消任务。该通知属于请求取消，而非持久化 Tasks。

此区别在路由边界处很重要。请求取消针对一个进行中的 JSON-RPC 操作或其请求作用域的 HTTP 响应。如果 `tools/call` 已经返回 `resultType: "task"`，则该请求已完成，关闭其传输无法命名或停止持久化作业。`tasks/cancel` 是一个新的授权 RPC。它携带 `params.taskId`，在 `Mcp-Name` 中镜像该 id，解析任务的拥有后端，记录协作取消意图，并返回确认而不声称 worker 已停止。

因此网关必须将请求协调器和任务路由保持在不同的表中。请求表可以在响应完成后消失。任务路由必须持续到终态和保留过期。[课程 29：MCP 可靠性、取消和流量控制](../../29-mcp-reliability-cancellation-and-flow-control/docs/en.md) 为两条路径构建竞态、超时、幂等性、背压和重试规则。

## 可选通知

轮询是基线。想要推送更新的客户端发送带有任务 id 的 `subscriptions/listen`。对于 Streamable HTTP，这是一个 POST，其响应是请求作用域的 SSE 流。没有独立的 GET 事件流，也没有需要保持存活的协议会话。

服务器通过 `notifications/subscriptions/acknowledged` 确认已接受的 id，然后通过 `notifications/tasks` 发送完整快照。确认和每个任务通知都在 `_meta` 中携带 `io.modelcontextprotocol/subscriptionId`，等于 `subscriptions/listen` 请求 id。每个任务通知在其他方面等价于 `tasks/get` 在该时刻会返回的内容。

客户端仍必须声明 Tasks 扩展。它们应重新连接并从持久任务 id 恢复，而非依赖事件重放或 `Last-Event-ID`。

## 失败语义

正确使用两个错误层。

### 协议错误

无效的方法参数或未知的任务 id 返回 JSON-RPC 错误，通常为 `-32602`。缺少扩展支持返回 `-32021`，附带必需能力对象。

### 任务执行结果

- 带有 `isError: true` 的正常工具结果仍然是 `completed` 任务，因为工具调用产生了其定义的结果。
- 延迟执行期间的 JSON-RPC 错误使任务 `failed`，并将该 JSON-RPC 错误存储在 `error` 下。
- 用户拒绝可以产生 `cancelled`、完成的拒绝结果或其他域特定的安全结果。记录选择。

## 持久化、过期和所有权

至少持久化任务 id、状态、时间戳、ttl、轮询间隔、原始操作所有权、结果或错误、待处理输入请求和所有已发出的输入键。

存储键必须包含或解析权威租户和主体。知道任务 id 不得授予访问权限。在每次 `tasks/get`、`tasks/update`、`tasks/cancel` 和订阅时检查所有权。

`ttlMs` 从创建时测量，可能发生变化。当任务停止产生可观察更新时，客户端可以将其视为安全网。服务器可以失败并稍后删除已过期的任务。不要将其描述为在完成后的那么多毫秒内保留完成结果的承诺。

使用原子写入或事务。课程写入临时文件并原子重命名它。多副本服务应使用共享持久存储和 worker 租约或等效并发控制。

```figure
tp-task-lifecycle
```

## 构建它

`code/main.py` 实现了一个确定性任务服务：

- `server/discover` 返回 `supportedVersions`、缓存提示和 Tasks 扩展。
- `tools/list` 返回确定性的、可缓存的 `generate_report` 描述符，带有有效的输入模式。
- `tools/call` 在返回 `resultType: "task"` 前创建并持久化任务。
- 新服务实例重新加载同一任务，演示重启恢复。
- `tasks/get` 返回完整任务快照。
- worker 从 `working` 转换到 `input_required`。
- `tasks/update` 接受表单响应并返回空完成确认。
- worker 存储带有其自己的 `resultType` 和服务器身份的嵌套 `CallToolResult`，然后转换到 `completed`。
- `tasks/cancel` 在此实现中是幂等的。
- HTTP 构建器为 `tasks/get`、`tasks/update` 和 `tasks/cancel` 将 `Mcp-Name` 设置为 `params.taskId`。
- 通知 helper 使用 `notifications/subscriptions/acknowledged` 和 `notifications/tasks`，都带有 listen 请求 id 标记。
- 无 id 的通知不产生 JSON-RPC 响应。

worker 显式推进而非在后台线程中睡眠。这使得每个状态转换确定，并保持协议示例与队列机制分离。

## 使用它

从仓库根目录：

```bash
cd phases/13-tools-and-protocols/13-mcp-async-tasks/code
python3 main.py
python3 -m unittest discover tests -v
```

预期结果序列：

```text
id=0 resultType=complete status=ack
id=1 resultType=task status=working
id=2 resultType=complete status=working
id=3 resultType=complete status=input_required
id=4 resultType=complete status=ack
id=5 resultType=complete status=completed
```

同时验证 `tasks/status`、`tasks/result` 和 `tasks/list` 在现代服务中返回 method-not-found。
验证 `tools/list` 是确定性的，且每个当前 HTTP 任务方法通过 `Mcp-Name` 镜像其任务 id。

## 交付它

`outputs/skill-task-store-designer.md` 现在产生一个扩展感知的设计：能力协商、持久化优先返回创建、当前方法、输入更新流程、所有权、过期、取消、订阅和从已移除的实验方法的迁移。

## 练习

1. 添加第二个待处理输入键。发送部分 `tasks/update` 并证明任务保持 `input_required`，直到两个键都得到回答。
2. 向存储添加租户所有权，并拒绝由错误认证主体呈现的有效任务 id。
3. 添加带有过期的 worker 租约。演示两个服务实例不能并发完成同一任务。
4. 为 `subscriptions/listen` 实现 POST 响应 SSE 适配器。不添加 GET、`Last-Event-ID` 或会话头。
5. 添加过期清理。区分过期任务与畸形任务 id，而不泄露跨租户存在。

## 关键术语

| 术语 | 当前扩展中的含义 |
|------|------------------|
| Tasks 扩展 | 用于持久化异步工作的可选 `io.modelcontextprotocol/tasks` 能力 |
| `CreateTaskResult` | 对符合条件请求的服务器驱动 `resultType: "task"` 响应 |
| `tasks/get` | 轮询完整当前任务快照，包括终端结果或待定输入 |
| `tasks/update` | 向任务的待处理 `inputRequests` 提交响应 |
| `tasks/cancel` | 确认协作取消意图 |
| `input_required` | 指示客户端输入待处理的任务状态 |
| `pollIntervalMs` | 服务器建议的下一次轮询前的最小延迟 |
| `ttlMs` | 从任务创建起测量的过期时长 |
| 持久化优先返回 | 任务 id 必须在发送其句柄前可解析的规则 |
| `notifications/tasks` | 在订阅的 SSE 响应上交付的可选完整任务快照 |

## 遗留兼容性

2025-11-25 实验性表面使用客户端请求的任务增强、`tasks/status`、`tasks/result` 和可选的 `tasks/list`。仅在固定的遗留适配器内保留这些名称。当前客户端使用扩展能力，接受服务器驱动的句柄，轮询 `tasks/get`，用 `tasks/update` 提供输入，并从任务快照读取最终结果。

## 延伸阅读

- [官方 MCP Tasks 扩展](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
- [MCP 2026-07-28 多往返请求](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
