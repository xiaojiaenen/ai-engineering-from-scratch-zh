# 无状态 MCP 网关与注册表准入

> 网关应当使每条路由都显式化。2026-07-28 协议在不依赖传输会话的情况下，赋予了它方法、名称、版本、能力、身份、缓存和追踪边界。

**类型：** 学习
**语言：** Python
**前置条件：** 第 13 · 15 阶段（安全性）、第 13 · 16 阶段（授权）
**时间：** 约 75 分钟

## 学习目标

- 在不依赖会话亲和性的情况下，将多个 MCP 服务器聚合到一个 2026-07-28 端点后面。
- 在策略或转发之前验证每个请求的元数据和路由头。
- 使用稳定命名空间、确定性顺序、描述符锁定、RBAC 和私有缓存来合并工具。
- 将注册表记录视为发现证据，但仍需要准入策略。
- 正确路由请求范围的 SSE、`subscriptions/listen`、MRTR 重试和 Tasks 扩展调用。
- 将旧式握手和会话支持与现代化路径隔离。

## 问题

将一个客户端直接连接到一台服务器很简单。更大的部署需要对更难的问题给出一致的答案：

- 哪些服务器被允许？
- 哪个主体可以查看和调用每个工具？
- 当两个后端暴露相同名称时会发生什么？
- 描述符变更如何审查？
- 速率限制和审计事件应用在哪里？
- 任何实例都能处理下一个请求吗？

网关位于客户端和后端 MCP 服务器之间。它呈现一个 MCP 端点，应用跨切面策略，并转发经过批准的请求。

旧的网关设计通常将一个客户端会话多路复用到多个后端会话并重写 `Mcp-Session-Id`。这是一种遗留兼容性设计。2026-07-28 核心没有协议会话。

## 概念

### 现代化网关路径

对于每个请求：

1. 从传输授权中认证主体。
2. 验证 `MCP-Protocol-Version`、`Mcp-Method`、`Mcp-Name` 和 `params._meta`。
3. 对主体、资源、方法、工具和参数进行授权。
4. 应用描述符、注册表、速率和数据策略。
5. 为选定的后端创建独立的新请求。
6. 验证后端结果并返回网关结果。
7. 记录审计事件但不记录密钥。

没有步骤需要隐藏的协议会话。应用程序状态仍然可以存在于数据库、显式句柄、Tasks 或完整性保护的 MRTR 状态中。

### 运行时策略是主要网关决策

准入决定哪个后端版本可以进入网关。它不对实时调用进行授权。对于每个请求，网关根据已认证主体、发行者和资源、租户、匹配的方法和名称、归一化参数、已准入的描述符锁、当前后端健康状态、能力交集、数据分类、速率状态以及任何操作绑定批准重新计算策略。

这个顺序很重要。一个注册表记录可以在用户角色被撤销后仍然保持活动状态。一个描述符可以在目标参数跨越租户边界时保持锁定。一个后端可以在事件策略隔离状态变更调用时保持批准。因此，运行时策略是主要的允许或拒绝决策，注册表和描述符证据作为输入。

不要在连接或已删除的会话标识符下缓存允许决策。如果策略不可用，按操作类别遵循声明的失败策略。一个安全的默认做法是对状态变更和敏感读取失败关闭，而明确批准的公共读取路径仅在风险模型允许时使用短期最后已知策略。记录哪个策略版本和失败路径做出了决策，然后在返回之前验证后端结果。

### 一个 POST 端点

现代 Streamable HTTP 通过 POST 发送每个 JSON-RPC 消息：

```text
POST /mcp
Authorization: Bearer <gateway-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.search
Accept: application/json, text/event-stream
```

网关可以返回 JSON 或该 POST 的请求范围 SSE。GET 和 DELETE 对现代请求返回 405。`Mcp-Session-Id` 和 `Last-Event-ID` 不创建权限、亲和性或重放行为。

头和正文值必须一致。在查找后端之前，拒绝不匹配并返回 `-32020`。这使负载均衡器、网关和速率限制器无需解析完整正文即可路由，同时保留端到端完整性。

按以下精确顺序验证：JSON-RPC 和元数据类型、头和正文相等性，然后是对匹配版本的支持。不匹配返回 HTTP 400 和 `-32020`。如果头和正文在不支持的版本上一致，则返回 HTTP 400 和 `-32022` 以及 `data` 精确为 `{"supported":["2026-07-28"],"requested":"<actual>"}`。未知方法返回 HTTP 404 和 `-32601`。

`ProtocolError` 携带可选 `data`，网关将其序列化为 JSON-RPC 错误对象。通知没有 `id`，因此永远不会收到 JSON-RPC 成功或错误。接受的 HTTP 通知返回 202 和空正文。

### 在每一层实现发现

网关为客户端实现 `server/discover`。它还发现每个后端，以便了解协议版本、能力和扩展。

示例网关结果：

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": true}
  },
  "ttlMs": 30000,
  "cacheScope": "private",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "enterprise-gateway",
      "version": "2.0.0"
    }
  }
}
```

仅通告网关能够端到端 honored 的能力交集。后端功能并非自动安全暴露。没有后端路径的网关功能不适合通告。

`serverInfo` 是自报的显示和诊断数据。不要将其用作注册表或发布者证明。

### 每个请求的客户端能力

每个转发请求都需要当前的 `_meta` 信封：

```json
{
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {},
  "io.modelcontextprotocol/clientInfo": {
    "name": "enterprise-gateway",
    "version": "1.0.0"
  }
}
```

不要盲目地将外部客户端能力复制到后端。网关是后端的客户端。仅通告网关将正确中介的功能。

### 确定性命名空间

在稳定公共名称下合并后端工具：

```text
notes.search
notes.create
issues.list
issues.open
```

保持从公共名称到后端和原始工具名称的映射。永远不要选择第一个或最后一个冲突项。公共名称是审批和审计合同的一部分，因此更改它是迁移。

`tools/list` 必须是确定性的。当可见性因主体而异时，返回 `cacheScope: private`。有界的 `ttlMs` 减少后端发现负载，同时不允许用户特定列表跨授权上下文泄漏。

每个暴露的工具描述符都包括稳定名称、描述和对象根 `inputSchema`。命名空间无法删除必需的描述符字段。完整的列表结果还包括 `resultType`、服务器身份元数据和缓存提示。

### 锁定已批准描述符

在准入时，规范化完整描述符并在限定公共名称下存储其摘要。在列表和调用时，将活动描述符与批准摘要进行比较。

如果发生变化：

- 从 `tools/list` 中移除它。
- 拒绝直接调用。
- 发出审计事件。
- 在更新锁之前需要策略或人工重新批准。

网关是一个有用的中央执行点，但它不会使首次看到的描述符变得安全。初始审查仍然是必要的。

### 注册表帮助发现，而非决策

注册表 `server.json` 提供发布元数据。包支持的记录可能如下所示：

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "com.example/notes",
  "description": "Example notes MCP server.",
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "npm",
      "identifier": "@example/notes-mcp",
      "version": "1.0.0",
      "transport": {"type": "stdio"}
    }
  ]
}
```

发布元数据不携带网关的安全决策。在单独的准入状态中保留经验证的发布者和溯源证据：

```json
{
  "registryName": "com.example/notes",
  "registryVersion": "1.0.0",
  "publisher": {"namespace": "com.example", "status": "verified"},
  "provenance": {
    "source": "registry.modelcontextprotocol.io",
    "recordId": "com.example/notes@1.0.0"
  },
  "admission": {"status": "approved", "reviewedBy": "gateway-policy"}
}
```

网关检查 `server.json` 形状并将其加入外部状态。网关仍然需要准入策略。

对于每个已准入的后端，记录：

- 确切的注册表和记录标识符。
- 经验证的发布者命名空间或域证据。
- 允许的传输和端点。
- 锁定版本或批准升级策略。
- 工件或描述符摘要。
- 授权发行者和资源。
- 审查者、批准时间和过期时间。

不要因其显示名称类似于熟悉的产品而接受服务器。不要将注册表存在视为运营安全检查。私有服务器可以通过相同的证据模式准入，即使它们从不出现在公共注册表中。

本教程实现网关接缝：在后端变得可路由之前，将发布证据加入本地准入。[课程 30：MCP 注册表供应链、准入、漂移和回滚](../../30-mcp-registry-supply-chain-and-drift/docs/en.md) 构建了完整的控制平面，用于精确命名空间证明、工件溯源、不可变锁定、活动描述符漂移、注册表状态协调、防篡改准入分类账和基于证据的回滚。将该供应链状态与上述请求范围的运行时决策保持分离。

### 凭据中介

网关对其调用者进行认证，并单独向后端认证。后端凭据永不发送给客户端。

保持这些绑定明确：

```text
外部主体 -> 网关角色和策略
后端发行者 + 资源 -> 后端注册和令牌
```

切勿将外部网关令牌传递给后端。切勿在不同发行者或资源处重用后端令牌。如果工具代表最终用户操作，则通过设计交换或声明模型保留该委托，而不是使用共享服务凭据冒充用户。

### 无会话的速率限制

按已认证主体、发行者、资源、公共工具、成本类和 time window 的关键限制。会话 id 不存在，即使存在也很容易被轮换。

在消耗昂贵工作之前应用廉价验证。决定被拒绝的调用是否计入滥用限制、业务配额或两者。

### 审计决策链

记录足够重建调用的内容：

- 请求和追踪标识符。
- 已认证主体和发行者。
- 公共工具和后端路由。
- 描述符锁版本。
- 策略决策和原因。
- 延迟和结果类别。
- 适用时的 MRTR 轮次或任务标识符。

脱敏 bearer 令牌、授权码、刷新令牌、原始密钥和不必要的敏感参数。

### 请求范围的 SSE

正常 POST 可能在工作流期间返回请求范围的 SSE。关闭响应流会取消该进行中的现代 HTTP 请求。

不要创建单独的 GET 流，也不要承诺 Last-Event-ID 重放。这些是较旧的传输假设。

### 长期变更通知

对于列表和资源变更通知，当前客户端通过 POST 发送 `subscriptions/listen` 并接收 SSE 响应。通知过滤器使用精确的平面字段 `toolsListChanged`、`promptsListChanged`、`resourcesListChanged` 和 `resourceSubscriptions`：

```json
{
  "jsonrpc": "2.0",
  "id": "listen-tools",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

第一个事件确认支持的子集。其订阅标识符是打开流请求的 JSON-RPC id：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": "listen-tools"
    },
    "notifications": {
      "toolsListChanged": true
    }
  }
}
```

然后网关仅转发已确认的更改类型。该流上的每个通知都在 `params._meta` 中携带相同的 `io.modelcontextprotocol/subscriptionId`。没有自动重放或自动重新监听。重新连接时，客户端重新打开订阅并刷新其依赖的列表。服务器发起的优雅关闭返回带有相同订阅 id 标记的最终完整结果。

现代化路径替换了 `resources/subscribe`、`resources/unsubscribe` 和未请求的独立 GET 流。仅在版本门控的旧路径中保留这些。

### 通过网关的 MRTR

当后端返回 `resultType: input_required` 时，网关可以转发该结果，前提是外部客户端支持所需的输入请求。按字节保留 `requestState`，除非网关故意终止并重新发出交互。

客户端使用新的 JSON-RPC id 和 `inputResponses` 重试原始公共工具。网关重新授权重试，检查相同的公共路由，然后转发新的后端请求。它不能假设较早的轮次授予了无限批准。

### Tasks 扩展路由

Tasks 是通过 `io.modelcontextprotocol/tasks` 标识的官方扩展。它们不是核心会话的替代。

客户端在请求范围的客户端能力内声明扩展，网关仅在能够端到端保留生命周期时才在发现中通告它。对于受支持的 `tools/call`，后端单独决定是返回普通结果还是 `resultType: task`。任务结果在结果中直接携带 `taskId`、`status`、时间戳、`ttlMs` 和可选的 `pollIntervalMs`。任务在该结果发送之前必须已经持久可读。

网关为不透明任务标识符记录已认证主体和后端路由。后续的 `tasks/get`、`tasks/update` 和 `tasks/cancel` 调用使用 `params.taskId` 作为 `Mcp-Name`，这为中间件提供了路由键。`tasks/get` 返回带有当前任务状态的 `resultType: complete`，并在终端状态中内联最终结果或协议错误。`tasks/update` 为待处理的任务输入发送带键的 `inputResponses` 并返回空完整确认。`tasks/cancel` 是具有空完整确认的合作意图，而不是保证工作停止。

不要实现新的 `tasks/list` 或 `tasks/result` 方法。它们属于较旧的实验模型。需要输入的任务通过 `tasks/get` 暴露完整的嵌入式请求；客户端通过 `tasks/update` 回答它们，而不是通过重试原始工具调用。客户端仍然按建议的间隔轮询；任务创建仍然由服务器指导。

持久的任务路由状态是按任务句柄键控的应用数据，而不是协议会话。

### 兼容性边界

如果网关必须服务较旧的客户端或后端：

- 明确检测时代。
- 将初始化、传输会话、GET 流、资源订阅和旧任务词汇保留在遗留适配器中。
- 切勿将遗留会话 id 泄漏到现代路由或授权中。
- 优先使用有界发现探测和显式回退策略，而非静默降级。

```figure
t3-gateway-funnel
```

## 构建它

`code/main.py` 实现了进程内协议网关和两个后端服务器。每个后端接收一个新鲜当前协议的请求。网关提供发现、用户过滤的确定性 `tools/list`、命名空间路由、注册表 `server.json` 加外部准入状态、描述符锁、RBAC、主体键控速率限制、审计决策和模拟的 `subscriptions/listen` SSE 确认。

模型接收解析的请求正文、路由头和已认证的 bearer 身份。它不是完整的 HTTP 适配器，也不解析 `Content-Type` 或完整的 `Accept` 合同。将其连接到第 09 课的 Streamable HTTP 适配器，它需要 `Content-Type: application/json` 和包含 `application/json` 和 `text/event-stream` 的 `Accept` 值。

运行它：

```bash
cd phases/13-tools-and-protocols/17-mcp-gateways-and-registries
python3 code/main.py
python3 -m unittest discover code/tests -v
```

演示打印外部请求 id 和新鲜后端请求 id，以便看到无状态跳跃。

## 使用它

用真实当前协议客户端替换进程内后端对象。保持相同的接缝：

- 连接前的准入记录。
- 能力暴露前的后端发现。
- 授权前的限定公共名称。
- 列表或调用前的描述符锁。
- 转发前的新鲜请求范围元数据。
- 返回前的结果验证。

## 交付它

本课程交付 `outputs/skill-gateway-bootstrap.md`。它生成一个现代网关设计，涵盖入口、发现、准入、命名空间、授权、缓存、流式传输、订阅、MRTR、Tasks、可观测性和遗留隔离。

## 练习

1. 将追踪上下文添加到外部和转发请求元数据，并在审计事件中记录关联。
2. 添加一个支持 Tasks 的后端并通过 `Mcp-Name` 中的任务 id 路由 `tasks/get`。
3. 更改一个后端描述符并证明发现和直接调用都被阻止。
4. 添加一个特定主体的服务器能力并解释为什么发现必须保持私有缓存。
5. 编写一个遗留适配器接口，而不向现代 `Gateway` 类添加任何遗留状态。

## 关键术语

| 术语 | 含义 |
|------|---------|
| MCP 网关 | 客户端和后端 MCP 服务器之间的策略和路由服务器 |
| 准入记录 | 允许一个后端进入网关的证据和策略决策 |
| 限定工具名称 | 稳定公共路由，如 `notes.search` |
| 描述符锁 | 在发现和分发期间检查的批准摘要 |
| 私有缓存范围 | 仅限于一个授权上下文的缓存结果 |
| 请求范围 SSE | 附加到一个 POST 请求的流式响应 |
| `subscriptions/listen` | 客户端打开的用于选定长期变更通知的 SSE 流 |
| 任务路由 | 从不透明任务 id 到其后端的应用程序映射 |
| 遗留适配器 | 旧握手和会话行为的显式版本门控边界 |

## 延伸阅读

- [Streamable HTTP 传输](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [服务器发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [官方注册表 server.json 要求](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [MCP Tasks 扩展](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
