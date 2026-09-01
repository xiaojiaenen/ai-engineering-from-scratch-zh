# Model Context Protocol (MCP)

> MCP 为 AI 主机提供了一个协议，用于发现和调用工具、资源和提示。2026-07-28 版本修订使该协议无状态：能力和版本上下文随每次请求携带，而非绑定在连接握手里。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 11 · 09（函数调用）、Phase 11 · 03（结构化输出）
**时间：** 约 75 分钟

## 学习目标

- 区分 MCP 主机、客户端、服务器、传输层和服务器原语。
- 构建符合 MCP 2026-07-28 要求的带元数据的 JSON-RPC 请求。
- 使用 `server/discover` 检查版本、身份和能力。
- 从工具、资源和提示中返回类型化且感知缓存的结果。
- 说明现代无状态 MCP 如何与握手时代的服务器互操作。
- 为服务器选择安全的状态、传输和审批边界。

## 问题背景

你的应用需要一个数据库查询、一个日历操作和一个文件读取器。在没有共享协议的情况下，每个 AI 主机都需要为这些相同能力编写自定义的发现、调用、错误处理、传输和授权胶水代码。

MCP 减少了这种集成矩阵。服务器发布标准的 JSON-RPC 接口。兼容的客户端可以发现该接口，将其呈现给模型或用户，调用它，并解释结果，而无需针对每个服务器的适配器。

一个容易忽略的重要边界是：MCP 标准化的是通信方式，并不决定模型应调用哪个工具、让不可信内容安全，或将无状态请求转化为持久应用状态。主机和服务器仍然负责这些决策。

## 概念说明

![MCP 主机、无状态请求和服务器原语](../assets/mcp-architecture.svg)

### 三种服务器原语

1. **工具（Tools）** 是可调用的操作。每个工具都有名称、描述、JSON Schema 输入和处理器。
2. **资源（Resources）** 是具名的、URI 寻址的内容，客户端可以读取。
3. **提示（Prompts）** 是可复用的模板，主机可以展示给用户。

主机是 AI 应用程序。该主机内的 MCP 客户端与一个服务器通信。传输层承载两者之间的 JSON-RPC 消息。

### 无状态请求取代握手

MCP 2026-07-28 移除了 `initialize` 和 `notifications/initialized`。同时移除了协议级会话。每个请求都在 `params._meta` 中携带了解释它所需的上下文：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

协议版本和客户能力为必填项。客户身份信息为推荐项。缺少 `_meta`、缺少必填字段或必填字段类型错误都属于格式不良，返回 Invalid Params（`-32602`）。服务器不支持的格式正确的版本字符串返回 `UnsupportedProtocolVersionError`（`-32022`）。服务器可以在没有恢复先前协商记录的情况下处理有效请求。

无状态并不意味着应用永远不能维护状态。它意味着状态不隐藏在 MCP 连接或 `Mcp-Session-Id` 背后。如果工作流需要连续性，服务器会生成一个不透明句柄，客户端在后来的调用中将该句柄作为普通工具参数传递。授权仍需在每个请求上进行检查。

### 发现和版本选择

每个现代服务器都实现了 `server/discover`。返回结果会公示支持的版本、能力和服务器身份：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "complete",
    "supportedVersions": ["2026-07-28"],
    "capabilities": {
      "tools": {},
      "resources": {},
      "prompts": {}
    },
    "ttlMs": 3600000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "demo-server",
        "version": "1.0.0"
      }
    }
  }
}
```

客户端也可以直接调用另一个方法并处理版本错误，但发现机制让能力展示和版本选择更加明确。不支持的版本返回 `UnsupportedProtocolVersionError`，错误码为 `-32022`。其 data 中包含 `supported`（服务器修订版本数组）和 `requested`（被拒绝的版本）。

在 stdio 上，双时代客户端通过 `server/discover` 进行探测。发现结果或类似 `UnsupportedProtocolVersionError` 的已识别现代错误，标识出这是一台现代服务器。任何不被识别为现代的错误或超时，则允许回退到 2025-11-25 的 `initialize` 流程。遗留行为是兼容代码，而非现代默认行为。

### 结果明确

每个核心 2026-07-28 结果都带有 `resultType`：

- `complete` 表示操作完成。
- `input_required` 表示服务器需要通过"多轮往返请求（Multi Round-Trip Requests）"模式再获取一轮交互。核心服务器仅可从 `tools/call`、`resources/read` 或 `prompts/get` 返回该值。

客户端应将省略 `resultType` 的遗留结果视为 complete。

服务器应在每个结果的 `_meta` 中包含 `io.modelcontextprotocol/serverInfo`。此身份信息是自报的，仅用于展示、日志和调试，而非安全决策依据。

列表和读取结果还携带 `ttlMs` 和 `cacheScope`。确定性排序的 `tools/list` 结果加上新鲜度提示，让客户端可以安全缓存发现结果，并提升提示缓存的稳定性。`cacheScope: public` 允许共享缓存；`private` 将复用限制在调用上下文内。

### 传输格式和传输层

MCP 使用基于 stdio 或 Streamable HTTP 的 JSON-RPC 2.0。

- 请求包含 `jsonrpc`、`id`、`method` 和 `params`。
- 响应包含匹配的 `id`，以及 `result` 或 `error`。
- 通知没有 `id`，也不期望收到响应。

现代 Streamable HTTP 暴露一个接受 POST 请求的端点。每条 JSON-RPC 消息各有一个 POST。请求 POST 接收一个 JSON 对象，或一个以最终响应结束的请求级 Server-Sent Events 流。被接受的通知 POST 返回 HTTP 202，无响应体；本核心修订版未定义 Streamable HTTP 上从客户端到服务器的通知。

2026-07-28 版本中没有独立的 MCP GET 流、DELETE 会话端点、`Mcp-Session-Id` 或 `Last-Event-ID` 重放。长期变更通知使用 `subscriptions/listen` POST，其响应保持为 SSE 流打开。

### 无需服务器发起请求的客户端输入

较早的版本允许服务器通过流发送 `sampling/createMessage`、`roots/list` 或 `elicitation/create` 等请求。当前协议改用多轮往返请求（MRTR）模式。符合条件的工具调用、资源读取或提示获取会返回 `resultType: input_required`，并附带至少一个 `inputRequests` 或 `requestState`。客户端收集所有请求的输入，使用新的 JSON-RPC ID 和对应的 `inputResponses` 重试原始方法，并在提供了 `requestState` 时原样回显。如果不存在 `inputRequests`，则重试时省略 `inputResponses`。

Roots、Sampling 和 Logging 仍然可用，但已弃用，因此新实现不应采用它们。现有的 Roots 或 Sampling 请求通过 MRTR 的 `inputRequests` 传递，而非作为独立的服务器到客户端 JSON-RPC 请求。应优先使用显式的文件或目录参数、资源 URI、服务器配置和直接模型提供商集成。使用 stderr 进行 stdio 诊断，使用 OpenTelemetry 进行生产遥测。

```figure
mcp-nxm-collapse
```

## 构建

### 步骤 1：注册服务器接口

尽管请求契约已变更，注册仍然简单：

```python
server = MCPServer("demo-server")

@server.tool(
    "add",
    "Add two integers.",
    {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    }
)
def add(a: int, b: int) -> dict:
    return {"sum": a + b}
```

`code/main.py` 中的自带实现还注册了资源与提示。它刻意使用标准库，让你可以看到每个信封结构，而不是将协议委托给 SDK。

### 步骤 2：在每个请求中附加元数据

```python
def request(method, params=None):
    body_params = dict(params or {})
    body_params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "demo-client",
            "version": "1.0.0"
        }
    }
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": body_params
    }
```

不要仅在连接对象中缓存此元数据。服务器会在每次请求时验证它。

### 步骤 3：可选，在列出前先发现

调用 `server/discover`，选择一个支持的版本，然后调用 `tools/list`。如果你已知道版本并能处理 `-32022`，直接调用 `tools/list` 也是有效的。

演示版本以名称顺序返回工具列表，并附加 `ttlMs`、`cacheScope`、`resultType` 和服务器身份。工具调用返回完整且不可缓存的结果，因为其输出可能依赖于当前状态。

### 步骤 4：将同一请求映射到 HTTP

远程 `tools/call` POST 包含与 JSON-RPC 主体镜像的头部：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: add
```

`MCP-Protocol-Version` 头部必须与 `_meta` 中的版本匹配。`Mcp-Method` 在每个 JSON-RPC 请求上为必填项，且必须与 `method` 匹配。`Mcp-Name` 仅在 `tools/call`、`resources/read` 和 `prompts/get` 上为必填项，且必须与工具名称、资源 URI 或提示名称匹配。缺少必填头部或不匹配将返回 HTTP 400，错误码为 `HeaderMismatch`（`-32020`）。

### 步骤 5：在协议状态之外强制安全策略

- 在每个 HTTP 请求上验证授权和受众。
- 将本地服务器绑定到 localhost，并在 Streamable HTTP 上验证 `Origin`。
- 将可变工具标记为 `destructiveHint: true` 并要求主机审批。
- 显式传递目录和文件范围，而不是依赖已弃用的 Roots。
- 将资源和工具输出视为不可信数据。
- 在 stdio 下保留 stdout 仅用于 JSON-RPC；将诊断写入 stderr。

## 使用

从其目录运行课程：

```bash
python3 code/main.py
cd code
python3 -m unittest discover tests -v
```

第一行应报告发现协议版本为 `2026-07-28` 的 `demo-server`。然后检查 `MCPClient.request`：它应为每次调用重建 `_meta`。从其中一个请求中移除元数据，观察服务器的拒绝行为。

## 交付

`outputs/skill-mcp-server-designer.md` 将领域转化为无状态 MCP 设计。其验收门禁包括：发现结果、逐请求元数据策略、确定性缓存感知列表、显式状态句柄、传输头部、授权和审批规则。

## 深入 MCP

本课提供了协议模型。Phase 13 将四个生产边界拆分为独立的构建与验证课程：

1. [MCP 工具契约与内容](../../../13-tools-and-protocols/28-mcp-tool-contracts-and-content/docs/en.md) 涵盖闭合输入架构、结构化内容、路由元数据、不透明分页、完成授权，以及协议错误与工具域错误之间的区别。
2. [MCP 可靠性、取消和流量控制](../../../13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/docs/en.md) 涵盖请求取消、持久任务取消、截止时间、幂等性、背压、代理缓冲和重连行为。
3. [MCP 注册表供应链、准入、漂移和回滚](../../../13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/docs/en.md) 涵盖命名空间证明、制品溯源、不可变 pin、实时漂移、注册表状态、准入证据和回滚。
4. [MCP 合规工程](../../../13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/docs/en.md) 涵盖金标准和负向传输记录、严格版本时代、SDK 差异、代理证据、脱敏、健康门禁和发布回滚。

当服务器需要跨越团队或信任边界时，请按顺序学习这些课程。它们将引导你从"方法可用"走向"契约在部署全程保持安全和可诊断"。

## 练习

1. 添加一个 `subtract` 工具，确认 `tools/list` 仍按字母顺序排列。
2. 移除协议版本键，验证 Invalid Params（`-32602`）。然后发送格式正确但不被支持的版本 `2025-11-25`，验证 `-32022`，确认 `requested` 回显该修订版，并从 `supported` 中作出选择。
3. 在为创建操作添加服务器生成的 `draftId` 后，要求将其作为更新操作的参数传入。说明为什么这是应用状态而非协议会话。
4. 从一个需要用户确认的工具返回 `input_required`。使用新 ID、一个 `inputResponses` 条目和精确的 `requestState` 重试原始调用，而不是发明一个服务器到客户端的 JSON-RPC 请求。
5. 勾勒一个双时代 stdio 客户端的设计。将发现结果或已识别的现代错误视为现代行为，仅在遇到未识别错误或超时时才允许回退到 `initialize`。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| MCP | "LLM 的工具协议" | 用于服务器发现、工具、资源、提示和扩展的 JSON-RPC 协议 |
| Host | "AI 应用" | 拥有模型和 UI，并挂载一个或多个 MCP 客户端 |
| Client | "连接器" | 代表主机与一个服务器通过 MCP 通信 |
| Stateless MCP | "无会话" | 每个请求携带版本和能力信息；协议状态不按连接键控 |
| `server/discover` | "能力探测" | 公示版本、能力和身份的服务器必填方法 |
| `resultType` | "结果状态" | 将结果标记为 `complete` 或 `input_required` |
| State handle | "工作流 ID" | 由服务器生成、作为普通参数传递的应用标识符 |
| Streamable HTTP | "远程传输层" | 单个 POST 端点，支持 JSON 或请求级 SSE 响应 |
| MRTR | "询问并重试" | 结果中嵌入输入请求，随后重试原始操作 |

## 延伸阅读

- [MCP 2026-07-28 关键变更](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP 服务器发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP 多轮往返请求](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 已弃用特性](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
