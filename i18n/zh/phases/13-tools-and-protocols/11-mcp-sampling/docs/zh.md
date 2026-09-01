# MCP Model Input: Sampling 迁移与无状态 MRTR

> MCP 2026-07-28 弃用 Sampling，并为新设计移除服务端到客户端的请求通道。如果现有工作流仍需要客户端的模型，服务端返回 `input_required` 结果，客户端则携带模型输出重试原始请求。推理循环在协议层变为显式、有界且无状态。

**类型:** 构建
**语言:** Python
**前置知识:** Phase 13 · 07（MCP 服务端）、Phase 13 · 10（资源和提示）
**时间:** 约 75 分钟

## 学习目标

- 解释为何 MCP 2026-07-28 弃用 Sampling，并为新服务端选择直接模型集成方案。
- 实现一个通过多轮往返请求（MRTR）承载 `sampling/createMessage` 的兼容工作流。
- 在每个请求的 `_meta` 对象中包含协议修订和客户端能力信息。
- 返回 `resultType: "input_required"`，并使用新的 JSON-RPC id 重试原始方法。
- 使用完整性保护 `requestState`，并将其绑定到主体、方法、参数和有效期。
- 通过能力检查、审批、响应验证和轮次限制来约束模型辅助循环。

## 协议之前的决策

一个类似 `summarize_repo` 的工具需要完成两类工作：

1. 确定性工作：列出文件、读取允许的文件、验证路径并组装内容。
2. 模型工作：选择代表性文件并综合摘要。

你如今有两种有效的架构。

### 新服务端：直接与模型提供商集成

这是当前的默认方案。服务端负责模型选择、凭据、预算、重试和可观测性。它向 MCP 客户端返回一个普通的 `tools/call` 结果。

当服务端已是托管服务，或可预测的模型行为比使用主机模型更重要时，选择此方案。

### 现有 Sampling 工作流：迁移到 MRTR

Sampling 在弃用窗口期内仍然存在。面向 2026-07-28 的服务端无法向客户端发送实时 `sampling/createMessage` 请求，而是将其嵌入到 `InputRequiredResult` 中。

仅在使用客户端模型和凭据是真实产品需求时选择此兼容路径。记录删除计划，因为新的实现不应采用已弃用的 Sampling。

## 无状态契约

2026 年 7 月的协议没有 `initialize` 交换、没有 `notifications/initialized`，也没有 `Mcp-Session-Id`。每个请求都携带了原本在握手过程中的信息：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

服务端对每个请求验证修订版本。缺少或非字符串形式的版本为无效参数，返回 `-32602`。不支持的版本字符串返回 `-32022`，附带精确数据 `{"supported":["2026-07-28"],"requested":"<客户端版本>"}`。缺少 Sampling 能力时返回 `-32021`，`data.requiredCapabilities` 设为 `{"sampling":{}}`。

没有 JSON-RPC `id` 的封包是通知。接收方可处理它，但不发出成功响应或错误响应。Streamable HTTP 适配器对已接受的通知返回 `202 Accepted`，无响应体。

服务端还实现了 `server/discover`，携带精确的 `supportedVersions` 键、能力、`ttlMs` 和 `cacheScope`，以便客户端能够在调用工具前学习和缓存服务端契约。由于发现接口会发布 `tools`，服务端还需实现强制的 `tools/list`。其确定性 `summarize_repo` 描述符包括有效的对象 `inputSchema`、`resultType: "complete"`、服务端标识元数据和公开的缓存提示。

每个成功的现代结果都带有鉴别器：

- `resultType: "complete"` 表示操作已完成。
- `resultType: "input_required"` 表示客户端必须完成嵌入的请求并重试。
- 扩展可以定义额外的结果类型。Tasks 扩展在第 13 课中添加了 `"task"`。

## 单轮 MRTR

服务端在处理请求期间无法调用客户端。它返回如下结果：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "pick_files": {
        "method": "sampling/createMessage",
        "params": {
          "messages": [
            {
              "role": "user",
              "content": {
                "type": "text",
                "text": "Choose three representative files and return a JSON array."
              }
            }
          ],
          "systemPrompt": "Return only the requested value.",
          "modelPreferences": {
            "costPriority": 0.8,
            "intelligencePriority": 0.2
          },
          "maxTokens": 400
        }
      }
    },
    "requestState": "opaque-integrity-protected-value"
  }
}
```

客户端验证其支持 Sampling，应用其审批和模型策略，然后获得模型响应。接着发送一个新请求，使用不同的 JSON-RPC id：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "inputResponses": {
      "pick_files": {
        "role": "assistant",
        "content": {
          "type": "text",
          "text": "[\"README.md\", \"server.py\", \"docs/intro.md\"]"
        },
        "model": "host-model",
        "stopReason": "endTurn"
      }
    },
    "requestState": "opaque-integrity-protected-value",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}}
    }
  }
}
```

重试不是协议会话的延续。它是一个新请求，重复原始方法参数，仅添加当前轮的 `inputResponses`，并逐字节回显 `requestState`。

MRTR 仅允许用于 `tools/call`、`prompts/get` 和 `resources/read`。服务端不得从无关方法返回 `input_required`。

## 多轮状态

本课需要两次模型调用：

1. `pick_files` 返回一个 JSON 数组。
2. `summary` 返回最终文本。

每次重试只携带该轮的响应。因此服务端将阶段和验证后的中间数据放入下一个 `requestState`。

将该值视为由攻击者控制。仅对原始阶段名称签名是不够的。将状态绑定到：

- 经过认证的主体，而非自报的 `clientInfo`；
- 发起的方法；
- 原始参数的摘要；
- 短有效期；
- 当前阶段和验证后的中间值。

当不需要保密性时使用 HMAC。当不允许客户端读取状态时使用认证加密。对签名错误、过期值、主体变更或参数变更使用 `-32602` 拒绝。

客户端不得解析或修改 `requestState`。它的唯一职责是在重试时原样回显该字符串。

## 模型偏好是提示

`costPriority`、`speedPriority` 和 `intelligencePriority` 是独立的偏好。它们不是概率分布，也不必加总为一。客户端可以忽略它们，因为客户端拥有模型策略。

如果你维护传统的 Sampling 流程，请保持 `includeContext` 为 `"none"`。其他上下文模式会增加泄露风险，且它们本身也被弃用。在请求中传递最少的显式上下文。

## 安全不变式

客户端是嵌入 Sampling 请求的信任边界。

- 当策略需要审批时，向用户展示服务端要求模型执行的操作。
- 限制 MRTR 轮次。否则恶意服务端可以创建模型消耗循环。
- 在使用前验证每个采样响应，作为文件名、URL 或工具输入。
- 限制每轮的字节数和 token 数。
- 拒绝当前客户端能力中未声明的输入请求。
- 将模型输出排除在授权决策之外。
- 记录发起的方法和输入请求键，但不要记录敏感的提示内容。

`clientInfo` 和 `serverInfo` 是展示和诊断元数据。绝不将它们用作经过认证的身份。

```figure
t3-sampling-flip
```

## 构建

`code/main.py` 实现了完整的两轮流程，无需第三方包：

- `server/discover` 返回 `supportedVersions`，发布工具支持，并返回缓存提示。
- `tools/list` 返回确定性、可缓存的 `summarize_repo` 描述符，包含对象输入模式。
- `tools/call` 对每个请求验证元数据。
- 第一个结果嵌入用于文件选择的 `sampling/createMessage`。
- 第一次重试验证模型结果并嵌入第二个请求。
- HMAC 保护的 `requestState` 在独立请求间承载阶段信息。
- 最终结果使用 `resultType: "complete"`。

伪装的宿主模型使示例保持确定性。连接真实宿主时仅替换 `fake_host_model`。服务端状态机应保持确定性和可测试性。

## 使用

从仓库根目录：

```bash
cd phases/13-tools-and-protocols/11-mcp-sampling/code
python3 main.py
python3 -m unittest discover tests -v
```

预期检查点：

- 发现接口返回包含 `ttlMs` 和 `cacheScope` 的完整结果。
- 工具发现返回相同的排序描述符，包含 `resultType`、服务端标识和缓存提示。
- 缺失的能力和不支持的版本使用精确的 `-32021` 和 `-32022` 错误数据。
- 没有 id 的通知不产生 JSON-RPC 响应。
- 请求 id 为 `[1, 2, 3]`，证明每个 MRTR 轮次是独立的。
- 前两个结果为 `input_required`。
- 最终结果为 `complete`，包含所选文件和摘要。
- 在重试时更改原始参数会导致状态检查失败。

## 交付

`outputs/skill-sampling-loop-designer.md` 现在是迁移规划器。它首先决定是移除 Sampling 以采用直接模型集成。如果需要兼容性，它生成 MRTR 轮次、状态绑定、能力门控、预算、验证和删除计划。

## 练习

1. 将文件选择响应改为无效 JSON。确认服务端返回 `-32602` 而非信任模型输出。
2. 在首次调用和重试之间更改 `audience`。解释密封状态如何阻止跨请求重用。
3. 添加第三轮，要求宿主对摘要进行审查。在签名状态中携带之前的摘要，并将整个流程限制在三轮内。
4. 通过用服务端拥有的模型适配器替换伪装宿主回调来移除 Sampling。列出哪些审批、计费、可观测性职责转移到服务端。
5. 添加使用超出截止时间一秒的状态值的过期测试。

## 关键术语

| 术语 | 2026-07-28 中的含义 |
|------|------------------------|
| Sampling | 已弃用的功能，向客户端的模型请求补全 |
| MRTR | 用于请求期间需要客户端输入的无状态重试模式 |
| `InputRequiredResult` | 带有 `resultType: "input_required"` 的结果 |
| `inputRequests` | 服务端分配的嵌入提示、采样或根请求的映射 |
| `inputResponses` | 当前轮的客户端结果，按 `inputRequests` 的键形式组织 |
| `requestState` | 由客户端原样回显、由服务端验证的不透明服务端状态 |
| `resultType` | 现代 MCP 结果必需的鉴别器 |
| Direct model integration | 需要模型推理的新服务端推荐的替代方案 |
| Capability gate | 防止发送客户端未声明的嵌入请求的规则 |
| Loop budget | 操作允许的最大轮次、token、字节、时间和花费 |

## 遗留兼容

固定在 2025-11-25 的客户端仍可通过实时连接使用旧的服务端发起的 `sampling/createMessage` 流程。仅在版本特定适配器中保留该行为。不要将使服务端会话依赖的路径作为 2026-07-28 服务端的架构。

官方 SDK 可以为旧版对端翻译现代 `input_required` 处理器。该适配层是兼容边界，而非添加新会话依赖逻辑的许可。

## 延伸阅读

- [MCP 2026-07-28 多轮往返请求](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 变更日志](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP Sampling 弃用](https://modelcontextprotocol.io/seps/2577-deprecate-roots-sampling-and-logging)
- [MCP 2026-07-28 服务端发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
