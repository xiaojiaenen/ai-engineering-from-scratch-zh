# MCP 安全：被污染的元数据、路由与 MRTR 状态

> 无状态不等于无信任。它意味着每次请求都会暴露服务器和网关用于独立验证调用所需的证据。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段 13 · 07（MCP 服务器）、阶段 13 · 08（MCP 客户端）
**时间：** 约 60 分钟

## 学习目标

- 将工具描述、注解、客户端信息和服务端信息视为不可信数据。
- 检测元数据投毒、描述符变更和跨服务器名称冲突。
- 验证 2026-07-28 请求元数据和 Streamable HTTP 路由头。
- 保护 MRTR `requestState` 免受篡改，并将确认绑定到精确参数。
- 将授权和速率限制应用于主体，而非已移除的协议会话。

## 问题所在

模型读取工具描述以决定调用什么。路由器读取工具名称以决定将请求发送到何处。用户读取标签以决定批准什么。一个恶意描述符可以同时针对这三方。

官方 MCP 安全指南明确指出：除非来自可信服务器，否则描述和注解应被视为不可信数据。即便来自可信服务器，部署信任也可能发生变化。服务器更新、被入侵的软件包、注册表错误或网关合并都可能改变模型所看到的内容。

当前协议也改变了安全边界。在 2026-07-28 中没有核心握手和传输会话。一种仅基于 `Mcp-Session-Id` 来锚定批准、速率限制或审计历史的安全设计，已经不适用于当前协议。

## 概念阐述

### 七个值得检查的攻击面

使用具体清单，而非模糊的"小心"指令。

1. **元数据投毒。** 描述中包含与声明的工具行为无关的指令。
2. **描述符偷梁换柱。** 之前已批准的名字、描述、Schema 或注解发生变化。
3. **跨服务器遮挡。** 两个后端暴露相同的无限定工具名，路由静默选择其中一个。
4. **头与体混淆。** `Mcp-Method` 或 `Mcp-Name` 与 JSON-RPC 请求不一致。
5. **能力升级。** 对等方声明某扩展或客户端功能，而服务器将该声明误认为授权。
6. **MRTR 状态篡改。** 客户端修改 `requestState`、回答不同的问题，或用不同参数重用确认。
7. **供应链身份混淆。** 将熟悉的品牌名视为发布者或服务端身份的证明。

这些攻击面相互重叠。哈希固定有助于防范描述符变更，但无法证明初始描述符是安全的。静态扫描能捕获明显的短语，却抓不住隐晦的指令。命名空间可防止一类冲突，却无法防止恶意命名的服务器。堆叠这些控制措施。

### 当前请求信封是证据，而非身份

每条 2026-07-28 请求都包含：

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "elicitation": {"form": {}}
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "security-lab",
      "version": "1.0.0"
    }
  }
}
```

在每条请求上验证版本和能力形状。用能力来选择兼容的响应形状。不要将 `clientInfo` 作为经过认证的主体来使用。它是自报告的。

对结果元数据中 `io.modelcontextprotocol/serverInfo` 也适用相同警告。它对日志和调试有用。它不是证书、注册表证明或授权决策。

### 在策略之前验证路由

对于 `tools/call`，Streamable HTTP 包含：

```text
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.export
```

头方法必须等于主体方法。头名称必须等于 `params.name`。在选择后端、应用 RBAC 或消耗速率限制令牌之前，发现不一致就拒绝，返回 `-32020`。

这一顺序关闭了一个常见的歧义：一个组件授权了主体，而另一个组件按头进行路由。

导线验证遵循一个精确序列。验证 JSON-RPC 和元数据类型，将头值与主体比较，然后检查匹配的版本是否受支持。不匹配的头返回 HTTP 400 配合 `-32020`。如果头和主体在不支持版本上达成一致，返回 HTTP 400 配合 `-32022`，`data` 严格为 `{"supported":["2026-07-28"],"requested":"<actual>"}`。未知方法返回 HTTP 404 配合 `-32601`。

当契约需要结构化恢复信息时，每个错误对象都包含可选的 `data`。通知没有 `id`，因此它永远不会收到 JSON-RPC 成功或错误响应。HTTP 接受的通知返回 202 配合空主体。

### 固定整个描述符

仅对描述哈希进行固定，会遗漏 Schema 和注解的变更。对用户已批准的那些描述符字段进行规范化并计算哈希：

```python
normalized = json.dumps(tool, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(normalized.encode()).hexdigest()
```

在限定键下存储该摘要，例如 `notes.export`，并与在本玩具示例之外的发布者证据和批准时间一并保存。

每次刷新时：

- 未知键：隔离，等待审查。
- 相同键，不同摘要：作为偷梁换柱进行隔离，直到重新批准。
- 重复的无限定名称：要求确定性的命名空间。
- 扫描器命中：阻止并审查完整描述符。

哈希相等证明的是稳定性，而非安全性。被投毒的描述符即使完美固定，仍然是被投毒的。

### 静态扫描是触发线

简单的模式可以标记角色标签、指令覆盖、隐藏、机密访问和被模糊的网络目标。它们足够便宜，可以在安装时或 CI 中使用。

它们不是语义证明。一个安全的描述可能在合法警告中包含被标记的短语。一个恶意描述可能避开所有短语。将扫描器输出视为审查证据，而非自动的清白评分。

### 合并前先命名空间

假设两个服务器都暴露了 `search`。决不要让发现顺序决定谁获胜。

```text
notes.search
issues.search
```

限定名是公开的网关名称。单独记录后端映射。稳定的名称使批准、审计、哈希固定和 `Mcp-Name` 路由指向同一对象。

### 能力是兼容性声明

每条请求的 `clientCapabilities` 告知服务器客户端能够处理哪些协议功能。它并不授予客户端访问工具、数据或操作的权限。

授权仍然来自经过认证的主体和资源策略。顺序如下：

1. 认证传输凭证。
2. 验证版本、头和请求形状。
3. 检查能力兼容性。
4. 授权主体、工具、资源和参数。
5. 执行或请求用户输入。

### 保护无状态的 MRTR 确认

高风险工具可能需要用户确认。当前 MCP 使用多轮往返请求（MRTR），而非服务端到客户端的回拨。

首次响应：

```json
{
  "resultType": "input_required",
  "inputRequests": {
    "confirm": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "导出笔记到归档？",
        "requestedSchema": {
          "type": "object",
          "properties": {
            "confirm": {"type": "boolean"}
          },
          "required": ["confirm"]
        }
      }
    }
  },
  "requestState": "opaque-integrity-protected-value"
}
```

客户端获取输入并以新的 JSON-RPC id 重试原始方法：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes.export",
    "arguments": {"query": "private", "destination": "archive"},
    "requestState": "opaque-integrity-protected-value",
    "inputResponses": {
      "confirm": {
        "action": "accept",
        "content": {"confirm": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

每个 `inputRequests` 值都是一个完整的嵌入请求，包含 `method` 和 `params`。其键必须与 `inputResponses` 中的对应条目匹配。表单 elicitation 使用对象根 `requestedSchema`，且客户端必须在服务器请求之前声明表单 elicitation 能力。

当前能力有两种有效的表单声明。`{"elicitation":{}}` 隐式支持表单 elicitation，而 `{"elicitation":{"form":{}}}` 显式声明。仅 URL 的声明 `{"elicitation":{"url":{}}}` 不支持表单请求。服务器返回 HTTP 400 配合 `-32021`，且 `data.requiredCapabilities` 等于 `{"elicitation":{"form":{}}}`。

将 `requestState` 视为恶意输入。签名或加密它、验证它，并在重放相关时将它与方法、工具、精确参数、目的、过期时间、主体和一次性 nonce 绑定。本课代码使用 HMAC 和精确参数匹配使该边界可见。

nonce 台账不应存在于单一网关对象内。可运行的模型注入一个有界、TTL 裁剪的重放存储，可被多个网关实例共享。其原子声明就是执行边界：只有经过验证的接受或显式的终止拒绝才会消费状态。格式错误的响应或 `cancel` 不执行任何操作，且在过期前始终保持可重试。生产集群需要在共享持久化存储中实现相同的条件性声明。

不要将隐藏的确认上下文存储在协议会话中。任何服务器实例都应能够验证重试。

### 高风险调用的双重规则

沿三个维度对调用进行分类：

- 它消费不可信输入。
- 它可以访问敏感数据。
- 它会导致后果严重的外部操作。

单个自动步骤不应同时结合这三项。拆分它、降低权限，或通过 MRTR 请求显式用户输入。这是一条设计启发式规则，而非协议能力。

### 在执行前降低权限

仅凭无状态并不是安全。它消除了隐藏的协议历史，但一个自包含的请求仍可能要求一个权限过高的处理器去泄露数据或做出不可逆的更改。安全来自于在每个边界处降低权限：

1. **类型化动词。** 暴露一个有界操作，例如 `archive_note`，而非可以表达无关权限的通用 `run` 或 `request` 工具。
2. **已验证的参数。** 在可行时使用闭合 Schema，拒绝未知字段，对标识符做一次规范化，限制大小，并在策略评估之前验证目标、租户和资源所有权。
3. **当前授权。** 将认证主体绑定到精确的动词、资源、环境和规范化参数。工具注解和客户端能力并不授予此权限。
4. **动作绑定的批准。** 对于后果严重的调用，将批准绑定到类型化动词和规范化参数的摘要、主体、过期时间和一次性策略。任何变更字段都需要新的决策。
5. **一等公民的拒绝。** 将模型拒绝、过期批准、用户拒绝和不安全目标建模为普通结果，不执行任何副作用。不要将拒绝转化为较弱的回退工具。
6. **脱敏的审计证据。** 记录谁请求、使用了哪个已批准描述符和策略版本、授权了哪个规范化目标、决策为何允许或拒绝、以及执行是否开始。存储摘要或脱敏值，而非机密。

每一步都收窄下一步组件可执行的范围。最终处理器应接收一个已经验证过的域命令，而非原始模型文本加宽泛凭证。在 MRTR 重试、任务更新或网关转发调用时，重复整个链路。早期的批准不会让后续请求变为可信会话流量。

### 当前与遗留交互路径

Roots、Sampling 和 Logging 对于新的 2026-07-28 实现已被弃用。网关可能仅作为版本门控的兼容路径而保留旧的请求通道代码。

不要围绕逐会话采样限制器构建新的防御。将配额应用于认证主体、发行人、资源、工具和 timeframe。对于当前的交互式工作，请检查 MRTR 输入请求和响应。

### 无状态传输检查

- 在单一 POST 端点接受现代 MCP 消息。
- 对现代 GET 和 DELETE 返回 405。
- 不要生成或依赖 `Mcp-Session-Id`。
- 将遗留会话和重放头忽略为权威输入。
- 为该 POST 返回 JSON 或请求作用域的 SSE。
- 仅在 opted-in 的长生命期变更通知中使用 `subscriptions/listen`。

```figure
tp-tool-poisoning
```

## 构建它

`code/main.py` 实现了一个小型进程内安全网关模型。它规范化并固定完整的工具描述符，报告元数据投毒和遮挡，验证现代请求信封和路由值，并执行带有签名 `requestState` 和注入共享重放存储的两轮确认导出。

模型在 HTTP 适配器解析 JSON 主体和路由头之后启动。它不验证 `Content-Type` 或 `Accept`。将同一个分发器连接到第 9 课的完整 Streamable HTTP 适配器，后者要求 `Content-Type: application/json` 且 `Accept` 值同时包含 `application/json` 和 `text/event-stream`。

运行它：

```bash
cd phases/13-tools-and-protocols/15-mcp-security-tool-poisoning
python3 code/main.py
python3 -m unittest discover code/tests -v
```

示例有意变异了一个描述符。扫描器和摘要比较各自产生独立发现。随后导出演示了 `input_required` 响应和无状态重试。

## 使用它

将 `SAFE_TOOLS` 替换为你自己已批准服务器的一份规范化快照。将凭证和机密移出快照。在更新其摘要之前，审查每一个新出现或已变更的描述符。

在网关上，在发现阶段运行相同检查，并在分发前再次运行。缓存可以减少发现工作，但当描述符变更时，缓存中的批准必须过期或被失效。

## 交付它

本课交付 `outputs/skill-mcp-threat-model.md`。它跨元数据、路由、能力、授权、MRTR、缓存、注册表和兼容边界，生成当前协议威胁模型。

## 练习

1. 将认证主体和当前授权决策绑定到密封的 MRTR 状态，然后拒绝以不同主体发起的重试。
2. 将内存重放存储替换为持久化条件插入，并证明两个进程不能同时申领同一 nonce。
3. 在重放申领之后但在模拟导出之前注入故障。定义并测试使恢复安全的事务或幂等规则。
4. 在不变更描述的情况下修改工具的 `inputSchema`。确认整描述符固定能够捕获它。
5. 添加一条策略：当 `tools/list` 因主体而异时拒绝公共缓存。
6. 在网关后模拟一个旧服务器。将所有握手和会话行为置于显式的 `2025-11-25` 兼容分支之下。

## 关键术语

| 术语 | 含义 |
|------|------|
| 元数据投毒 | 嵌入在工具描述符中的指令或欺骗性声明 |
| 偷梁换柱 | 对之前已批准描述符的变更 |
| 工具遮挡 | 由重复的无限定名称引起的歧义路由 |
| 头不匹配 | 路由头与 JSON-RPC 主体不一致，错误 `-32020` |
| 哈希固定 | 完整已批准描述符的摘要 |
| MRTR | 服务端请求输入的无状态响应与重试模式 |
| `requestState` | 必须视为不可信输入的透明往返值 |
| 能力声明 | 协议兼容性声明，而非授权 |
| 隐式表单支持 | 空的 `elicitation` 能力对象，等价于表单支持 |
| 限定工具名 | 稳定的网关名称，如 `notes.search` |

## 延伸阅读

- [MCP 安全与信任指南](https://modelcontextprotocol.io/specification/2026-07-28#security-and-trust--safety)
- [多轮往返请求](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Streamable HTTP 传输](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [已弃用功能](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
