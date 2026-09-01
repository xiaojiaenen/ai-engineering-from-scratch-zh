```markdown
# 显式作用域与无状态征求

> Roots 已在 MCP 2026-07-28 中弃用，且从未作为安全沙箱使用。将作用域放在可见的工具参数或资源 URI 中，在服务端授权它，并在工具真正需要用户输入时使用 MRTR。用户看到决策，模型看到句柄，任何服务端实例都可以处理重试。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 13 · 07 (MCP server)、Phase 13 · 11 (无状态 MRTR)
**时间：** ~60 分钟

## 学习目标

- 用显式的工作区参数、资源 URI 或服务器配置替换已弃用的 Roots。
- 将作用域提示与授权、路径约束和操作系 统沙箱分离。
- 通过 MRTR `input_required` 结果交付表单模式的 `elicitation/create`。
- 在每请求客户端能力中通告征求支持并拒绝不支持的模式。
- 验证 `accept`、`decline` 和 `cancel` 作为独立的结果。
- 将破坏性确认绑定到经过身份验证的主体、原始参数、候选集和过期时间。

## 两个看似相似的问题

一个笔记工具收到如下请求："删除旧的 TPS 报告。"

服务器必须回答两个不同的问题。

1. 哪个工作区可以被此操作触及？
2. 三个匹配的笔记中，用户指的是哪一个？

前者是作用域和授权问题。后者是交互式消歧问题。将它们混在一起会导致危险的设计，例如将客户端提供的文件夹视为证明调用者可以删除其中所有内容。

## Roots 是一个迁移面

较早的 MCP 修订版本允许客户端通告 Roots 并在列表变化时通知服务器。Roots 只是信息性指引。它们不限制服务器进程可以读取什么，不授权调用者，也不创建操作系统沙箱。

MCP 2026-07-28 弃用了 `roots/list` 和 `notifications/roots/list_changed`，用于新设计。推荐使用以下显式替代方案之一：

- 当作用域每次调用不同时，使用 `workspaceUri` 或 `directory` 工具参数。
- 当操作已经针对某个资源时，使用资源 URI。
- 当一个部署拥有一个固定工作区时，使用服务器配置。
- 当代码必须在技术上无法逃逸时，使用进程沙箱或监狱文件系统。

如果现有的 2026-07-28 集成在弃用窗口内仍需要 `roots/list`，则服务器将其嵌入到 MRTR `inputRequests` 中。它不得发送实时的反向请求。这是迁移适配器；新的处理器应接受显式作用域。

模型可以看见并复现一个显式句柄。隐藏的传输会话作用域更难检查、重放、审计和路由。

### 三层规则

一个显式 URI 本身并不授权。强制执行所有三层：

1. **授权：** 这个经过身份验证的主体是否被允许使用此工作区？
2. **约束：** 归一化的目标 URI 是否保持在授权的工作区边界内？
3. **沙箱：** 操作系统能否阻止被攻陷的服务端无论如何逃逸？

可运行的服务端维护一个授权工作区 URI 的白名单，归一化 percent-encoded 路径，检查真实的路径组件边界，并在删除前立即重新检查约束。

朴素的前缀字符串检查是错误的：

```text
allowed:   file:///work/notes
attacker:  file:///work/notes-evil/secret.md
traversal: file:///work/notes/%2e%2e/private.md
```

两个恶意路径都以误导性字符串开头。先归一化，再比较路径组件。生产级文件系统服务端还必须防御符号链接竞争和平台特定路径语义。

## 征求仍然存在，但交付方式改变了

征求是当前客户端在 `tools/call`、`prompts/get` 或 `resources/read` 期间收集用户输入的功能。方法名仍为 `elicitation/create`。改变的是线上传输方向。

2026-07-28 的服务端不发送反向 JSON-RPC 请求。它返回一个 `InputRequiredResult`：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "delete_choice": {
        "method": "elicitation/create",
        "params": {
          "mode": "form",
          "message": "选择一个匹配的笔记并确认删除。",
          "requestedSchema": {
            "type": "object",
            "properties": {
              "note_id": {
                "type": "string",
                "enum": ["note-3", "note-7", "note-14"]
              },
              "confirm": {"type": "boolean"}
            },
            "required": ["note_id", "confirm"]
          }
        }
      }
    },
    "requestState": "integrity-protected-delete-state"
  }
}
```

主机渲染表单。用户可以接受、明确拒绝或关闭它。客户端随后用新的 id 重试原始 `tools/call`：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes_delete",
    "arguments": {
      "workspaceUri": "file:///Users/alice/Documents/Notes",
      "title": "TPS report"
    },
    "inputResponses": {
      "delete_choice": {
        "action": "accept",
        "content": {"note_id": "note-14", "confirm": true}
      }
    },
    "requestState": "integrity-protected-delete-state",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

两次调用之间没有协议会话。服务端验证回显的状态，根据预期 schema 验证响应，检查所选笔记是否在签名候选集中，重新授权工作区，重新检查约束，然后执行删除。

## 能力协商是每请求的

支持表单模式征求的客户端声明：

```json
{
  "io.modelcontextprotocol/clientCapabilities": {
    "elicitation": {"form": {}}
  }
}
```

空的征求能力 `"elicitation": {}` 为了兼容性仍然等同于仅表单支持。显式的 `"elicitation": {"form": {}}` 也支持表单模式。仅 URL 的声明 `"elicitation": {"url": {}}` 不支持。即使之前的请求通告了表单模式，服务端也不应嵌入当前请求能力中不存在的模式。

每个请求也携带 `io.modelcontextprotocol/protocolVersion`。缺失或非字符串版本返回 `-32602`。不支持的字符串返回 `-32022`，带有精确的 `supported` 和 `requested` 数据。缺失或仅 URL 的征求支持返回 `-32021`，`data.requiredCapabilities` 设置为 `{"elicitation":{"form":{}}}`。

没有 JSON-RPC `id` 的包是一个通知。处理它而不产生 JSON-RPC 成功或错误响应。在 Streamable HTTP 上，被接受的通知接收 `202 Accepted`，无主体。

`clientInfo` 应包含用于诊断，但它是自我报告的，不能用于授权识别用户。

服务端实现 `server/discover` 并返回 `supportedVersions`、能力、`ttlMs` 和 `cacheScope`，带有 `resultType: "complete"`。它不为这种现代设计通告 Roots。因为它通告了工具，所以还实现了强制性的 `tools/list`。该结果返回确定性的 `notes_delete` 描述符、有效的对象 `inputSchema`、服务器身份元数据和公共缓存提示。

## 表单模式

表单模式使用为可用对话框设计的受限 JSON Schema。根是对象，其属性是扁平的原语字段或支持的枚举数组。深层嵌套对象和通用文档 schema 不属于确认对话框。

在以下情况使用表单模式：

- 在多个候选中选择一个；
- 确认破坏性操作；
- 收集非敏感偏好；
- 收集少量必须由用户而非模型决定的值。

不要使用表单模式收集密码、API 密钥、访问令牌或支付凭证。这些秘密会经过 MCP 客户端并可能到达日志或模型上下文。

服务端重新验证返回的内容。客户端表单验证改善用户体验，但不建立信任。

## URL 模式

URL 模式发送用于带外交互的安全 Web URL：

```json
{
  "method": "elicitation/create",
  "params": {
    "mode": "url",
    "message": "连接报告服务以继续。",
    "url": "https://mcp.example.com/connect/report-service"
  }
}
```

当敏感信息必须直接发送到服务端控制的 Web 流程（如第三方授权）时使用它。客户端显示完整的目标地址并在打开之前获得同意。不得预取 URL。

`accept` 响应意味着用户同意打开 URL。它不证明外部流程已完成。重试时，服务端检查自己的状态，要么完成操作，要么返回另一个 `input_required` 结果。

URL 征求不是 MCP 客户端和 MCP 服务端之间授权的替代品。它是 MCP 服务端代表用户需要执行的交互。服务端必须将浏览器用户绑定到启动 MCP 操作的同一经过身份验证的主体。

## 响应分支

将操作视为产品决策，而非别名：

| 操作 | 含义 | 安全的服务器行为 |
|------|------|------------------|
| `accept` | 用户提交了交互 | 验证内容并继续 |
| `decline` | 用户明确拒绝 | 返回完整的非错误拒绝结果 |
| `cancel` | 用户关闭或未能完成 | 安全停止并允许稍后重试 |

永远不要将缺失的内容解释为同意。永远不要将拒绝转换为重复提示循环。

## 保护破坏性 MRTR 状态

候选列表不能仅存在于提示或未签名的 Base64 值中。客户端控制它发送回的所有内容。

此课程签署一个状态负载，包含：

- 经过身份验证的主体；
- 原始方法；
- `workspaceUri` 和 `title` 的摘要；
- 表单中显示的允许笔记 id；
- 操作阶段；
- 短过期时间。

在变更之前，服务端还检查实时笔记记录。这捕获了删除竞争和表单显示后目标移到工作区之外的情况。

对于一次性财务或不可逆操作，仅 HMAC 不能防止在过期时间内重放有效的状态。在多个处理器实例共享的重放存储中精确一次存储和消费一个 nonce。课程注入一个有界的、TTL 修剪的存储，并在执行内存删除时保持其原子声明。生产数据库应在一个事务或等效的条件写入边界中耦合 nonce 声明和变更。

在声明 nonce 之前验证交互。格式错误的响应或 `cancel` 不执行任何变更，并在过期之前使状态可重试。显式的 `decline` 是终态的，因此课程消费 nonce 而不删除任何内容。

```figure
t3-roots-boundary
```

## 构建它

`code/main.py` 演示了一个现代的 `notes_delete` 工具：

- `tools/list` 返回一个确定性的、可缓存的描述符，包含所需的工作区和标题 schema。
- 作用域是显式的 `workspaceUri` 参数。
- 服务器配置为该课程主体授权该工作区。
- URI 归一化拒绝前缀混淆和编码遍历。
- 每个破坏性删除都需要表单模式征求。
- 征求通过 `resultType: "input_required"` 传递。
- 签名的 `requestState` 绑定确切的候选列表和原始参数。
- 注入的重放存储拒绝跨服务端实例的同一接受或拒绝状态。
- 重试使用一个新的请求 id 并返回 `resultType: "complete"`。

数据存储是内存中的，因此协议行为易于检查。安全规则在数据库中保持不变。

## 使用它

从仓库根目录：

```bash
cd phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/code
python3 main.py
python3 -m unittest discover tests -v
```

预期检查点：

- 发现通告没有 Roots 的工具。
- 工具发现返回带有 `resultType`、服务器身份和缓存提示的 `notes_delete`。
- 请求 id `1` 在 `inputRequests.delete_choice` 中返回表单。
- 请求 id `2` 回显签名状态并完成删除。
- 前缀路径和编码遍历路径都因约束失败。
- 更改的标题不能重用原始确认状态。
- 一个拒绝保持笔记不变。
- 两个共享笔记和重放状态的服务端对象不能都执行一个确认。
- 空声明和显式表单声明有效，而仅 URL 支持返回精确的 `-32021` 表单要求。
- 不支持的版本失败使用精确的 `-32022` 数据形状。
- 没有 id 的通知不产生 JSON-RPC 响应。

## 交付

`outputs/skill-elicitation-form-designer.md` 设计了显式作用域、授权检查、MRTR 表单、响应分支和状态绑定。它拒绝将已弃用的 Roots 视为沙箱或通过表单模式收集机密。

## 练习

1. 将内存重放存储替换为 SQLite。使用一个事务声明 nonce 并删除笔记，然后证明两个进程不能都提交。
2. 添加 `url` 能力协商和带外设置流程。将第三方凭证保持在 `inputResponses` 之外。
3. 用临时 SQLite 数据库替换内存笔记映射。在变更事务内重新检查授权和约束。
4. 为真实文件系统实现添加符号链接策略。解释为什么仅 URI 词汇约束不能阻止符号链接逃逸。
5. 设计一个 2025-11-25 适配器，将现代 MRTR 处理器输出映射到遗留服务端发起的征求。将其与当前处理器隔离。

## 关键术语

| 术语 | 2026-07-28 中的含义 |
|------|----------------------|
| Roots | 已弃用的信息性工作区提示，不是授权或沙箱化 |
| 显式作用域 | 请求参数中可见的工作区、目录或资源句柄 |
| 约束 | 归一化的路径组件检查，将目标保持在边界内 |
| 征求 | 在 MCP 操作期间获取用户输入的客户端功能 |
| 表单模式 | 使用受限扁平 schema 的带内结构化用户输入 |
| URL 模式 | 用于敏感或外部工作流的带外交互 |
| MRTR | 无状态的 input-required 结果后跟一个新重试 |
| `requestState` | 服务端精确回显和完整性检查的不透明状态 |
| Decline | 用户明确拒绝 |
| Cancel | 关闭或不完整交互，未获得批准 |

## 遗留兼容性

对于固定到 2025-11-25 的对等端，`roots/list`、`notifications/roots/list_changed` 和实时服务端发起的 `elicitation/create` 可能仍然存在。标记该适配器为遗留。不允许遗留 Root 列表绕过服务端授权，也不将协议会话假设带入现代处理器。

## 延伸阅读

- [MCP 2026-07-28 Elicitation](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Roots deprecation](https://modelcontextprotocol.io/specification/2026-07-28/client/roots)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
```
