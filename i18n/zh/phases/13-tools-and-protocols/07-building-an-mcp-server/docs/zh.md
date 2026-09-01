# 构建 MCP Server：无状态的 Python 与 TypeScript

> 现代 MCP server 不会记忆握手过程。它会在每个请求中验证元数据，执行一个处理器，并返回一个类型化的结果。

**类型：** 构建
**语言：** Python、TypeScript
**前置条件：** 第 13 阶段，第 06 课
**时间：** 约 85 分钟

## 学习目标

- 为 MCP `2026-07-28` 实现强制性的 `server/discover`。
- 在每个请求上验证协议版本和客户端能力。
- 以确定性列表顺序暴露工具、资源和提示。
- 在正确的结果上返回 `resultType`、服务器标识和缓存提示。
- 通过 newline-delimited stdio 在 Python 和 TypeScript 中提供相同的无状态契约。

## 问题所在

一个在第一条消息后存储客户端能力的 server 很容易构建，但难以运维。同一个进程可能服务多个顺序连接的客户端。远程请求可能落在不同的 worker 上。过期的能力声明可能跨越授权边界泄漏行为。

MCP `2026-07-28` 通过使每个请求自描述来解决该问题的协议部分。你的应用程序仍然可以保留持久化记录、作业或显式状态句柄。但它不能保留会改变后续请求解码方式的隐藏协议状态。

本课构建一个笔记 server 两份实现。Python 和 TypeScript 版本仅使用各自的标准库来实现协议核心。两者都暴露相同的方法并强制执行相同的传输契约。

## 概念解析

### 现代调度循环

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

三条 stdio 规则仍然重要：

- 仅向 stdout 写入 JSON-RPC 消息。将诊断信息发送到 stderr。
- 用换行符分隔消息，并刷新每个响应。
- 当 stdin 到达 EOF 时立即退出。

进程生命周期即传输生命周期。它不是现代 MCP 会话。

### 请求验证

每个请求都必须包含：

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

前两个字段是必需的。`clientInfo` 为推荐字段。验证存在的身份形状，但不要将其视为认证。

如果版本不受支持，返回代码 `-32022` 并附带 `requested` 和 `supported`。缺少请求元数据属于无效参数，代码为 `-32602`。切勿从之前的调用中填充缺失字段。

### 强制性发现

现代 server 必须实现 `server/discover`。完整的发现结果包括：支持的最新版本、能力、可选指令、缓存提示，以及结果 `_meta` 中的服务器标识：

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

发现不会解锁 server。客户端可以不调用发现而直接调用 `tools/list`，因为 `tools/list` 已携带相同的请求元数据。

### 工具

`tools/list` 返回确定性的工具描述符列表。稳定的排序有利于响应缓存并保持模型上下文稳定。结果还要求包含 `ttlMs` 和 `cacheScope`。

`tools/call` 返回内容块和 `isError`。当协议信封或方法参数无效时使用 JSON-RPC 错误。当有效工具调用运行但工具本身失败时使用 `isError: true`。

工具注解仍是提示而非强制：

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

主机应使用它们进行确认和展示。Server 仍必须强制执行真实的授权。

### 资源

`resources/list` 返回稳定的 URI 描述符。`resources/read` 返回类型化的内容。两者在 `2026-07-28` 中均可缓存，因此都包含 `ttlMs` 和 `cacheScope`。

对用户特定的笔记数据使用 `cacheScope: "private"`。共享缓存不得在跨授权上下文时重用私有响应。

现代变更投递不使用 `resources/subscribe`。客户端打开 `subscriptions/listen` 并请求 `resourceSubscriptions` 或列表变更类别。第 10 课将构建该流程。

### 提示

`prompts/list` 可缓存且具有确定性。`prompts/get` 使用参数渲染命名提示。渲染后的提示结果是完整的，但它不属于需要缓存提示的可缓存列表或读取结果。

### 每个成功结果都是类型化的

示例中对每个成功都使用一个包装器：

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

列表、读取和发现处理器会添加 `ttlMs` 和 `cacheScope`。集中此包装器可防止某个处理器悄悄省略现代结果字段。

### 无服务器发起的请求

现代 server 可以发送与客户端请求相关的通知，或在客户端打开的 `subscriptions/listen` 流上发送通知。它不得发送自己的 JSON-RPC 请求。

当处理器需要采样、提取或根输入时，它会返回 `input_required` 结果。客户端满足嵌入式输入请求，并使用新的请求 id 重试原始方法。第 11 课将介绍该多轮往返请求模式。

### 显式遗留兼容性

双时代 server 也可以在清晰的独立遗留分支上实现 `2025-11-25` 握手。当所需的现代 `_meta` 字段存在时选择现代行为，当接收到 `initialize` 时选择遗留行为。

不要将 `2026-07-28` 请求通过遗留握手路径。不要将现代 `resultType` 字段盖章到遗留初始化结果上。本课中的代码刻意仅支持现代协议，以便其不变量保持可见。

```figure
t3-dispatch-loop
```

## 使用方式

运行 Python server 的有限演示和测试：

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

使用 TypeScript runner 运行 TypeScript 端口：

```bash
npx tsx main.ts --demo
```

演示发送 `server/discover`，列出每个原语，调用工具，并展示不支持的版本错误。每个现代请求都会重复元数据。每个成功结果都包含服务器标识。

## 交付物

本课交付 `outputs/skill-mcp-server-scaffolder.md`。它生成一个现代 server 计划，包含发现契约、逐请求验证、确定性可缓存列表和可选的独立遗留适配器。

## 练习

1. 从一个请求中移除能力声明，并证明 server 不会重用之前请求的声明。
2. 反转 `TOOLS`、`PROMPTS` 和笔记插入顺序。确认所有列表结果保持稳定。
3. 添加一个破坏性的 `notes_delete` 工具，并在执行器内部要求授权检查。将 `destructiveHint` 保留为仅 UX 提示。
4. 添加 `resources/templates/list`，包含 `ttlMs`、`cacheScope` 和确定性排序。
5. 为 `2025-11-25` 构建单独的遗留适配器。添加测试证明现代请求永远不会进入该适配器。

## 关键术语

| 术语 | 含义 |
|------|---------|
| 无状态 server | 从其自身元数据而非协议会话记忆中处理每个请求 |
| `server/discover` | 强制性的现代方法，用于advertise版本和能力 |
| 完整结果 | 具有 `resultType: "complete"` 的成功现代结果 |
| 可缓存结果 | 具有 `ttlMs` 和 `cacheScope` 的发现、列表或资源读取结果 |
| 确定性列表 | 相同逻辑注册表产生相同的项目顺序 |
| 服务器标识 | 结果 `_meta` 中推荐的 `io.modelcontextprotocol/serverInfo` |
| 工具错误 | 有效工具调用但返回带有 `isError: true` 的内容 |
| 协议错误 | 通过 `error` 返回的无效 JSON-RPC 或 MCP 请求 |

## 进一步阅读

- [MCP 规范 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server 发现](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP 工具](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP 资源](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP 提示](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio 传输](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
