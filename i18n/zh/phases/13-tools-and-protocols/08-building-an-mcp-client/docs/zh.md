# 构建 MCP 客户端：发现、路由与双时代回退

> 现代 MCP 客户端会在每次请求时重新声明其契约。其最棘手的兼容性决策在于：判断一个旧服务器是真正的旧，还是一个现代服务器在报告可纠正的错误。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 13，第 07 课
**预计时长：** 约 85 分钟

## 学习目标

- 为每个 MCP `2026-07-28` 请求附带当前元数据。
- 对 stdio 服务器执行 `server/discover` 探测，并选择双方都支持的版本。
- 仅对明确列入白名单的端点对受限制的旧时代探测进行授权。
- 仅在验证到对某一受支持修订版的 `initialize` 结果为正后，才接受旧时代。
- 合并确定性工具列表，避免静默覆盖冲突项。
- 将调用路由到拥有该工具的端点，而非凭空创建协议会话。

## 问题所在

一个代理主机通常会与多个 MCP 服务器通信。它必须发现每台服务器、合并工具目录、解析重名、路由调用，并从传输故障中恢复。

`2026-07-28` 修订版让稳态更简单，因为每次请求都是自包含的。但兼容性使启动过程更微妙。客户端可能遇到以下情况：

- 支持首选版本的现代服务器；
- 返回已知版本或请求头错误的现代服务器；
- 从未听说过 `server/discover` 的旧服务器；
- 在收到 `initialize` 前保持沉默的旧服务器。

将所有探测错误都视为旧时代信号是危险的。格式错误的现代请求、过载的服务器、死掉的进程以及旧服务器都可能产生相同的超时或连接关闭。这些信号具有歧义性。客户端必须在选择旧时代之前，将明确的操作员意图与正向协议证据相结合。

## 概念说明

### 端点记录，而非协议会话

为每个服务器进程或端点保留一条传输端点记录：

- 传输句柄或发送函数；
- 已选定的协议时代与版本；
- 上次发现的服务端能力；
- 上次确定的工具列表；
- 用于关联的挂起请求 id；
- 传输健康状态。

这是客户端账本记录，并非协议会话状态。在现代 MCP 下，服务器仍会在每次请求时收到当前版本与能力信息。

### 从头构建每个现代请求

```python
def modern_request(request_id, method, params, version, capabilities):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            **params,
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": version,
                "io.modelcontextprotocol/clientCapabilities": capabilities,
                "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
            },
        },
    }
```

不要只在连接对象上附加一次元数据就假定它已到达网络。应针对最终序列化后的请求进行时间戳标记与检查。

### 现代发现

`server/discover` 返回支持的版本、服务端能力、操作说明、缓存提示和推荐的服务端标识。客户端选择最高且双方均支持的非旧版本。

对于仅现代模式的客户端，发现步骤是可选的，但在 stdio 上建议执行。部分旧服务器在接受初始化之前就会接受某项操作，因此先发送 `tools/list` 可能会产生语义模糊的成功结果。`server/discover` 能建立清晰的断代边界。

### stdio 兼容性探测

双时代 stdio 客户端在向任何其他请求发送之前，先以自己的首选现代元数据发送 `server/discover`。有三种结果类型：

1. **DiscoverResult（发现结果）。** 服务器是现代的。选择双方共同支持的版本，并继续采用逐请求元数据模式。
2. **已识别的现代错误。** 服务器是现代的。对于 `-32022`，从 `data.supported` 中选择并重试，使用新的请求 id。对于请求头或能力错误，修正请求。不要发送 `initialize`。
3. **歧义信号。** 未识别的 JSON-RPC 错误、超时、连接关闭或空响应均不能标识时代。除非该确切端点已配置为兼容旧模式，否则应失败关闭。

已识别的现代协议错误包括：

- `-32020` HeaderMismatch（请求头不匹配）
- `-32021` MissingRequiredClientCapability（缺少必需的客户端能力）
- `-32022` UnsupportedProtocolVersion（不支持的协议版本）

即使该端点在旧模式白名单中，已识别的现代错误仍表明其为现代服务器。一旦服务器证明自己理解现代错误词汇，再发送 `initialize` 就是降级。

不要把 `-32601` 当作正向旧时代证据。它仅使明确列入白名单的端点有资格获得一次旧模式探测。超时、连接关闭或空响应也适用同一规则。

### 白名单是操作员意图，而非证据

旧模式兼容性必须是某一固定端点配置的显式属性：

```python
client.add_server("archive", archive_transport, allow_legacy=True)
```

将该选择绑定到已配置的命令或端点。不要使用通配符，让任意服务器自行切换到弱语义。没有 `allow_legacy=True` 的端点在发现结果产生歧义时直接失败，且永远不会收到 `initialize`。

白名单授予的是探测权限，并不选定时代。客户端会在传输强制的截止期限下发送一次 `initialize`，然后要求满足以下条件全部成立：

- JSON-RPC `2.0` 响应，且请求 id 匹配；
- 恰好有一个 `result` 且没有 `error`；
- `protocolVersion` 处于客户端配置的旧时代修订版集合中；
- `capabilities` 字段为对象类型；
- `serverInfo` 对象中 `name` 和 `version` 字段为非空字符串。

超时、连接关闭、错误响应、结果格式异常、请求 id 不匹配或不支持的修订版，一律失败关闭。只有结构合法的正向结果才会选定旧时代。代码将 `legacy_probe_timeout_ms` 传递给传输适配器；真正的 stdio 或 HTTP 适配器必须强制执行该截止期限，而非仅仅记录。

为传输端点缓存已选定的时代。不要在每次调用前再次探测。

### 旧时代是兼容性分支

一旦受限探测返回了合法的正向旧时代证据，客户端就会按照该修订版定义的方式，严格使用已选定的旧版本：

1. 验证响应信封与关联 id。
2. 验证协商后的修订版在配置的旧时代集合中。
3. 记录已验证的能力与服务端标识。
4. 仅在全部检查通过后发送 `notifications/initialized`。
5. 在该传输生命周期内使用旧式请求形状。

这个分支的存在是为了与已知端点互操作。它不是新服务器或新请求的默认设计。如果传输重启或其端点发生变化，则丢弃端点-时代缓存并重新协商。

### 发现并缓存工具

对每个活跃端点调用 `tools/list`。现代结果包含 `resultType`、`ttlMs` 和 `cacheScope`。在正确的授权上下文中遵循新鲜度提示。过期后或收到订阅的列表变更事件时重新获取。

客户端必须将来自旧服务器的缺少 `resultType` 的响应视为 `"complete"`。不要对从前一个协商时代返回的响应强求现代缓存字段。

服务器应返回确定性顺序。客户端在合并前也应对列表排序，以避免本地注册表顺序依赖进程启动时机。

### 防碰撞命名空间合并

两台服务器可能都暴露了 `search` 工具。选择一种声明策略：

1. **冲突时加前缀。** 保留首个规范名称，将后续冲突项暴露为 `<server>/<tool>`。
2. **冲突时拒绝。** 不加载重复项，并给出清晰的配置错误提示。
3. **静默覆盖。** 永远不要使用此策略。它会隐藏模型选择的操作究竟由哪台服务器执行。

同时存储规范名称与本地名称。模型看到的是规范名称，而发出的 `tools/call` 使用拥有服务器所声明的本地名称。

### 路由调用

路由是一种纯查找：

```text
规范工具名称
  -> 端点名称 + 本地工具名称
  -> 新建 JSON-RPC 请求 id
  -> 现代请求元数据或明确的旧式形状
  -> 匹配的响应 id
```

当拥有该调用的传输不可用时，不要发送调用。重连或重启传输，然后重新运行发现与 `tools/list`。现代挂起请求在断开的传输上丢失时，若操作的 SAFETY 策略允许，可使用新的 JSON-RPC id 重试。

### 通知与订阅

现代列表与资源变更仅在客户端打开的 `subscriptions/listen` 流中到达。客户端发送通知过滤器，等待 `notifications/subscriptions/acknowledged`，并将事件与监听请求 id 在通知元数据中关联。

断开时，打开新的监听请求并重新获取相关列表或资源。现代流不会通过 `Last-Event-ID` 恢复。

### 服务端不发起请求

现代服务器不会以独立的 JSON-RPC 请求主动调用客户端去采样、 elicitation 或读取根目录。它们返回 `input_required`，客户端在满足内嵌输入请求后重试原始请求。

在满足输入时不要阻塞端点的响应读取器。保持关联，并为重试创建新的 JSON-RPC id。

```figure
tp-client-merge
```

## 使用它

`code/main.py` 使用进程内端点函数，使协议决策保持可见。它连接两个现代端点和一个明确列入白名单的旧端点，然后合并并路由它们的工具。传输可调用函数接收一个超时预算，从而确保兼容性分支无法掩盖无界的探测。

```bash
cd code
python3 main.py
python3 -m unittest discover tests -v
```

测试证明了常规演示遗漏的边界条件：

- 现代请求会重复元数据；
- `-32022` 重试现代发现而不初始化；
- 已识别的现代错误永远不会降级，即使对列入白名单的端点也是如此；
- 超时、连接关闭、空响应和未识别的错误在没有白名单的情况下不会触发 `initialize`；
- 列入白名单的端点只有在收到合法、受支持的 `initialize` 结果后才会进入旧时代；
- 格式错误或不支持的旧结果会让端点处于不可用状态；
- 成功选定的时代会在整个传输生命周期内被缓存。

## 交付物

本课交付 `outputs/skill-mcp-client-harness.md`。它为现代请求时间戳标记、stdio 时代协商、确定性命名空间合并、路由以及 fail-closed 旧模式兼容性分支搭建脚手架。

## 练习

1. 让一个假服务器返回 `-32022` 且没有双方共同支持的版本。确认客户端直接失败，而不是发送 `initialize`。
2. 将一个假旧服务器列入白名单，令其受限的 `initialize` 探测超时，并证明该端点保持 `unknown` 且不可用。
3. 为两种授权上下文分别添加 `cacheScope: "private"` 的工具列表。确认客户端不会将某一上下文的缓存结果共享给另一上下文。
4. 将冲突策略改为拒绝，并在两台端点名称都出现在错误信息中时使启动失败。
5. 添加一个有限状态的 `subscriptions/listen` 模拟器。在流丢失时，使用新的请求 id 重新监听并重新获取工具。

## 关键术语

| 术语 | 含义 |
|------|------|
| Peer（端点） | 客户端侧针对单个服务器传输及其发现数据的记录 |
| Protocol era（协议时代） | 现代逐请求元数据或旧式初始化语义 |
| Discovery probe（发现探测） | 用于识别 stdio 时代的初始 `server/discover` |
| Recognized modern error（已识别的现代错误） | 证明现代行为并禁止旧时代回退的错误 |
| Legacy allowlist（旧模式白名单） | 操作员配置，允许对固定端点进行单次受限的兼容性探测 |
| Positive legacy evidence（正向旧时代证据） | 针对显式支持的旧修订版的合法、可关联的 `initialize` 结果 |
| Merged namespace（合并命名空间） | 跨所有活跃端点的规范工具名称集合 |
| Collision policy（冲突策略） | 用于处理重复工具名称的前缀或拒绝规则 |
| Era cache（时代缓存） | 为单个传输端点存储的已选定的现代或旧时代行为 |
| Transport recovery（传输恢复） | 重启或重连、重新发现、重新列表，并使用新 id 安全重试 |

## 延伸阅读

- [MCP 规范 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Versioning](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
