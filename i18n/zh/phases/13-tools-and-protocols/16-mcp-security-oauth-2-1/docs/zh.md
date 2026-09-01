# MCP 授权：CIMD、发行方绑定、PKCE 与权限升级

> 远程 MCP 请求是无状态的，但其授权并非匿名。将每个凭据绑定到创建它的发行方，将每个令牌绑定到接收它的资源。

**类型：** 构建
**语言：** Python
**前置条件：** 第 13 阶段 · 09（传输层）、第 13 阶段 · 15（安全）
**时间：** 约 90 分钟

## 学习目标

- 通过受保护资源元数据发现授权服务器。
- 优先使用客户端 ID 元数据文档（CIMD），而非已弃用的动态客户端注册（DCR）。
- 当 DCR 兼容路径不可避免时，声明正确的 `application_type`。
- 验证授权响应中的 `iss`，并按发行方隔离凭据。
- 使用 PKCE、资源指示符、受众验证和增量作用域。
- 无需协议会话即可发送经过授权的 MCP 2026-07-28 请求。

## 问题

远程 MCP 服务器可能读取私人记录、写入外部系统或触发高成本操作。认证告诉服务器是谁提交了凭据。授权还必须回答：

- 哪个授权服务器签发了该凭据？
- 该令牌属于哪个 MCP 资源？
- 哪个客户端和重定向 URI 完成了流程？
- 用户批准了哪些操作？
- 这个精确的请求是否仍然符合该授权？

2026-07-28 授权配置文件强化了客户端注册和发行方处理。它优先使用客户端 ID 元数据文档，弃用动态客户端注册，要求在 DCR 上使用正确的 `application_type`，验证 RFC 9207 发行方响应，并禁止跨发行方重用凭据。

这些规则是对无状态核心方案的补充。它们不会恢复核心握手或 `Mcp-Session-Id`。

## 概念

### 了解三个角色

- **MCP 客户端：** 代表资源所有者发送请求。
- **MCP 资源服务器：** 接受访问令牌并提供 MCP 端点。
- **授权服务器：** 对资源所有者进行认证，收集用户同意并发放令牌。

资源服务器和授权服务器可以协同运营，但需保持其标识符和验证职责分离。

### 授权适用于 HTTP

MCP 授权规范适用于基于 HTTP 的传输。本地 stdio 服务器在进程和操作系统信任边界内运行。不要仅为形式上的对称而在 stdio 上添加虚假的浏览器 OAuth 流程。

对于远程 Streamable HTTP，在每个请求的 `Authorization` 头中发送 bearer 令牌。永远不要将其放在 URL 中。

### 从受保护资源元数据开始

资源服务器发布 RFC 9728 元数据：

```json
{
  "resource": "https://notes.example.com/mcp",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:delete", "notes:read", "notes:write"]
}
```

客户端从 MCP 资源 URL 出发，获取该文档，选择一个已公布的授权服务器，然后获取该服务器的 OAuth 或 OpenID Connect 元数据。

构造 RFC 9728 well-known URL 时保留资源路径。对于资源 `https://notes.example.com/mcp`，本课使用 `https://notes.example.com/.well-known/oauth-protected-resource/mcp`。丢弃 `/mcp` 后缀可能会选中同一源上不同受保护资源的元数据。

不要从主机名猜测授权服务器。不要从未经验证的错误体中发现的发行方跟随链接。维护一份客户端愿意信任的发行方策略。

### 验证授权服务器元数据

元数据应暴露端点和受支持的控件：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "code_challenge_methods_supported": ["S256"],
  "authorization_response_iss_parameter_supported": true,
  "client_id_metadata_document_supported": true
}
```

要求使用 S256 进行 PKCE。记录精确的发行方字符串。该精确值将成为注册和令牌存储的键。

### 遵循注册优先级

当客户端已与所选发行方建立明确关系时，使用预注册客户端信息。否则，当授权服务器公布支持时，优先使用客户端 ID 元数据文档。仅在以下机制均不可用时，将 DCR 作为已弃用的兼容回退路径，并提示输入客户端信息。

### 优先使用客户端 ID 元数据文档

客户端 ID 元数据文档为授权服务器提供了一个 HTTPS URL，它既是客户端标识符，也是其元数据的位置：

```json
{
  "client_id": "https://client.example.com/oauth/metadata.json",
  "client_name": "Notes 桌面客户端",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

授权服务器获取并验证该文档。`client_id` 必须是带有路径的 HTTPS URL，且文档内的值必须与该 URL 完全一致。必需的文档字段是 `client_id`、`client_name` 和 `redirect_uris`。`application_type` 在此示例中出现，但不是 CIMD 的必需项。它的新增强制使用场景专门针对 DCR 路径。

将获取文档视为 SSRF 敏感操作。解析并验证目标地址，拒绝回环、私有、链接本地及其他不允许的地址，在重定向和 DNS 变更后重新检查，限制重定向次数、字节数和超时时间，要求 JSON 格式，并仅根据已验证的 HTTP 缓存控制进行缓存。将 `client_name` 和其他展示字段视为不受信任的文本。

CIMD 消除了每次首次联系都生成新动态标识符的需求。但它不会消除重定向 URI 验证、发行方策略或用户同意。

### DCR 是兼容路径

动态客户端注册仍可用于旧版授权服务器，但对新的 MCP 实现已弃用。

使用 DCR 时，声明 `application_type`：

```json
{
  "client_name": "Notes 桌面客户端",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

- 桌面、移动、命令行和回环客户端使用 `native`。
- 远程托管的浏览器应用使用 `web` 及远程 HTTPS 重定向。

省略该字段可能导致在 OpenID Connect 注册实现中默认为 `web`，从而使合法的回环重定向失败。

将 DCR 代码置于显式的回退决策之后。不要在任意的 CIMD 验证失败后静默回退。这可能会将安全失败转化为更弱的注册路径。

### 将凭据绑定到发行方

在精确的发行方下存储由发行方签发的注册材料：

```text
issuer_credentials[issuer] = pre_registered_or_dcr_client
tokens[(issuer, resource)] = access_token
```

如果受保护资源发现从 `https://auth-one.example` 变为 `https://auth-two.example`，需重新评估信任。切勿将第一个发行方的客户端密钥、DCR 客户端 ID、注册访问令牌、刷新令牌或访问令牌发送给第二个。预注册和 DCR 客户端必须使用为新发行方签发的凭据。

CIMD 客户端 ID 不同，因为它是一个自托管的 HTTPS URL，而非由授权服务器签发的凭据。相同的 CIMD URL 可移植：新可信发行方获取并验证文档，无需 DCR 重新注册。授权响应和令牌仍在新发行方下验证和存储。

### 带 PKCE 的授权码流程

交互式流程为：

1. 生成高熵的 `code_verifier`。
2. 派生 S256 `code_challenge`。
3. 使用精确的 `client_id`、`redirect_uri`、`scope`、`code_challenge` 和 `resource` 发送授权请求。
4. 接收包含 `code` 以及在提供时包含 `iss` 的授权响应。
5. 在使用任何响应字段之前，对照记录的精确发行方验证 `iss`。
6. 使用 `code_verifier`、相同的重定向 URI 和相同的 `resource` 兑换代码。
7. 将所得令牌存储在 `(issuer, resource)` 下。

RFC 8707 的 `resource` 参数同时出现在授权请求和令牌请求中。它标识规范的 MCP 服务器 URI。

### 精确验证 `iss`

RFC 9207 防止将来自一个发行方的授权响应与来自另一个发行方的响应混淆。

当 `iss` 存在时，将其与记录的发行方进行比较，不进行大小写折叠、尾部斜杠变更、默认端口移除或百分号编码规范化。如果不匹配，不要对该代码采取任何行动，甚至不要显示该响应中攻击者可控的错误详情。

包含 `iss` 的授权服务器会在元数据中公布 `authorization_response_iss_parameter_supported: true`。当前客户端即使缺少该公布信息，仍会验证存在的 `iss`。

### 在 MCP 服务器验证受众

资源服务器只接受为其自身签发的令牌：

```text
token.issuer == configured_authorization_server
token.audience == canonical_mcp_resource
```

无效、已过期、错误发行方或错误受众的令牌返回 401。MCP 服务器不得接受或转发为本应服务于其他服务的令牌。

### 请求当前最小的作用域

从当前所需的作用域开始。如果后续工具需要更多权限，服务器将返回 403 及权威的作用域挑战：

```text
WWW-Authenticate: Bearer error="insufficient_scope",
  scope="notes:delete",
  resource_metadata="https://notes.example.com/.well-known/oauth-protected-resource/mcp"
```

客户端解释新的权限要求，获得用户同意，使用合并后的作用域集执行新的授权流程，并使用新的 JSON-RPC id 重试 MCP 请求。

不要假设被挑战的作用域是 `scopes_supported` 的子集。该挑战对当前操作具有权威性。

### 授权与无状态 MCP 线路

经授权的工具调用仍然携带完整的当前请求信封：

```text
POST /mcp
Authorization: Bearer <access-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.delete
```

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "notes.delete",
    "arguments": {"id": "note-7"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "oauth-lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

令牌授权主体。请求元数据协商协议行为。二者不可互相替代。

按固定顺序验证线路：JSON-RPC 和元数据类型、头与体相等性、然后验证协议支持。路由或版本头不匹配返回 HTTP 400 及 `-32020`。如果头和体对不支持的版本达成一致，则返回 HTTP 400 及 `-32022` 和 `data` 精确为 `{"supported":["2026-07-28"],"requested":"<actual>"}`。未知方法返回 HTTP 404 及 `-32601`。

所有请求错误，包括 401 无效令牌和 403 权限不足，都是带有原始请求 `id` 的 JSON-RPC 错误信封。结构化恢复信息属于可选的错误 `data`；`WWW-Authenticate` 保持为 HTTP 响应头。通知没有 `id`，因此不接收 JSON-RPC 响应体。接受的 HTTP 通知返回 202 及空响应体。

服务器实现 `server/discover` 并公布工具，因此也实现了强制性的 `tools/list` 方法。其工具描述符具有稳定的名称、描述和对象根 `inputSchema` 值。列表是确定性的，并返回 `resultType`、服务器身份元数据、有界的 `ttlMs` 和 `cacheScope`。发现功能和与用户无关的工具列表可在授权前可用。如果两者因主体而异，则应用正常策略和私有缓存。

### 禁止令牌透传

MCP 服务器不得将客户端的 MCP 访问令牌转发给下游 API。使用正确的受众获取单独的下游令牌，或使用显式的令牌交换设计。只有当服务拒绝为他人签发的令牌时，受众验证才有效。

### 刷新令牌

刷新令牌是可选的。当签发时，保密存储并按发行方和资源索引。不要假定它们存在。当授权服务器支持轮换时进行轮换，并检测已失效值的重复使用。

```figure
t3-scope-stepup
```

## 构建

`code/main.py` 是一个进程内协议和授权模拟器。它实现了受保护资源发现、授权服务器元数据、CIMD 注册、版本门控的 DCR 回退、应用类型检查、PKCE、发行方验证、资源绑定令牌、作用域权限升级、`server/discover`、`tools/list` 以及无状态工具请求。

该模拟器接收解析后的请求体和路由头。它不是完整的 HTTP 适配器，也不解析 `Content-Type` 或 `Accept`。将其连接到第 09 课的 Streamable HTTP 适配器，后者要求 `Content-Type: application/json` 且 `Accept` 值同时包含 `application/json` 和 `text/event-stream`。

运行它：

```bash
cd phases/13-tools-and-protocols/16-mcp-security-oauth-2-1
python3 code/main.py
python3 -m unittest discover code/tests -v
```

输出依次展示发现流程、CIMD 注册、普通读取操作、两次独立的作用域权限升级以及按发行方索引的凭据存储。

## 使用

将模拟器对象映射到生产组件：

- `ResourceServer.protected_resource_metadata` 成为 RFC 9728 端点。
- `AuthorizationServer.metadata` 成为 RFC 8414 或 OpenID Connect 发现。
- `Client.enroll` 成为 CIMD 解析加上显式的 DCR 兼容分支。
- 由发行方签发的客户端凭据和 `tokens_by_issuer_resource` 成为加密记录。CIMD URL 可能保持可移植，而其授权结果保持绑定到发行方。
- `ResourceServer.handle` 成为中间件，在分发前验证当前 MCP 头、令牌和工具作用域，同时将所有请求错误保持在匹配的 JSON-RPC 信封中。

## 交付

本课交付 `outputs/skill-oauth-scope-planner.md`。它现在设计注册优先级、绑定到发行方的凭据存储、应用类型、PKCE、资源指示符、作用域挑战以及当前无状态请求边界。

## 练习

1. 添加刷新令牌轮换并拒绝重用之前的刷新令牌。
2. 添加发行方白名单。在发行方变更时，仅复用可移植的 CIMD URL；拒绝所有先前发行方签发的凭据和令牌。
3. 为授权码添加过期时间，并确认延迟兑换会失败。
4. 构建使用远程 HTTPS 重定向的 Web 客户端变体，并将其 DCR 元数据与原生客户端进行比较。
5. 在同一发行方下添加第二个资源。确认其访问令牌不能在第一资源上使用。

## 关键术语

| 术语 | 含义 |
|------|---------|
| 受保护资源元数据 | 标识资源和授权服务器的 RFC 9728 文档 |
| CIMD | URL 即 OAuth 客户端标识符的 HTTPS 元数据文档 |
| DCR | 为兼容而保留的已弃用动态客户端注册 |
| `application_type` | `native` 或 `web`，用于验证重定向 URI 规则 |
| PKCE | 保护被拦截的授权码的验证器和 S256 挑战 |
| `iss` | RFC 9207 授权响应发行方标识符 |
| 资源指示符 | 将令牌请求绑定到 MCP 资源的 RFC 8707 参数 |
| 受众 | 令牌有效的资源 |
| 权限升级 | 为当前操作的新增作用域进行新的同意和令牌签发 |
| 绑定到发行方的凭据 | 按精确授权服务器发行方隔离的注册和令牌记录 |

## 延伸阅读

- [MCP 2026-07-28 授权规范](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [RFC 9728：OAuth 2.0 受保护资源元数据](https://www.rfc-editor.org/rfc/rfc9728)
- [RFC 8707：OAuth 2.0 资源指示符](https://www.rfc-editor.org/rfc/rfc8707)
- [RFC 9207：OAuth 2.0 授权服务器发行方标识](https://www.rfc-editor.org/rfc/rfc9207)
- [OAuth 客户端 ID 元数据文档草案](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)
