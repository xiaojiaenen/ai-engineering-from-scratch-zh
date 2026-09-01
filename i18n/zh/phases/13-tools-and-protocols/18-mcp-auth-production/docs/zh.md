# 生产环境中的 MCP 认证：发行方绑定注册与令牌

> 第 16 课构建了 OAuth 2.1 状态机。本课针对 MCP 2026-07-28 加固其生产边界：优先使用 Client ID Metadata Documents，仅保留 DCR 作为兼容性回退、授权响应发行方验证、发行方密钥化的客户端凭据、JWKS 刷新，以及在每个无状态请求上固定受众的令牌。
>
> **规范说明（2026-07-28）：** 动态客户端注册（DCR）已被弃用，取而代之的是 Client ID Metadata Documents。DCR 仍作为兼容性机制保留。当使用 DCR 时，客户端必须声明正确的 `application_type`。客户端会验证出现的 RFC 9207 `iss` 值，且不会在不同授权服务器发行方之间重用凭据。

**类型：** 构建
**语言：** Python（标准库）
**前置知识：** 第 13 阶段 · 16（OAuth 2.1 状态机），第 13 阶段 · 17（网关）
**预计时间：** 约 90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并验证合约。
- 通过 Client ID Metadata Document 注册，并将已弃用的 DCR 隔离为回退路径。
- 验证 RFC 9207 `iss`、按授权服务器发行方注册的密钥，以及按发行方和资源绑定的密钥化资源令牌。
- 按计划缓存和刷新 JWKS 密钥，使签名验证能够 survive 密钥轮换。
- 使用 RFC 8707 资源指示器将令牌固定到单个 MCP 资源，拒绝混淆副手重放。
- 选择 JWT 验证或令牌内省，定义撤销时效性，并在身份依赖不可用时安全降级。
- 分离授权服务器、资源服务器和客户端，使每个角色仅执行自己的校验。
- 对照部署检查清单审计授权服务器，拒绝不安全的注册或令牌重用。

## 问题

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境存在三个仅靠内存模拟器无法察觉的操作缺口。

第一个缺口是注册和凭据隔离。一个真实组织可能运行数百个 MCP 服务器和数千个 MCP 客户端。2026-07-28 修订版优先采用 **Client ID Metadata Document**：客户端使用一个以它控制的路径作为标识符的 HTTPS URL，授权服务器从中拉取元数据。RFC 7591 动态注册仅作为已弃用的兼容路径保留。当不可避免地需要使用 DCR 时，请求必须声明正确的 `application_type`。客户端按授权服务器发行方存储注册信息，并按 `(发行方, 资源)` 对存储访问令牌。发行方变更意味着需要重新注册，资源不同意味着需要单独固定受众的令牌。

第二个缺口是密钥轮换。JWT 验证依赖于授权服务器的签名密钥，这些密钥以 JSON Web Key Set (JWKS) 形式发布。授权服务器按计划轮换这些密钥（通常每小时一次，有时在事件响应期间更快）。一个仅在启动时拉取一次 JWKS 的 MCP 服务器在轮换窗口之前验证正常，之后所有请求都会失败直到重启。生产环境将 JWKS 配置为带刷新任务的缓存值，在旧密钥过期前覆盖缓存，并在缓存未命中时回退拉取，以处理收到由比缓存更新密钥签名的令牌的情况。

第三个缺口是受众绑定。第 16 课引入了 RFC 8707 资源指示器。在生产环境中，该指示器成为每个请求的硬声明检查。MCP 服务器将 `token.aud` 与其自身的规范资源 URL 进行比较，对不匹配项返回 HTTP 401。这是防止上游 MCP 服务器（或持有指向某服务器令牌的恶意客户端）在同一信任 mesh 中对另一服务器重放该令牌的唯一防御。

本课将每个缺口映射到表面的具体部分。元数据文档是一个 HTTP 端点。JWKS 缓存刷新是一个调度任务加上一个键值缓存。JWT 验证是资源服务器在分发任何工具之前执行的例程。保持三个角色分离，每个角色仅执行自己负责的校验：授权服务器颁发和轮换密钥，资源服务器缓存和验证，客户端发现和注册。

## 范围：第 16 课之后的生产执行

[第 16 课：使用 OAuth 2.1 的 MCP 安全](../../16-mcp-security-oauth-2-1/docs/en.md) 负责授权码状态机、PKCE、受保护资源发现、资源指示器和作用域决策。本课不定义第二个 OAuth 流程。它在这些合约已存在的基础上，探讨部署的资源服务器如何在密钥轮换、不透明令牌验证、撤销、依赖故障、发布和事件响应期间继续执行这些合约。

生产边界更窄且更具操作性：

- JWT 路径在每个请求上验证固定发行方、算法、签名密钥、受众、时间声明和作用域，同时安全地刷新 JWKS。
- 不透明令牌路径调用发行方的认证内省端点，并验证返回的 active 状态、受众或资源、过期时间、主体和作用域。
- 撤销策略定义凭据必须停止工作的速度，以及哪个缓存可能延迟该事实。
- 故障策略决定在发现、JWKS、内省或撤销基础设施不可用时会发生什么。
- 证据记录驱动结果的发行方元数据、密钥集或内省响应、令牌声明、策略版本和拒绝原因，但不存储令牌。

这种区分使课程可组合。第 16 课证明流程可行。第 18 课证明令牌在到达真实 MCP 请求路径后仍然可信，或被拒绝。

## 概念

### RFC 8414 — OAuth 授权服务器元数据

位于 `/.well-known/oauth-authorization-server` 的文档描述了客户端所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "client_id_metadata_document_supported": true,
  "registration_endpoint": "https://auth.example.com/register",
  "authorization_response_iss_parameter_supported": true,
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

获得 MCP 资源 URL 的客户端链式发现：来自 RFC 9728 的 `oauth-protected-resource`（资源服务器的文档）指明发行方，然后 `oauth-authorization-server`（本 RFC）指明所有端点。客户端从不硬编码授权 URL。

对于带路径的资源标识符，在该路径之前插入 well-known 段。例如，`https://mcp.example.com/team/server` 在 `https://mcp.example.com/.well-known/oauth-protected-resource/team/server` 解析受保护资源元数据。在资源路径之后附加 `/.well-known/...` 是错误的。

在信任 IdP 用于 MCP 之前，你验证的合约：

- `code_challenge_methods_supported` 包含 `S256`（RFC 7636 的 PKCE）。规范明确指出：如果此字段**缺失**，授权服务器不支持 PKCE，客户端**必须**拒绝继续。
- `grant_types_supported` 包含 `authorization_code` 并拒绝 `password` 和 `implicit`。
- 至少有一条注册路径可用：`client_id_metadata_document_supported: true`（CIMD，首选）、预注册客户端或 `registration_endpoint`（已弃用的 RFC 7591 兼容）。
- 如果 `authorization_response_iss_parameter_supported` 为 true，客户端要求返回的 RFC 9207 `iss`，并在重定向前将其与记录的发行方精确比较。
- `response_types_supported` 对于 OAuth 2.1 必须是 `["code"]`。

如果缺少 `S256`，MCP 服务器拒绝针对此 IdP 部署——PKCE 没有降级模式。如果*既未*advertise 注册路径*又*没有预注册 `client_id`，你也无法注册；部署清单有误，而非代码。

### RFC 9728（回顾）— 受保护资源元数据

第 16 课涵盖了 RFC 9728。生产中的差异：此文档是客户端查找*此* MCP 服务器信任的授权服务器的唯一位置。单个 MCP 服务器可能接受来自多个 IdP 的令牌（一个用于员工，一个用于合作伙伴）。RFC 9728 声明该集合；RFC 8414 文档说明每个 IdP 支持什么。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents（推荐默认值）

CIMD 将注册从*推送*逆转为*拉取*。客户端不使用授权服务器生成 `client_id`，而是使用其控制的 HTTPS URL **作为** `client_id`。该 URL 解析为 JSON 元数据文档；授权服务器在 OAuth 流程中按需拉取。信任根植于 DNS：如果服务器操作员信任 `app.example.com`，则信任从 `https://app.example.com/client.json` 提供的客户端。无需注册往返，无需耗尽 `client_id` 命名空间，无需同步每台服务器的状态。

客户端托管的元数据文档：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文档中的 `client_id` 值**必须**等于其服务的 URL（授权服务器会验证此点；不匹配会被拒绝）。授权服务器在其 RFC 8414 元数据中使用 `client_id_metadata_document_supported: true` 广告支持。

对于当前 CIMD 合约，`client_id`、`client_name` 和非空 `redirect_uris` 数组是必需的。客户端标识符是带路径的绝对 HTTPS URL。`application_type` 可以包含，但不是强制 CIMD 字段。不要将 DCR 对 `application_type` 的要求复制到首选 CIMD 路径。

规范明确指出的两个安全事实：

- **SSRF。** 授权服务器拉取攻击者提供的 URL。必须防御服务器端请求伪造（禁止拉取内部/管理员端点）。
- **localhost 冒充。** CIMD 本身无法阻止本地攻击者声称合法客户端的元数据 URL 并绑定任何 `localhost` 重定向。授权服务器**必须**在同意期间清晰显示重定向 URI 主机名，并**应**对仅 `localhost` 的重定向发出警告。

由于 CIMD 不需要服务端状态，因此没有像 DCR 那样需要建立的注册方。客户端侧是只读的：从静态 HTTPS 端点提供你的元数据文档，让授权服务器拉取。

如果授权服务器操作员已预置了客户端标识符，在尝试自动注册前先使用该发行方范围的注册。否则优先使用 CIMD。仅在发行方无法使用预注册或 CIMD 时使用已弃用的 DCR。

### RFC 7591：已弃用的兼容注册

DCR 在 2026-07-28 修订版中已被弃用。仅在无法消费 CIMD 且预注册不切实际的授权服务器上保留它。兼容客户端发布：

```json
POST /register
Content-Type: application/json

{
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器返回 `client_id` 和 `registration_access_token` 用于后续更新：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`application_type` 不是装饰性的。回环桌面客户端声明 `native`；服务器托管客户端声明 `web` 并使用 HTTPS 重定向 URI。`token_endpoint_auth_method: none` 是公共 native 客户端的正确默认值。它仅获得 `client_id`，PKCE 提供持有证明。

三个生产陷阱：

- 注册端点必须按源 IP 速率限制。否则，恶意行为者会脚本化数百万个虚假注册并耗尽 `client_id` 命名空间。在注册方处理请求前运行速率限制检查。
- `software_statement`（为客户端担保的签名 JWT）被某些企业 IdP 要求。本课程的 mock 跳过了它；生产环境会连接一个验证步骤，拒绝来自非 localhost 重定向 URI 的任何无签名注册。
- `registration_access_token` 必须以哈希形式存储，而非明文。此令牌的泄露意味着攻击者可以重写客户端的重定向 URI。

### RFC 8707（回顾）— 资源指示器

第 16 课确立了形状。生产规则：每个令牌请求都包含 `resource=<canonical-mcp-url>`，MCP 服务器在每个调用上验证 `token.aud` 是否与其自身资源 URL 匹配。规范 URI 是服务器的*最具体*标识符：使用小写方案主机，不含 fragment，且惯例上不含尾部斜杠。路径组件**不得**按规则剥离——规范在需要标识单个 MCP 服务器时保留它。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是有效的规范 URI。为每个服务器选择一个并将 `aud` 固定为恰好该值。（本课的 mock 为简洁起见使用裸主机受众如 `https://notes.example.com`；在同一源下托管多个 MCP 服务器的部署通过路径区分它们。）

### RFC 7636（回顾）— PKCE

PKCE 在 OAuth 2.1 中是强制性的。课程的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器拒绝任何没有验证器或验证器哈希与存储挑战不匹配的令牌请求。

### MCP 2026-07-28 授权配置

当前 MCP 修订版保持 OAuth 资源服务器边界，同时使 MCP 传输无状态。没有协议会话可以缓存身份决策。因此授权层独立验证每个请求：

- 实现 RFC 9728 受保护资源元数据，并通过 401 上的 `WWW-Authenticate: Bearer resource_metadata="..."` 头**或** well-known URI `/.well-known/oauth-protected-resource` 提供其位置（SEP-985 使头可选，带有 well-known 回退）。元数据 `authorization_servers` 字段**必须**至少命名一个服务器。
- 仅在**每个**请求上通过 `Authorization: Bearer ...` 接受令牌——绝不在查询字符串中，绝不仅在会话开始时验证。
- 按请求验证 `aud`、`iss`、`exp` 和必需作用域。服务器**必须**验证令牌是专门为其颁发的（受众）；缺失或不匹配的 `aud` 会被拒绝，从不视为通配符。
- 在 401/403 上，返回携带 `error=...` 的 `WWW-Authenticate: Bearer`，`resource_metadata="<PRM-URL>"` 参数（元数据文档的 URL，*非*裸资源），以及在 `insufficient_scope`（403）时的 `scope="..."`。注意：参数是 `resource_metadata`，一个发现指针——挑战中没有 `resource` 参数。
- 授权服务器发现接受**任意** RFC 8414 OAuth 元数据**或** OpenID Connect Discovery 1.0；客户端必须按优先级顺序尝试两种 well-known 后缀。
- 客户端（而非服务器）防御**混合攻击**：它在重定向前记录预期的 `issuer`，并在兑换代码前验证实际授权响应中返回的 `iss` 值（RFC 9207）。仅靠 PKCE 无法阻止混合攻击，因为客户端将其 `code_verifier` 交给它被引导到的任何令牌端点。
- 客户端凭据属于一个授权服务器发行方。如果发现解析到不同发行方，客户端重新注册，而非呈现旧的 `client_id`、注册令牌或访问令牌。
- CIMD 是首选注册机制。DCR 已被弃用；兼容 DCR 请求仍声明正确的 `application_type`。

OAuth 2.1 草案是基底；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 是表面；MCP 规范是配置。

### 部署能力检查清单

供应商功能表很快就会过时。检查你将实际部署的授权服务器返回的元数据。闸门是机械的：

| 检查项 | 必需决策 |
|---|---|
| 发现的发行方 | 策略预期的精确 HTTPS 发行方 |
| PKCE | 广告 `S256`；否则停止 |
| 注册 | 首选 CIMD，接受预注册，DCR 仅作为已弃用的兼容 |
| 授权响应 | 出现或广告时验证 RFC 9207 `iss` |
| 资源绑定 | 令牌请求携带 `resource`；资源服务器要求匹配的 `aud` |
| 凭据存储 | 按发行方键控客户端 ID 和注册凭据；按发行方加资源键控访问令牌 |
| DCR 兼容 | 声明 `native` 或 `web`；拒绝与声明的应用类型不匹配的重定向 URI |

不要从产品名称或定价层级推断支持。在部署证据中捕获发现的文档，并在缺少强制字段时关闭。

### JWKS 刷新模式（在 AS 旋转，在资源服务器刷新）

保持两个动词分离，因为混淆它们是真实的生产 bug：

- **旋转**是*授权服务器*所做的：生成新签名密钥，在 JWKS 中发布，稍后退役旧密钥。资源服务器不参与此操作也无法执行——它不持有 IdP 的私钥。
- **刷新**是*资源服务器*所做的：将发布的 JWKS 重新 `GET` 到其缓存中。这是资源服务器永远执行的唯一 JWKS 操作。

生产故障模式是陈旧缓存。通过调度刷新任务加上键值缓存解决它。资源服务器运行一个任务（cron、计时器、你运行时提供的任何机制），在固定间隔内拉取 `<issuer>/.well-known/jwks.json` 并覆盖 `cache[issuer] = {keys, fetched_at}`。验证器从该缓存读取。缓存中缺失 `kid` 的令牌触发**一次**同步刷新作为回退，然后重新检查。这同时处理两种情况：调度刷新，以及令牌由全新密钥签名但在下次调度刷新前到达的密钥重叠窗口。

回退**必须是重新拉取，绝不旋转**。如果你将缓存未命中路径连接到旋转并生成，两件事会出错：(1) 生成新鲜密钥会产生一个*仍然*不匹配令牌的 `kid`，所以查找无论如何都会失败；(2) 向随机 `kid` 值喷洒令牌的攻击者会迫使无限制的关键字创建系列——自 DoS。重新拉取是可幂等的，因此伪造的 `kid` 最多花费一次浪费的拉取。

缓存形状：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时存在两个密钥是稳态。授权服务器通过在新密钥（`k_2026_04`）退役旧密钥（`k_2026_03`）之前引入下一个密钥来旋转，因此在过期之前旧密钥签发的令牌保持有效。缓存保存并集；验证器按 `kid` 选择。

### 验证例程

MCP 服务器在分发任何工具之前运行验证。`code/main.py` 使用：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解码 JWT，从 JWKS 缓存解析签名密钥（在未命中时刷新一次），验证签名，然后按允许列表检查 `iss`，按此服务器的规范资源检查 `aud`，检查 `exp` 和必需作用域——在第一次失败时返回 `WWW-Authenticate` 挑战。将其保持在资源服务器上的单个例程意味着每个入口点（每个工具调用、每个传输）都通过相同的检查；没有不先验证就能到达工具的路径。

### 不透明令牌使用内省而非猜测

并非每个访问令牌都是 JWT。如果发行方文档化了不透明令牌，资源服务器无法将其解码为可信声明。它通过认证的后通道将令牌发送到发行方的 RFC 7662 内省端点，并要求 `active: true`、预期的发行方上下文、精确的 MCP 受众或资源、未过期的时间声明，以及具体工具所需的作用域。

按发行方、单向令牌摘要和 MCP 资源缓存内省。绝不使用明文令牌作为日志或缓存标签。以令牌过期、发行方缓存指导和部署撤销时效目标的最小值约束正缓存条目。保持负缓存足够短，使新发行的令牌不会保持虚假的非活跃状态。一个资源的内省结果不能授权另一个资源，即使不透明令牌字符串相同。

不要从攻击者控制的令牌内容中选择验证模式。将 JWT 与内省行为固定到已验证的发行方元数据和部署配置。在 JWT 路径上，固定接受的算法和可信 `jwks_uri`；绝不跟随仅由令牌头选择的密钥 URL 或算法。

### 撤销是时效性合约

RFC 7009 允许客户端请求授权服务器撤销令牌。该请求不会擦除每个资源服务器已缓存的副本。定义最大可接受的撤销延迟，并使每个缓存遵守它。

不透明令牌部署可以通过在每个高风险调用上内省或使用短正缓存实现更紧密的撤销。自包含 JWT 部署通常结合短访问令牌生命周期与刷新令牌撤销、发行方级事件的密钥退役，以及可选的主体、会话或令牌 ID 拒绝列表以实现紧急本地拒绝。已签名 JWT 在密码学上保持有效直到过期，除非资源服务器有当前的外部撤销证据。

注销、账户禁用、同意撤回和事件响应是不同的触发器，但必须收敛到一个可衡量的声明：在最多多于声明的撤销窗口后，每个副本都拒绝该凭据。通过负载均衡器测试该声明，而不仅针对一个热进程。

### 依赖故障需要声明决策

绝不在异常处理程序内即兴发挥可用性策略。

| 故障 | 安全生产行为 |
|---|---|
| 调度 JWKS 刷新失败，已知 `kid` 仍在有效边界缓存中 | 仅在声明的错误时陈旧窗口内继续，并发出降级健康证据 |
| 令牌具有未知 `kid` 且允许的一次刷新失败 | 拒绝；绝不接受不可验证的签名 |
| 内省不可用 | 对受保护调用关闭；不要将网络故障转换为 `active: true` |
| 受保护资源或发行方元数据意外更改 | 停止新注册和令牌获取；仅在边界事件策略下保留明确固定的未过期配置 |
| 撤销端点不可用 | 报告注销或撤销不完整，尽可能在本地保留凭据为不可用，不声称全局撤销成功 |
| 时钟源或声明类型无效 | 拒绝而不是放宽偏差直到令牌通过 |

将故障与无效凭据分开分类。依赖中断是带有健康和重试策略的操作错误。签名、发行方、受众、过期时间或作用域错误是授权拒绝。两者都不到达工具处理器，都不应将令牌内容泄漏到审计证据中。

### 受众重放演练（访问令牌权限限制）

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）都针对同一个授权服务器注册。服务器 A 被攻陷。攻击者获取用户的笔记令牌并重放到服务器 B。

服务器 B 的验证器：

1. 解码 JWT，按 `kid` 拉取 JWKS，验证签名。
2. 按受保护资源元数据的 `authorization_servers` 检查 `iss`。（通过——同一 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——令牌的 `aud` 是 `https://notes.example.com`。）
4. 返回 401 带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

受众声明是对抗此攻击的协议层唯一防御。为性能跳过它是最常见的生产错误；验证器必须在每个请求上运行，而不仅是在会话开始时。规范称此为**访问令牌权限限制**：MCP 服务器**必须**拒绝任何未在受众中命名它的令牌。

> **命名说明。** 规范保留术语 *混淆副手* 用于相关但不同的问题：作为 OAuth **代理**操作到第三方 API 的 MCP 服务器，使用静态客户端 ID，在未取得 per-client 用户同意情况下转发令牌。受众绑定修复上述重放；混淆副手修复是 per-client 同意**加**永不将入站令牌传递到上游 API（MCP 服务器**必须**获取自己的单独上游令牌）。

### 混合攻击（服务器无法提供的客户端防御）

客户端在其生命周期中与他众多授权服务器通信。恶意 AS 可以尝试使客户端在攻击者的令牌端点兑换诚实 AS 的授权码。受众绑定在此无帮助——攻击发生在任何令牌存在之前。防御存在于客户端（RFC 9207）：

1. 在重定向前，客户端从已验证的 AS 元数据记录预期的 `issuer`。
2. 在授权响应上，客户端在将代码发送到任何地方之前，将返回的 `iss` 参数与记录的发行方进行比较（简单字符串比较，无规范化）。
3. 不匹配（或当 AS 广告 `authorization_response_iss_parameter_supported` 时 `iss` 缺失）→ 拒绝，甚至不显示 `error` 字段。

仅靠 PKCE 无法阻止混合攻击，因为客户端将其 `code_verifier` 交给它被引导到的任何令牌端点。这就是规范按请求记录发行方 alongside PKCE 验证器和 `state` 的原因。

### 故障模式

- **陈旧 JWKS。** 验证器在 AS 旋转密钥后拒绝有效令牌。修复是上面的 cron 刷新 + 缓存未命中重新拉取模式。绝不在无刷新任务的情况下缓存 JWKS。
- **旋转作为回退。** 将缓存未命中路径连接到旋转和生成而非重新拉取是一个真实 bug：它永远不会产生缺失的 `kid`，并将攻击者控制的 `kid` 值转化为密钥创建 DoS。回退必须是可幂等的 `refresh-jwks`。
- **缺失 `aud` 声明。** 某些 IdP 默认在令牌请求中缺少 `resource` 时省略 `aud`。验证器必须拒绝缺失 `aud` 的令牌，而非将缺失视为通配符。
- **通过缺失 `iss` 检查的混合。** 不验证 RFC 9207 `iss` 授权响应参数与重定向前记录的发行方的客户端，可能被引导到在攻击者的令牌端点兑换诚实 AS 的代码。这是客户端故障；资源服务器无法补偿。
- **作用域升级竞争。** 同一用户的两个并发逐步升级流程可能都成功并产生两个具有不同作用域的访问令牌。验证器必须使用请求上呈现的令牌，而非查找"用户当前作用域"——这创造 TOCTOU 窗口。
- **注册令牌 theft。** 泄露的 `registration_access_token` 使攻击者能够重写重定向 URI。静默时哈希这些；每次更新时要求客户端呈现明文；在怀疑时轮换。
- **未固定 `iss`。** 接受任何 `iss` 的验证器使攻击者能够建立自己的授权服务器，为目标受众注册客户端，并颁发令牌。受保护资源元数据的 `authorization_servers` 列表是允许列表；强制执行它。
- **凭据或令牌缓存碰撞。** 仅按资源键控注册的客户端可能向另一个授权服务器呈现一个授权服务器的身份。仅按发行方键控访问令牌的客户端可能在错误受众重放令牌。按已验证发行方键控注册，按 `(发行方, 资源)` 键控访问令牌，每当发行方变更时重新注册。

```figure
t3-jwks-rotate
```

## 使用

`code/main.py` 使用 stdlib Python 和三个角色：`AuthorizationServer`、`ResourceServer` 和 `Client` 演示完整生产流程。流程：

从仓库根目录运行：

```bash
cd phases/13-tools-and-protocols/18-mcp-auth-production
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

第一个命令打印发行方绑定注册和令牌验证转录。第二个报告十八个通过检查。两个命令都不打开网络监听器或写入凭据。

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点并检查其注册选项（CIMD 的 `client_id_metadata_document_supported`，DCR 的 `registration_endpoint`）和 `S256` PKCE 支持。
3. 客户端检查发行方范围预注册，否则使用其 HTTPS Client ID Metadata Document 注册。已弃用的 DCR 保持为单独可测试的兼容方法。
4. 客户端记录已验证发行方，创建 S256 挑战，接收一次性授权码加 `iss`，验证返回的发行方，并与原始验证器和 RFC 8707 `resource` 指示器兑换代码。
5. MCP 客户端使用 `Authorization: Bearer ...` 在 MCP 服务器上调用工具。
6. MCP 服务器运行 `validate`，从 JWKS 缓存解析签名密钥。
7. IdP 旋转密钥；调度刷新将 JWKS 重新拉取到缓存中。
8. 下次调用使用刷新后的密钥验证而无需重启，之前的令牌在重叠窗口期间仍验证。
9. 针对不同 MCP 资源的受众重放尝试获得 401 带 `audience mismatch` 和 `resource_metadata` 指针。

此处的 JWT 使用 HS256 与共享密钥（以便课程仅在 stdlib 上运行）。生产环境使用 RS256 或 EdDSA 加上上述 JWKS 模式；验证逻辑其余相同。因为 IdP 和资源服务器存在于一个进程中，`refresh_jwks` 直接读取授权服务器的密钥列表；在网络上它是到 `jwks_uri` 的 HTTP `GET`。

## 交付

本课产生 `outputs/skill-mcp-auth.md`。给定 MCP 服务器配置和 IdP 能力集，技能发出要搭建的认证表面——受保护资源元数据、要使用的注册路径（CIMD、预注册或 DCR 回退）、JWKS 刷新计划、作用域映射，以及在 IdP 不支持完整 RFC 配置时应用的拒绝规则。

## 练习

1. 运行 `code/main.py`。追踪流程。注意 IdP 在步骤 6 如何旋转密钥，调度 `refresh_jwks` 如何重新拉取发布的集合，以及旧令牌（重叠窗口）和新令牌如何在不重启的情况下验证。

2. 向受保护资源元数据的 `authorization_servers` 列表添加新 IdP。颁发由新 IdP 签发的令牌并确认验证器接受它。颁发由未列出 IdP 签发的令牌并确认验证器拒绝带 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`。

3. 向 `register_client` 添加在注册方接受请求之前运行的速率限制检查。使用按源 IP 的令牌桶，保存在按 IP 键控的小字典中。

4. 阅读 RFC 7591 并识别课程 `/register` 处理器未验证的两个字段。添加验证。（提示：`software_statement` 和 `redirect_uris` URI 方案。）

5. 添加第二个授权服务器。确认客户端存储单独的发行方键控注册，并拒绝重用第一个发行方的令牌或 `client_id`。

6. 证明 DoS 修复。向验证器发送具有随机 `kid` 的令牌并确认 `refresh_jwks` 最多运行一次且授权服务器的密钥计数不增长。然后故意将回退重新连接到旋转和生成，观察每个伪造令牌密钥计数攀升——之后恢复重新拉取。

7. 使用 `native` 和 `web` 客户端练习已弃用的 DCR。确认 HTTP 重定向 URI 的 web 客户端和没有精确回环重定向的 native 客户端被拒绝。

## 关键术语

| 术语 | 人们说什么 | 实际含义 |
|------|------------------------|------------------------|
| ASM | "OAuth 元数据文档" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "客户端元数据 URL" | Client ID Metadata Document：用作 `client_id` 的 HTTPS URL；AS 拉取 JSON。MCP 2026-07-28 的首选注册 |
| DCR | "自助客户端注册" | RFC 7591 `POST /register`；当前 MCP 已弃用，仅保留用于兼容 |
| JWKS | "JWT 验证的公钥" | JSON Web Key Set，从 `jwks_uri` 拉取，按 `kid` 索引 |
| 旋转与刷新 | "更新密钥" | *旋转* = AS 生成/退役签名密钥；*刷新* = 资源服务器重新拉取发布的集合。资源服务器永远只刷新 |
| 资源指示器 | "受众参数" | RFC 8707 `resource` 参数将令牌固定到一个服务器 |
| `aud` 声明 | "受众" | 验证器与规范资源 URL 比较的 JWT 声明 |
| 受众重放 | "令牌重放" | 为服务器 A 颁发的令牌呈现给服务器 B；通过受众验证防御（规范：访问令牌权限限制） |
| 混淆副手 | "代理令牌滥用" | 具有静态客户端 ID 转发票据的 MCP 代理而未取得 per-client 同意；与受众重放不同 |
| 混合攻击 | "错误的令牌端点" | 客户端被引导到在攻击者端点兑换诚实 AS 的代码；通过 RFC 9207 `iss` 在客户端防御 |
| `iss` 允许列表 | "受信任的授权服务器" | 在受保护资源元数据的 `authorization_servers` 中命名的集合 |
| `resource_metadata` | "在哪里找到 PRM 文档" | 401/403 上命名 RFC 9728 元数据 URL 的 `WWW-Authenticate` 参数 |
| 公共客户端 | "Native 或浏览器客户端" | 无 `client_secret` 的 OAuth 客户端；PKCE 补偿 |
| `WWW-Authenticate` | "401/403 响应头" | 携带驱动客户端恢复的 `Bearer error=...` 指令 |

## 延伸阅读

- [MCP 授权规范（2026-07-28）](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) - 当前 MCP 授权配置
- [MCP 2026-07-28 变更日志](https://modelcontextprotocol.io/specification/2026-07-28/changelog) - CIMD、发行方验证、DCR 弃用和发行方键控凭据变更
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — 发现合约
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（回退路径）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端持有证明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — 防御混合攻击的 `iss` 参数
- [RFC 7662: OAuth 2.0 Token Introspection](https://datatracker.ietf.org/doc/html/rfc7662)
- [RFC 7009: OAuth 2.0 Token Revocation](https://datatracker.ietf.org/doc/html/rfc7009)
