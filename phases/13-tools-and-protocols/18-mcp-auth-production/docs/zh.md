# 生产环境中的MCP认证 —— DCR、JWKS轮换与基于iii原语的受众绑定令牌

> 第16课在内存中搭建了OAuth 2.1状态机。到2026年，每个交付给真实组织的MCP服务器都需置于生产级认证之后：动态客户端注册（RFC 7591）、授权服务器元数据发现（RFC 8414）、能在凌晨三点不影响令牌验证的JWKS轮换，以及拒绝混淆代理重用的受众绑定令牌。本课将通过iii原语——用于HTTP和定时任务的`iii.registerTrigger`、用于认证逻辑的`iii.registerFunction`、用于缓存密钥的`state::set/get`——将这一切串联起来，使认证层面具备可观察性、可重启性，并像引擎中其他所有工作负载一样可重放。

**类型：** 构建
**语言：** Python（标准库，课程环境中iii原语已模拟）
**前置课程：** 第13阶段·16（OAuth 2.1状态机），第13阶段·17（网关）
**时间：** 约90分钟

## 学习目标

- 通过RFC 8414元数据发现授权服务器并验证其契约。
- 实现RFC 7591动态客户端注册，使MCP客户端无需管理员干预即可注册。
- 使用定时触发器缓存并轮换JWKS密钥，使签名验证能在密钥滚动更新时正常进行。
- 使用RFC 8707资源指示符将令牌绑定到单一MCP资源，并拒绝混淆代理重用。
- 将每个端点和后台任务都连接为iii原语——HTTP触发器、定时触发器、命名函数和`state::*`读取——使单次重启即可重建认证层面。
- 阅读身份提供者（IdP）能力矩阵，并在IdP无法满足MCP认证配置文件时拒绝部署。

## 问题所在

第16课的模拟器在内存中运行OAuth 2.1。生产环境存在三个内存模拟器无法看到的运营缺口。

第一个缺口是注册环节。真实组织运行着数百个MCP服务器和数千个MCP客户端。运维人员不会为每个Cursor用户手动注册为OAuth客户端。RFC 7591动态客户端注册允许客户端`POST /register`向授权服务器注册，并当场获得`client_id`（可选`client_secret`）。服务器在其RFC 8414元数据中发布`registration_endpoint`；客户端无需带外配置即可发现它。

第二个缺口是密钥轮换。JWT验证依赖授权服务器的签名密钥，这些密钥以JSON Web Key Set（JWKS）形式发布。授权服务器按计划轮换这些密钥（通常每小时一次，事件响应时可能更快）。一个在启动时仅获取一次JWKS的MCP服务器在轮换窗口前验证正常——随后所有请求都将失败，直到重启。生产环境将JWKS作为带刷新任务的缓存值接入，在旧密钥过期前覆盖缓存，并为“收到比缓存更新密钥签名的令牌”这种情况设置缓存未命中时的回退获取。

第三个缺口是受众绑定。第16课引入了RFC 8707资源指示符。在生产环境中，该指示符成为每个请求的强制声明检查。MCP服务器将`token.aud`与其自身规范资源URL进行比较，并在不匹配时返回HTTP 401拒绝。这是防御上游MCP服务器（或持有为某个服务器签发的令牌的恶意客户端）在同一信任网格中向另一个服务器重放该令牌的唯一手段。

本课将每个缺口都视为一个iii原语。元数据文档是一个HTTP触发器，返回函数的输出。JWKS轮换是一个定时触发器，调用`auth::rotate-jwks`，后者写入`state::set("auth/jwks/<issuer>", ...)`。JWT验证是一个其他函数通过`iii.trigger("auth::validate-jwt", token)`调用的函数。MCP服务器本身只是另一个HTTP触发器，在分发前调用验证逻辑。重启引擎：触发器注册表重建；状态得以保留；认证层面无需手动调和即可运行。

## 核心概念

### RFC 8414 — OAuth授权服务器元数据

位于`/.well-known/oauth-authorization-server`的文档描述了客户端所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

给定MCP资源URL的客户端进行链式发现：RFC 9728（资源服务器的文档）中的`oauth-protected-resource`指明签发者，然后`oauth-authorization-server`（本RFC）列出所有端点。客户端无需硬编码授权URL。

在信任IdP用于MCP之前需验证的契约：

- `code_challenge_methods_supported` 包含 `S256`（基于RFC 7636的PKCE）。
- `grant_types_supported` 包含 `authorization_code` 并拒绝 `password` 和 `implicit`。
- `registration_endpoint` 存在（支持RFC 7591）。
- `response_types_supported` 恰好是 `["code"]`，用于OAuth 2.1。

如果缺少其中任何一项，MCP服务器将拒绝部署到该IdP。部署清单有误，而非代码问题。

### RFC 9728（回顾）— 受保护资源元数据

第16课涵盖了RFC 9728。生产环境中的变化：这是客户端查找*此*MCP服务器信任的授权服务器的唯一地方。单个MCP服务器可能接受来自多个IdP的令牌（一个用于员工，一个用于合作伙伴）。RFC 9728声明该集合；RFC 8414记录每个IdP支持的内容。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591 — 动态客户端注册

没有DCR，每个MCP客户端（Cursor、Claude桌面版、自定义代理）都需要与IdP管理员进行带外交换。有了DCR，客户端只需发送：

```json
POST /register
Content-Type: application/json

{
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

服务器响应`client_id`和用于后续更新的`registration_access_token`：

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

`token_endpoint_auth_method: none`是运行在用户设备上的MCP客户端的合适默认设置。它们仅获得`client_id`——没有可泄露的`client_secret`。PKCE为公共客户端提供了所需的持有证明。

三个生产环境陷阱：

- 注册端点必须按源IP进行速率限制。否则，恶意行为者可脚本化数百万虚假注册并耗尽`client_id`命名空间。iii使其变得简单：注册HTTP触发器在分发到注册器之前调用`auth::rate-limit`函数。
- 一些企业IdP要求`software_statement`（为客户端担保的签名JWT）。本课程的模拟跳过了它；生产环境会接入一个验证步骤，拒绝来自除本地回环重定向URI之外任何来源的未签名注册。
- `registration_access_token`必须以哈希形式存储，而非明文。此令牌被盗意味着攻击者可以重写客户端的重定向URI。

### RFC 8707（回顾）— 资源指示符

第16课确立了其形式。生产环境规则：每个令牌请求都包含`resource=<canonical-mcp-url>`，MCP服务器在每次调用时验证`token.aud`是否匹配其自身资源URL。如果MCP服务器可通过`https://notes.example.com/mcp`访问，则规范URL是`https://notes.example.com`——排除路径部分，以便单个服务器在同一个受众下托管多个路径。

### RFC 7636（回顾）— PKCE

在OAuth 2.1中，PKCE是强制性的。本课程的授权码流程始终携带`code_challenge`和`code_verifier`。服务器拒绝任何没有验证器或验证器哈希值与存储的质询不匹配的令牌请求。

### MCP规范 2025-11-25 认证配置文件

MCP规范（2025-11-25）对MCP服务器授权层的要求非常明确：

- 发布`/.well-known/oauth-protected-resource`（RFC 9728）。
- 仅通过`Authorization: Bearer ...`接受令牌。
- 验证每个请求的`aud`、`iss`、`exp`以及所需作用域。
- 对于所有401和403响应，使用携带`Bearer error=...`的`WWW-Authenticate`进行响应，并在适用时包含`scope=`和`resource=`参数。
- 拒绝`aud`与规范资源不匹配的令牌。
- 拒绝`iss`不在受保护资源元数据的`authorization_servers`列表中的令牌。

OAuth 2.1草案是基础；RFC 8414/7591/8707/9728 + RFC 7636是表面；MCP规范是配置文件。

### IdP能力矩阵

并非每个IdP都支持完整的MCP配置文件。下表根据2025-11-25规范记录了事实能力声明。它是一道*部署门禁*，而非推荐。

| IdP类别 | RFC 8414元数据 | RFC 7591 DCR | RFC 8707资源 | RFC 7636 S256 PKCE | 说明 |
|---|---|---|---|---|---|
| 自托管 (Keycloak) | 是 | 是 | 是 (自24.x起) | 是 | 本课程MCP配置文件的参考IdP；端到端支持所有RFC。 |
| 企业SSO (Microsoft Entra ID) | 是 | 是 (高级层级) | 是 | 是 | DCR可用性因租户层级而异；部署前需在目标租户中验证。 |
| 企业SSO (Okta) | 是 | 是 (Okta CIC / Auth0) | 是 | 是 | DCR在Auth0（现Okta CIC）上可用；经典Okta组织需要管理员预注册。 |
| 社交登录IdP (通用) | 不同 | 罕见 | 罕见 | 是 | 多数社交IdP将客户端视为静态合作伙伴；不要依赖DCR。仅用作身份源，在其上层搭建自己的MCP感知授权服务器。 |
| 自定义/自研 | 视情况而定 | 视情况而定 | 视情况而定 | 视情况而定 | 如果自研，请提供完整配置文件。跳过上述四个RFC中的任何一个都会破坏MCP认证契约。 |

部署清单的拒绝规则：如果所选IdP不返回`registration_endpoint`且未在`code_challenge_methods_supported`中列出`S256`，则MCP服务器拒绝启动。没有降级模式。

### 基于iii的JWKS轮换模式

生产环境的失败模式是过期的JWKS缓存。使用定时触发器和`state::*`缓存来解决：

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

每六小时，定时触发器调用`auth::rotate-jwks`，后者获取`<issuer>/.well-known/jwks.json`并写入`state::set("auth/jwks/<issuer>", {keys, fetched_at})`。验证器从`state::get`读取。缓存中缺少`kid`的令牌会触发同步的`auth::rotate-jwks`调用作为回退。这同时处理了两种情况：计划轮换（定时任务）和密钥重叠窗口（同步回退）。

状态形状：

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时持有两个密钥是稳态。授权服务器通过在退役旧密钥（`k_2026_03`）之前引入新密钥（`k_2026_04`）来进行轮换，因此旧密钥签发的令牌在过期前仍然有效。缓存包含两者并集；验证器通过`kid`选择。

### iii原语连接（本课的实际重点）

五个原语组合成认证表面：

```python
# 1. RFC 8414 metadata document
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 dynamic client registration
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT validation as a callable function (the resource server triggers it)
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. Step-up issuance for incremental scope (SEP-835 from L16)
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cron-driven JWKS rotation
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

MCP服务器本身从不直接调用验证。它执行：

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

这种间接调用是iii的赌注。明天你可以将验证器换成并行查询两个IdP的扇出器，或者添加span发射器，或者缓存正验证结果。MCP服务器无需更改。

### 受众绑定混淆代理解析

服务器A（`notes.example.com`）和服务器B（`tasks.example.com`）都在同一授权服务器上注册。服务器A被攻陷。攻击者获取用户的笔记令牌并将其重放到服务器B。

服务器B的验证器：

1. 解码JWT，通过`kid`获取JWKS，验证签名。
2. 将`iss`与其受保护资源元数据的`authorization_servers`进行比较。（通过——同一IdP。）
3. 检查`aud == "https://tasks.example.com"`。（失败——令牌的`aud`是`https://notes.example.com`。）
4. 返回401及`WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`。

受众声明是协议层防御此攻击的唯一手段。为了性能跳过它是最常见的生产环境错误；验证器必须在每个请求上运行，而非仅在会话开始时。

### 失败模式

- **JWKS过期。** 密钥轮换后，验证器拒绝有效令牌。修复方法是上述的定时任务+回退模式。切勿在没有刷新任务的情况下缓存JWKS。
- **缺少`aud`声明。** 一些IdP默认省略`aud`，除非令牌请求中存在`resource`。验证器必须拒绝缺少`aud`的令牌，不能将缺失视为通配符。
- **作用域升级竞态。** 同一用户的两个并发升级流程可能都成功，并生成具有不同作用域的两个访问令牌。验证器必须使用请求中呈现的令牌，而非查找“用户的当前作用域”——那会造成TOCTOU窗口。
- **注册令牌泄露。** 泄露的`registration_access_token`允许攻击者重写重定向URI。在静态存储中将其哈希；要求客户端在每次更新时呈现明文；在有疑虑时进行轮换。
- **`iss`未绑定。** 接受任何`iss`的验证器允许攻击者搭建自己的授权服务器，为目标受众注册客户端，并签发令牌。受保护资源元数据的`authorization_servers`列表是允许列表；必须强制执行。

## 动手实践

`code/main.py`使用标准库Python和一个小型`iii_mock`注册表（模拟`iii.registerFunction`、`iii.registerTrigger`、`iii.trigger`和`state::set/get`）演示完整的生产环境流程。流程如下：

1. 授权服务器在`/.well-known/oauth-authorization-server`发布RFC 8414元数据。
2. MCP客户端调用元数据端点，发现注册端点。
3. MCP客户端向`/register`（RFC 7591）发送请求并接收`client_id`。
4. MCP客户端运行带有`resource`指示符（RFC 8707）的PKCE保护授权码流程（RFC 7636）。
5. MCP客户端使用`Authorization: Bearer ...`调用MCP服务器上的工具。
6. MCP服务器触发`auth::validate-jwt`，后者从`state::get`读取JWKS。
7. 定时触发器触发`auth::rotate-jwks`，替换状态中的JWKS。
8. 下一次调用无需重启即可使用新密钥进行验证。
9. 针对不同MCP资源的混淆代理尝试因受众不匹配而收到401响应。

此处的模拟JWT使用带共享密钥的HS256（因此课程仅依赖标准库）。生产环境使用RS256或EdDSA，并采用上述JWKS模式；验证逻辑在其他方面是相同的。

## 交付成果

本课程产出`outputs/skill-mcp-auth-iii.md`。给定MCP服务器配置和IdP能力集，该技能输出需要注册的iii原语、JWKS轮换计划、作用域映射以及IdP不支持完整RFC配置文件时应用的拒绝规则。

## 练习

1. 运行`code/main.py`。跟踪9步流程。注意`state::get`在`auth::rotate-jwks`覆盖它之前立即返回过时数据的位置，以及下一次请求现在如何使用新密钥进行验证。

2. 向受保护资源元数据的`authorization_servers`列表添加新的IdP。签发由新IdP签名的令牌并确认验证器接受它。签发由未列出IdP签名的令牌并确认验证器因`WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`而拒绝。

3. 将`auth::rate-limit`实现为iii函数，并在注册器运行之前从注册HTTP触发器内部调用它。使用存储在`state::set("auth/ratelimit/<ip>", ...)`中的每源IP令牌桶。

4. 阅读RFC 7591并找出本课程`/register`处理程序未验证的两个字段。添加验证。（提示：`software_statement`和`redirect_uris` URI方案。）

5. 阅读MCP规范2025-11-25授权部分。找出本课程验证器目前未发出的关于`WWW-Authenticate`头部的唯一规范性要求。将其添加。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| ASM | “OAuth元数据文档” | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| DCR | “自助客户端注册” | RFC 7591 `POST /register` 流程 |
| JWKS | “JWT验证公钥” | JSON Web Key Set，从`jwks_uri`获取，按`kid`索引 |
| 资源指示符 | “受众参数” | RFC 8707 `resource` 参数，将令牌绑定到单个服务器 |
| `aud`声明 | “受众” | 验证器将其与规范资源URL比较的JWT声明 |
| 混淆代理 | “令牌重放” | 将为服务器A签发的令牌提交给服务器B的攻击 |
| `iss`允许列表 | “受信授权服务器” | 在受保护资源元数据`authorization_servers`中指定的集合 |
| 密钥轮换 | “滚动JWKS” | 定期替换签名密钥并带有重叠窗口 |
| 公共客户端 | “原生或浏览器客户端” | 没有`client_secret`的OAuth客户端；PKCE补偿 |
| `WWW-Authenticate` | “401/403响应头” | 承载`Bearer error=...`指令以驱动客户端恢复 |

## 延伸阅读

- [MCP — 授权规范 (2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 本课程实现的MCP认证配置文件
- [RFC 8414 — OAuth 2.0 授权服务器元数据](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 动态客户端注册协议](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636 — 用于代码交换的证明密钥 (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端持有证明
- [RFC 8707 — OAuth 2.0 的资源指示符](https://datatracker.ietf.org/doc/html/rfc8707) — 受众绑定
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [OAuth 2.1 草案](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 整合的OAuth基础