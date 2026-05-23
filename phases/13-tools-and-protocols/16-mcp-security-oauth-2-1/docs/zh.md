# MCP 安全性 II — OAuth 2.1、资源指示符与增量作用域

> 远程 MCP 服务器需要授权，而不仅仅是身份验证。2025-11-25 规范与 OAuth 2.1 + PKCE + 资源指示符 (RFC 8707) + 受保护资源元数据 (RFC 9728) 保持一致。SEP-835 添加了增量作用域同意，并在 403 WWW-Authenticate 响应时进行升级授权。本课将升级授权流程实现为状态机，以便您查看每一步。

**类型:** 构建
**语言:** Python (标准库, OAuth 状态机模拟器)
**前置要求:** 阶段 13 · 09 (传输层), 阶段 13 · 15 (安全性 I)
**时长:** ~75 分钟

## 学习目标

- 区分资源服务器与授权服务器的职责。
- 遍历受 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource` (RFC 8707) 和受保护资源元数据 (RFC 9728) 防止混淆代理攻击。
- 实现升级授权：服务器使用 WWW-Authenticate 响应 403，请求更高权限作用域；客户端重新提示用户同意并重试。

## 问题

早期 MCP（2025 年之前）提供的远程服务器使用临时 API 密钥，甚至没有身份验证。2025-11-25 规范通过完整的 OAuth 2.1 配置弥补了这一缺陷。

三个实际需求：

- **普通远程服务器。** 用户安装访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。带 PKCE 的 OAuth 2.1 是合适的方案。
- **作用域升级。** 一个笔记服务器授予了 `notes:read` 权限，之后可能需要 `notes:write` 权限来执行特定操作。无需重新执行整个流程，升级授权（SEP-835）会请求附加作用域。
- **防止混淆代理。** 客户端持有一个针对服务器 A 的、具有受众范围限定的令牌。服务器 A 是恶意的，尝试将该令牌提交给服务器 B。资源指示符 (RFC 8707) 将令牌绑定到其预期受众。

OAuth 2.1 并不是新的。新的是 MCP 的配置：强制指定的流程（仅授权码 + PKCE；默认不允许隐式模式，不允许客户端凭证模式），每次令牌请求都必须包含资源指示符，以及发布受保护资源元数据，以便客户端知道去哪里获取。

## 概念

### 角色

- **客户端。** MCP 客户端（Claude Desktop、Cursor 等）。
- **资源服务器。** MCP 服务器（笔记、GitHub、Postgres 等）。
- **授权服务器。** 颁发令牌。可以与资源服务器是同一服务，也可以是独立的 IdP（Auth0、Keycloak、Cognito）。

在 MCP 配置中，资源服务器和授权服务器*可以*是同一主机，但*应*通过 URL 区分。

### 授权码 + PKCE

流程如下：

1. 客户端生成 `code_verifier`（随机）和 `code_challenge`（SHA256 哈希）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意授权。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端向 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...` 发起 POST 请求。
5. 授权服务器将验证者的哈希与存储的挑战进行验证，并颁发访问令牌。
6. 客户端使用该令牌：在每次向资源服务器的请求中都携带 `Authorization: Bearer ...`。

PKCE 可防止授权码拦截攻击。资源指示符可防止令牌在其他地方生效。

### 受保护资源元数据 (RFC 9728)

资源服务器发布一个 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。减少了配置工作量——客户端只需要资源 URL。

### 资源指示符 (RFC 8707)

令牌请求中的 `resource` 参数将令牌的预期受众固定。颁发的令牌包含 `aud: "https://notes.example.com"`。收到此令牌的另一个 MCP 服务器会检查 `aud` 并拒绝它。

### 作用域模型

作用域是空格分隔的字符串。常见的 MCP 约定：

- `notes:read`, `notes:write`, `notes:delete`
- `admin:*` 用于管理能力（谨慎使用）
- `profile:read` 用于身份标识

作用域选择应遵循最小权限原则：只请求当前所需，需要更多时再升级。

### 升级授权 (SEP-835)

用户授予了 `notes:read` 权限。他们随后要求代理删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端看到 `insufficient_scope` 错误，会提示用户同意附加作用域，为其执行一个迷你 OAuth 流程，然后使用新令牌重试请求。

### 令牌受众验证

每次请求：服务器检查 `token.aud == self.resource_url`。不匹配则返回 401。这可以阻止跨服务器的令牌重用。

### 短生命周期令牌与轮换

访问令牌*应*是短生命周期的（默认 1 小时）。刷新令牌在每次刷新时轮换。客户端在后台处理静默刷新。

### 无令牌透传

采样服务器（阶段 13 · 11）*禁止*将客户端的令牌透传给其他服务。采样请求是边界。

### 防止混淆代理

令牌绑定到 `aud`。客户端绑定到 `client_id`。每次请求都会根据这两者进行验证。规范明确禁止了在 MCP 之前的远程工具生态系统中常见的“令牌传递”模式。

### 客户端 ID 发现

每个 MCP 客户端在固定 URL 上发布其元数据。授权服务器可以获取客户端的元数据文档，以发现重定向 URI 和联系信息。这消除了手动客户端注册。

### 网关与 OAuth

阶段 13 · 17 展示了企业网关如何处理 OAuth：网关持有上游服务器的凭据，颁发给客户端的令牌是网关颁发的，上游令牌永远不会离开网关。这颠覆了信任模型——用户只需向网关进行一次身份验证；网关处理 N 个服务器的授权。

## 实践应用

`code/main.py` 将完整的 OAuth 2.1 升级授权流程模拟为状态机。它实现了：

- PKCE code-verifier / challenge 生成。
- 带资源指示符的授权码流程。
- 受保护资源元数据端点。
- 带受众检查的令牌验证。
- `insufficient_scope` 时的升级授权。

本课不包含 HTTP 服务器；状态机在内存中运行，以便您可以跟踪每一步。阶段 13 · 17 的网关课程将其连接到实际传输层。

## 交付成果

本课产出 `outputs/skill-oauth-scope-planner.md`。给定一个带有工具的远程 MCP 服务器，该技能负责设计作用域集合、固定规则和升级授权策略。

## 练习

1. 运行 `code/main.py`。跟踪双作用域的升级授权流程。注意在升级时哪些步骤会重复。

2. 添加刷新令牌轮换：每次刷新都颁发一个新的刷新令牌并使旧的失效。模拟一个被窃取的刷新令牌在轮换后被使用，并确认其会失败。

3. 使用标准库 http.server 将受保护资源元数据端点实现为真实的 HTTP 响应。镜像第 09 课中的 `/mcp` 端点。

4. 为 GitHub MCP 服务器设计一个作用域层级结构：读取仓库、写入 PR、批准 PR、合并 PR、管理员。在每个级别之间使用升级授权。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中 MCP 使用方式与 RFC 示例不同的一个字段。（提示：它与 `scopes_supported` 有关。）

## 关键术语

| 术语 | 人们常说 | 它的实际含义 |
|------|----------|--------------|
| OAuth 2.1 | "现代 OAuth" | 整合了 RFC，强制要求 PKCE 并禁止隐式流程 |
| PKCE | "持有证明" | 代码验证器 + 挑战值，用于防御授权码拦截攻击 |
| 资源指示符 | "令牌受众" | RFC 8707 中的 `resource` 参数，将令牌固定到一台服务器 |
| 受保护资源元数据 | "发现文档" | RFC 9728 中的 `.well-known/oauth-protected-resource` |
| 升级授权 | "增量同意" | SEP-835 中按需添加作用域的流程 |
| `insufficient_scope` | "403 带 WWW-Authenticate" | 服务器发出的信号，要求为更大的作用域重新同意 |
| 混淆代理 | "跨服务令牌重用" | 一种攻击，其中受信任的持有者不当地转发令牌 |
| 短生命周期令牌 | "访问令牌 TTL" | 会很快过期的持有者令牌；通过刷新令牌续期 |
| 作用域层级 | "最小权限栈" | 分级的作用域集合，在各级别之间进行升级 |
| 客户端 ID 元数据 | "客户端发现文档" | 客户端发布其自身 OAuth 元数据的 URL |

## 延伸阅读

- [MCP — 授权规范](https://modelcontextprotocol.io/specification/draft/basic/authorization) — MCP 的规范 OAuth 配置
- [den.dev — MCP 十一月授权规范](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更的详细解读
- [RFC 8707 — OAuth 2.0 的资源指示符](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定的 RFC
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档的 RFC
- [Aembit — MCP OAuth 2.1, PKCE 与 AI 授权的未来](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实用的升级授权流程详解