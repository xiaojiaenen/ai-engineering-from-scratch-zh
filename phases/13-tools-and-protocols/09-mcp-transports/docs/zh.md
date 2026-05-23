# MCP 传输协议 — stdio 与 Streamable HTTP 与 SSE 迁移指南

> stdio 仅在本地工作，无法用于远程。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE 传输协议已弃用，并将在2026年中期移除。选择错误的传输协议将导致迁移成本；选择正确的则可获得一个支持会话连续性和 DNS 重绑定保护的、可远程托管的 MCP 服务器。

**类型：** 学习指南
**语言：** Python（标准库，Streamable HTTP 端点骨架）
**先决条件：** 阶段 13 · 07, 08（MCP 服务器与客户端）
**时间：** 约 45 分钟

## 学习目标

- 根据部署形态（本地 vs 远程，单进程 vs 集群）在 stdio 和 Streamable HTTP 之间做出选择。
- 实现 Streamable HTTP 单端点模式：POST 用于请求，GET 用于会话流。
- 强制执行 `Origin` 验证和会话ID语义以防御 DNS 重绑定攻击。
- 在2026年中期移除截止日期前，将旧版 HTTP+SSE 服务器迁移至 Streamable HTTP。

## 问题背景

首个 MCP 远程传输协议（2024-11）是 HTTP+SSE：包含两个端点，一个用于客户端的 POST 请求，另一个用于服务器到客户端的流式 Server-Sent-Events 通道。它能工作，但也显得笨拙：每个会话需要两个端点，某些 CDN 前的缓存会失效，并且严重依赖长连接 SSE，而某些 WAF 会主动终止这些连接。

2025-03-26 规范用 Streamable HTTP 取代了它：单端点，POST 用于客户端请求，GET 用于建立会话流，两者共享 `Mcp-Session-Id` 标头。此后构建或迁移的所有服务器都使用 Streamable HTTP。旧的 SSE 模式正在被弃用 — Atlassian Rovo 于2026年6月30日移除；Keboola 于2026年4月1日移除；大多数剩余的企业服务器将在2026年底前完成。

而 stdio 对于本地服务器仍然很重要。Claude Desktop、VS Code 以及所有类 IDE 客户端都通过 stdio 启动服务器。正确的思维模型是：stdio 用于“本机”，Streamable HTTP 用于“跨网络”。两者无交集。

## 核心概念

### stdio

- 子进程传输。客户端启动服务器，通过 stdin/stdout 通信。
- 每行一个 JSON 对象。换行符分隔。
- 无会话 ID；进程身份即会话。
- 无需认证（子进程继承父进程的信任边界）。
- 切勿用于远程服务器 — 你将需要 SSH 或 socat 进行隧道传输，此时应使用 Streamable HTTP。

### Streamable HTTP

单端点 `/mcp`（或任意路径）。支持三种 HTTP 方法：

- **POST /mcp。** 客户端发送 JSON-RPC 消息。服务器回复单个 JSON 响应，或一个 SSE 流，其中包含一个或多个响应（适用于批量响应和与该请求相关的通知）。
- **GET /mcp。** 客户端打开一个长连接 SSE 通道。服务器用它来发送服务器到客户端的请求（采样、通知、信息诱导）。
- **DELETE /mcp。** 客户端显式终止会话。

会话由服务器在第一个响应中设置的 `Mcp-Session-Id` 标头标识，客户端在后续每个请求中回显此标头。会话 ID 必须是加密随机的（128位以上）；为安全起见，客户端选择的 ID 会被拒绝。

### 单端点与双端点模式

旧规范中的双端点模式在2026年仍可调用 — 规范声明其为“旧版兼容”。但所有新服务器都应采用单端点模式。官方 SDK 生成单端点；仅在与未迁移的远程服务器通信时才使用旧模式。

### `Origin` 验证与 DNS 重绑定

浏览器目前不是 MCP 客户端，但攻击者可以制作一个网页，诱使浏览器向 `localhost:1234/mcp` 发送 POST 请求 — 而用户的本地 MCP 服务器正在监听此地址。如果服务器不检查 `Origin`，浏览器的同源策略也无法保护它，因为 `Origin: http://evil.com` 是合法的跨域来源。

2025-11-25 规范要求服务器拒绝其 `Origin` 不在允许列表中的请求。允许列表通常包含 MCP 客户端主机（`https://claude.ai`、`vscode-webview://*`）以及用于本地 UI 的 localhost 变体。

### 会话 ID 生命周期

1.  客户端发送第一个请求，不包含 `Mcp-Session-Id`。
2.  服务器分配一个随机 ID，在响应标头中设置 `Mcp-Session-Id`。
3.  客户端在后续所有请求以及用于流的 `GET /mcp` 中回显该标头。
4.  会话可被服务器撤销；客户端在后续请求中会看到 404，必须重新初始化。
5.  客户端可以显式 DELETE 会话以实现干净关闭。

### 保活与重连

SSE 连接会中断。客户端通过使用相同的 `Mcp-Session-Id` 重新 GET 来重建连接。服务器必须将中断期间（在合理窗口内）丢失的事件排队，并通过客户端回显的 `last-event-id` 标头进行重放。

阶段 13 · 13 讲解了任务（Tasks），它能让长时间运行的工作在完全会话重连后也能存活。

### 向后兼容性探测

一个希望同时支持新旧服务器的客户端：

1.  向 `/mcp` 发送 POST 请求。
2.  如果响应是 `200 OK` 且包含 JSON 或 SSE，则为 Streamable HTTP。
3.  如果响应是 `200 OK` 且包含 `Content-Type: text/event-stream` 以及一个指向次要端点的 `Location` 标头，则为旧版 HTTP+SSE；遵循 `Location` 操作。

### Cloudflare、ngrok 和托管

2026年生产环境中的远程 MCP 服务器运行在 Cloudflare Workers（配合其 MCP Agents SDK）、Vercel Functions 或容器化的 Node/Python 上。关键点：你的托管环境必须支持 SSE GET 所需的长时间 HTTP 连接。Vercel 的免费套餐限制为10秒，不适用。Cloudflare Workers 支持无限流。

### 网关组合

当你使用网关（阶段 13 · 17）将多个 MCP 服务器前置时，该网关是一个单一的 Streamable HTTP 端点，负责重写会话 ID 并多路复用上游。工具在网关层合并；客户端看到的是一个单一的逻辑服务器。

### 传输协议故障模式

- **stdio SIGPIPE。** 子进程在写入过程中死亡会引发 SIGPIPE；服务器应干净地退出。客户端应检测 EOF 并将会话标记为死亡。
- **HTTP 502 / 504。** Cloudflare、nginx 和其他代理在上游故障时发出这些响应。Streamable HTTP 客户端应在短时间退避后重试一次。
- **SSE 连接中断。** TCP RST、代理超时或客户端网络变更关闭了流。客户端使用 `Mcp-Session-Id` 和可选的 `last-event-id` 重新连接以恢复。
- **会话撤销。** 服务器使一个会话 ID 失效；客户端在下次请求时看到 404。客户端必须重新握手。
- **时钟偏移。** 客户端的资源 TTL 计算与服务器不一致。客户端应将服务器时间戳视为权威。

### 何时绕过 Streamable HTTP

一些企业在其内部网络中将 MCP 服务器部署在 gRPC 或消息队列传输协议之后。这是非标准的 — MCP 规范并未正式定义这些。网关可以向 MCP 客户端暴露一个 Streamable HTTP 接口，同时内部使用 gRPC。保持外部接口符合规范；网关负责翻译工作。

## 实践指南

`code/main.py` 使用 `http.server`（标准库）实现了一个最小的 Streamable HTTP 端点。它在 `/mcp` 上处理 POST、GET 和 DELETE，在第一个响应中设置 `Mcp-Session-Id`，验证 `Origin`，并拒绝来自非允许列表源的请求。处理程序复用了课程07笔记服务器的调度逻辑。

需要关注的部分：

- POST 处理程序读取 JSON-RPC 主体，进行调度，并写入 JSON 响应（单响应变体；SSE 变体在结构上类似）。
- `Origin` 检查拒绝了默认的 `http://evil.example` 探测，但接受 `http://localhost`。
- 会话 ID 是随机的128位十六进制字符串；服务器在内存中保持每个会话的状态。

## 部署应用

本课程产生 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE（旧版）MCP 服务器，该技能将生成一份迁移至 Streamable HTTP 的计划，包括会话ID连续性、Origin 检查和向后兼容性探测支持。

## 练习

1.  运行 `code/main.py`。从 `curl` 发送一个 `initialize` POST 请求，并观察 `Mcp-Session-Id` 响应标头。发送第二个请求回显该标头，并验证会话连续性。

2.  添加一个 GET 处理程序以打开一个 SSE 流。每五秒发送一个 `notifications/progress` 事件。使用相同的会话 ID 重新 GET 进行重连，并确认服务器接受了它。

3.  实现 `last-event-id` 重放逻辑。在重连时，重放自该 ID 以来生成的任何事件。

4.  扩展 `Origin` 验证以支持通配符模式（`https://*.example.com`），并确认它接受 `https://app.example.com` 但拒绝 `https://evil.example.com.attacker.net`。

5.  从官方注册表中选取一个旧版 HTTP+SSE 服务器（有几个），并规划迁移：端点处理、会话ID生成和标头语义方面需要做哪些更改。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| stdio 传输协议 | “本地子进程” | 通过 stdin/stdout 的 JSON-RPC 通信，换行符分隔 |
| Streamable HTTP | “远程传输协议” | 单端点 POST + GET + 可选 SSE，2025-03-26 规范 |
| HTTP+SSE | “旧版” | 双端点模型，将在2026年中期移除 |
| `Mcp-Session-Id` | “会话标头” | 服务器分配的随机 ID，在后续每个请求中回显 |
| `Origin` 允许列表 | “DNS 重绑定防御” | 拒绝 Origin 未获批准的请求 |
| 单端点 | “一个 URL” | `/mcp` 处理所有会话操作的 POST / GET / DELETE |
| `last-event-id` | “SSE 重放” | 用于在不丢失事件的情况下恢复中断流的标头 |
| 向后兼容性探测 | “新旧检测” | 客户端通过响应形状检查自动选择传输协议 |
| 长连接 HTTP | “SSE 流式传输” | 服务器在一条 TCP 连接上推送事件数分钟或数小时 |
| 会话撤销 | “强制重新初始化” | 服务器使一个会话 ID 失效；客户端必须重新握手 |

## 延伸阅读

- [MCP — 基础传输协议规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 和 Streamable HTTP 的规范参考
- [MCP — 基础传输协议规范 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的修订版
- [Cloudflare — MCP 传输协议](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers 托管的 Streamable HTTP 模式
- [AWS — MCP 传输机制](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 跨部署形态的比较
- [Atlassian — HTTP+SSE 弃用通知](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体的迁移截止日期示例