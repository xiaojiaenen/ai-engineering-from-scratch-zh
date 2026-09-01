# 无状态协议上的 MCP Apps

> 交互结果仍然是 MCP 工具和资源交换。2026-07-28 核心使该交换自包含，而 Apps 扩展增加了沙箱化浏览器表面。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 13 · 07（MCP 服务器）、阶段 13 · 10（资源）
**时间：** ~75 分钟

## 学习目标

- 通过 `server/discover` 和逐请求扩展能力来宣告 MCP Apps。
- 在工具被调用前宣告一个 `ui://` 资源。
- 在 2026-07-28 无状态线上返回完整的工具和资源结果。
- 将 Apps `ui/initialize` 桥接消息与已移除的 MCP 核心握手分离。
- 应用源验证、沙箱化、CSP 和最小权限原则。

## 问题

文本结果可以描述时间线。它无法给用户一个他们可以过滤、检查或操作的时间线。

MCP Apps 通过可选扩展解决展示问题。工具定义指向一个 `ui://` 资源。主机可以在工具运行前提取和审查该资源，在沙箱 iframe 中渲染它，并通过 JSON-RPC 桥接调解所有应用操作。

核心协议在 2026-07-28 发生了变化。不要将 App 包装在旧连接生命周期中：

- 没有核心 `initialize` 请求或 `notifications/initialized` 通知。
- 没有 `Mcp-Session-Id` 头部。
- 每个请求都在 `params._meta` 中携带协议版本和客户端能力。
- 服务器实现 `server/discover`，供客户端检查版本、核心能力和扩展。
- 每个成功结果都有一个 `resultType` 判别器。
- Streamable HTTP 对每个请求使用一个 POST。现代 GET 和 DELETE 端点返回 405。

Apps 桥接仍然有一个名为 `ui/initialize` 的方法。它属于 iframe postMessage 方言。它不会重建核心 MCP 会话。

## 概念

### 两种协议，一个功能

保持各层明确：

1. MCP 核心承载 `server/discover`、`tools/list`、`tools/call`、`resources/list` 和 `resources/read`。
2. MCP Apps 扩展声明 UI 并定义 iframe 到主机的桥接。
3. 浏览器沙箱规则限制 UI 的访问范围。

扩展标识符为 `io.modelcontextprotocol/ui`。双方都选择加入。客户端在每个请求的能力对象内发送扩展支持：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "server/discover",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/ui": {}
        }
      },
      "io.modelcontextprotocol/clientInfo": {
        "name": "timeline-host",
        "version": "1.0.0"
      }
    }
  }
}
```

`clientInfo` 建议用于诊断。它是自我报告的数据，不是授权身份。

### 在渲染之前先发现

服务器的发现结果宣告了扩展：

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {},
    "resources": {},
    "extensions": {
      "io.modelcontextprotocol/ui": {}
    }
  },
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "timeline-app-server",
      "version": "2.0.0"
    }
  }
}
```

服务器必须支持发现。客户端不必在每个操作之前调用发现，因为每个操作都携带其自身的能力。

### 在工具定义上宣告 UI

现代 Apps 契约将 UI 绑定到 `tools/list` 中的工具：

```json
{
  "name": "notes_timeline",
  "description": "渲染笔记时间线。",
  "inputSchema": {
    "type": "object",
    "properties": {}
  },
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline.html"
    }
  }
}
```

这是有意为之的预调用元数据。主机可以在结果要求显示它之前预加载、缓存和安全审查 HTML。旧的扁平元数据键可能会被兼容性代码接受，但新服务器应发出嵌套的 `_meta.ui.resourceUri` 形式。

`tools/list` 在当前核心中是可缓存的。包含确定性排序、`ttlMs` 和 `cacheScope`。当可见工具因用户或令牌而异时使用 `private`。

### 返回数据，然后让主机绑定视图

工具调用返回普通内容加上结构化数据：

```json
{
  "resultType": "complete",
  "content": [
    {"type": "text", "text": "时间线已就绪。"}
  ],
  "structuredContent": {
    "notes": [
      {"id": "note-1", "title": "发现", "created": "2026-07-28"}
    ]
  },
  "isError": false
}
```

主机已经知道哪个视图属于该工具。避免发明新的内容块来重复 URI。

### 将应用作为资源提供

服务器在发现中宣告 `resources`，因此它也实现强制性的 `resources/list` 操作。其确定性列表条目包含规范 URI、稳定名称、描述和 MIME 类型。列表结果包含 `resultType`、服务器身份元数据、`ttlMs` 和 `cacheScope`，与确定性工具列表相同。

主机发送 `resources/read`。在 Streamable HTTP 上，请求为：

```text
POST /mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: resources/read
Mcp-Name: ui://notes/timeline.html
```

头部值和 JSON-RPC 正文必须匹配。不匹配则是协议错误 `-32020`。

结果包含 HTML 资源和缓存提示：

```json
{
  "resultType": "complete",
  "contents": [
    {
      "uri": "ui://notes/timeline.html",
      "mimeType": "text/html;profile=mcp-app",
      "text": "<!doctype html>...",
      "_meta": {
        "ui": {
          "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
            "baseUriDomains": []
          },
          "permissions": {}
        }
      }
    }
  ],
  "ttlMs": 60000,
  "cacheScope": "public"
}
```

### 将 UI 资源作为可执行内容缓存

App 资源与普通散文不可互换。其缓存条目可以执行桥接代码、渲染工具数据并请求主机中介的操作。通过规范的 `ui://` URI、认可的服务器身份和版本、资源内容摘要以及 `cacheScope` 为 private 时的授权上下文来建立键。切勿在不同主体之间重用私有 App 资源，因为即使 URI 相同，HTML 或其策略元数据也可能不同。

当其 `ttlMs` 过期、工具的 `_meta.ui.resourceUri` 绑定发生变化、服务器版本或认可的描述符锚点发生变化，或已确认的资源变更订阅命名了该 URI 时，使条目失效。在重新挂载前重新获取并重新应用 CSP 和权限审查。陈旧 iframe 不得仅因为新资源版本尚未加载就保留更广泛的权限。

### 在功能策略之前拒绝线上的歧义

验证有故意的顺序。首先验证 JSON-RPC 形状并要求字符串协议元数据加对象客户端能力映射。接下来将路由头部与正文进行比较。然后才决定匹配的协议版本是否受支持。此顺序可防止代理和服务器解释不同的请求。

| 条件 | HTTP | JSON-RPC 错误 |
|------|------|---------------|
| 头部和正文的版本、方法或名称不一致 | 400 | `-32020` |
| 头部和正文就一个不受支持的版本达成一致 | 400 | `-32022`，`data` 恰好为 `{"supported":["2026-07-28"],"requested":"<actual>"}` |
| `resources/read` 缺少 Apps 扩展能力 | 400 | `-32021`，`data.requiredCapabilities.extensions.io.modelcontextprotocol/ui` |
| 方法未知 | 404 | `-32601` |

JSON-RPC 通知没有 `id`，因此服务器永远不会为其发出 JSON-RPC 响应。接受的 HTTP 通知返回 202 和空正文。错误可以更改 HTTP 状态码，但仍不能为通知创建 JSON-RPC 错误正文。

### 沙箱是边界，而非信任判定

主机控制 iframe。App 无法直接读取主机 cookie、本地存储或页面 DOM。所有特权操作必须穿过桥接。

使用以下默认值：

- 留空所有 CSP 域名列表，然后仅添加 App 需要的源。使用 `connectDomains` 处理 fetch、XHR 和 WebSocket；使用 `resourceDomains` 处理脚本、样式、图片和字体。
- 在实际可行的情况下捆绑代码和数据。
- 除非可见功能需要，否则不请求相机、麦克风或位置权限。
- 将 `postMessage` 固定到确切的对等源，并拒绝来自所有其他源的事件。
- 将工具参数、工具结果、资源文本和桥接消息视为不可信输入。
- 将用户同意保留在主机中。iframe 不能批准自己的关键操作。

不要从教程中复制固定的 `sandbox` 属性到每个主机。主机必须根据 App 的源模型和自身隔离设计来选择标志。

允许的域名仍然是泄露路径。`connectDomains: ["https://api.example.com"]` 意味着在 App 内执行的任何脚本都可以向其发送允许的数据。精确的源匹配可以防止目标混淆，但不能决定有效载荷是否合适。默认保持 connect 访问为空，避免在 iframe 中放置 bearer token，在实际可行时通过主机代理窄操作，限制响应和请求大小，并审计哪个用户操作导致了每个出站请求。将 `resourceDomains` 与 `connectDomains` 分开对待；加载字体或脚本的权限不应授予任意数据上传。

### Apps 桥接有其自身的生命周期

Apps 桥接是 `postMessage` 上的 JSON-RPC 方言。它可以交换 `ui/initialize` 和 `ui/*` 通知，并可以代理类似核心的方法如 `tools/call`。

View 发送带有 `appInfo` 和 `appCapabilities` 对象的 `ui/initialize`。主机返回其能力和主机上下文。只有在该响应之后，View 才发送 `ui/notifications/initialized`。主机必须等待此 Apps 通知后才能向 View 发送消息。

该本地握手创建了一个 iframe 和一个主机帧之间的桥接。它不会协商 MCP 协议版本、创建服务器状态或发行传输会话。注意确切的前缀：核心 `notifications/initialized` 已被移除，而 Apps `ui/notifications/initialized` 仍然存在。由桥接工具调用生成的核心请求是一个具有新 JSON-RPC id 和完整请求元数据的新自包含请求。

### 主机上下文、操作和撤销

桥接初始化后，主机仍然是权威。View 只能通过主机宣告的能力来请求工具操作、导航、剪贴板使用或其他特权效果。主机验证类型化请求、当前用户、目标和参数，应用审批策略，并可能拒绝它。按钮点击和有效的桥接消息表达意图；两者都不授予权限。

将主题、尺寸和可访问性视为变化的主机上下文，而非一次性渲染输入：

- 应用主机提供的颜色和排版标记，然后在主题或对比度偏好变化时响应。
- 让 View 报告期望的维度，但让主机限制并应用 iframe 尺寸，使内容无法脱离其布局或创建欺骗性覆盖。
- 在 iframe 内保持键盘顺序、可见焦点、可访问名称、屏幕阅读器状态、足够对比度、缩放和减少运动行为。
- 调整大小和重新渲染后重新测试主机控件与 View 控件之间的焦点转移。

能力可以在 App 打开时被撤销，因为用户更改账户、策略变化、服务器被隔离或主机缩小同意范围。在操作时检查能力和授权，而不仅是在 `ui/initialize` 期间。撤销时，拒绝待处理的特权调用，停止不再符合策略的网络活动，清除敏感的渲染状态，并在 UI 资源本身不再被认可时重新挂载或回退到文本。View 必须将拒绝视为正常结果，而不是重试直到主机让步。

### 回退是合同的一部分

感知 Apps 的服务器仍然可以为不宣告 UI 扩展的主机提供服务：

- 在 `tools/list` 中返回不带 `_meta.ui` 的相同工具。
- 为 `tools/call` 保留有用的文本结果。
- 以缺失能力错误拒绝 UI 的 `resources/read`。
- 在决定工具是否完成时绝不假设 iframe 存在。

```figure
t3-ui-sandbox
```

## 构建它

`code/main.py` 构建了一个小型进程内协议模型，无需 SDK。它验证当前请求信封和 Streamable HTTP 路由值，通过 `server/discover` 宣告 Apps，列出工具和资源，执行工具，并提供自包含的 HTML 资源。

该模型接收已解析的正文和路由头部。它不是完整的 HTTP 适配器，也不解析 `Content-Type` 或 `Accept`。使用课程 09 中的完整 Streamable HTTP 适配器，它要求 `Content-Type: application/json` 和包含 `application/json` 与 `text/event-stream` 的 `Accept` 值。

运行它：

```bash
cd phases/13-tools-and-protocols/14-mcp-apps
python3 code/main.py
python3 -m unittest discover code/tests -v
```

在输出中检查四件事：

1. 每次调用都是独立的。
2. 每个请求都有 `_meta` 能力。
3. `resources/list` 在任何资源读取之前返回稳定描述符。
4. 每个结果都有 `resultType` 和服务器身份元数据。
5. 没有核心会话标识符出现。

## 使用它

从 `server/discover` 开始。确认 `io.modelcontextprotocol/ui` 出现在服务器扩展映射中。然后调用 `tools/list` 两次，一次带 Apps 能力，一次不带。第一次响应宣告了该资源。第二次保持为一个可用的纯文本工具。

读取 `ui://notes/timeline.html`。在 HTML 中搜索 `hostOrigin` 和 `event.origin` 守卫。这两行是最小可见证明，表明桥接不使用通配符目标。

## 交付

本课交付 `outputs/skill-mcp-apps-spec.md`。用它来在编写框架代码之前审查 App 合同。它迫使作者声明当前核心信封、扩展协商、回退、UI 资源、缓存策略、CSP、权限、桥接方法和同意边界。

## 练习

1. 将客户端能力更改为空扩展映射。确认 `tools/list` 保留工具但移除 UI 绑定。
2. 发送 `Mcp-Name: ui://notes/other.html` 同时正文读取时间线。确认错误 `-32020`。
3. 将资源更改为 `cacheScope: private`。描述证明其合理的用户特定条件。
4. 将脚本移至 `https://static.example.com/app.js`。将该源添加到 `resourceDomains` 并解释新的供应链风险。
5. 添加一个 `notes_open` 工具并通过主机路由按钮点击。将用户批准保留在主机中。

## 关键术语

| 术语 | 含义 |
|------|------|
| MCP Apps | 由 MCP 主机渲染的交互式 HTML 可选扩展 |
| `io.modelcontextprotocol/ui` | 双方宣告的扩展标识符 |
| `ui://` | App UI 模板的资源方案 |
| `text/html;profile=mcp-app` | MCP App HTML 的 MIME 类型 |
| `server/discover` | 当前用于协议和能力发现的 RPC |
| `resources/list` | 服务器宣告资源时的强制资源列岀方法 |
| `resultType` | 现代成功结果的必需判别器 |
| `ui/initialize` | 第一个 Apps 桥接请求，与已移除的核心初始化分离 |
| `ui/notifications/initialized` | 主机响应后发送的 Apps View 就绪通知 |
| CSP | 限制脚本、样式、图片和网络源范围的浏览器策略 |
| 文本回退 | 为无 Apps 支持的主机保留的工具行为 |

## 延伸阅读

- [MCP 2026-07-28 基础协议](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Apps 概览](https://modelcontextprotocol.io/extensions/apps/overview)
- [MCP Apps 构建指南](https://modelcontextprotocol.io/extensions/apps/build)
- [官方扩展支持矩阵](https://modelcontextprotocol.io/extensions/client-matrix)
