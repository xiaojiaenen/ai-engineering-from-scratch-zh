# MCP Apps — 通过 `ui://` 实现的交互式 UI 资源

> 纯文本工具输出限制了代理的展示能力。MCP Apps (SEP-1724，2026年1月26日正式发布) 允许工具返回沙盒化的交互式HTML，并在Claude Desktop、ChatGPT、Cursor、Goose和VS Code中内联渲染。仪表盘、表单、地图、3D场景，全通过一个扩展实现。本课将介绍 `ui://` 资源方案、`text/html;profile=mcp-app` MIME类型、iframe沙盒 postMessage 协议，以及允许服务器渲染HTML所带来的安全面。

**类型:** 构建
**语言:** Python（标准库，UI资源发射器）、HTML（示例应用）
**先决条件:** 阶段 13 · 07（MCP服务器）、阶段 13 · 10（资源）
**时间:** 约 75 分钟

## 学习目标

- 从工具调用返回 `ui://` 资源，并设置正确的MIME和元数据。
- 使用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联的UI。
- 实现iframe沙盒 postMessage JSON-RPC，用于UI与宿主通信。
- 应用默认的CSP和权限策略，以防御源自UI的攻击。

## 问题所在

2025年代的 `visualize_timeline` 工具可以返回“这是按时间顺序组织的14条笔记：...”。这是一段文本。用户实际想要的是交互式时间线。在MCP Apps出现之前，选项是：特定客户端的部件API（Claude的artifacts、OpenAI的Custom GPT HTML），或者完全没有UI。

MCP Apps (SEP-1724，2026年1月26日发布) 标准化了这一契约。工具结果包含一个 `resource`，其URI为 `ui://...`，MIME类型为 `text/html;profile=mcp-app`。宿主在沙盒化的iframe中渲染它，该iframe具有受限的CSP，除非明确授予，否则没有网络访问。iframe内的UI通过一个小型的postMessage JSON-RPC方言向宿主发送消息。

每个兼容客户端（Claude Desktop、ChatGPT、Goose、VS Code）都以相同方式渲染相同的 `ui://` 资源。一个服务器，一个HTML包，通用UI。

## 概念介绍

### `ui://` 资源方案

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

然后宿主对 `ui://notes/timeline` URI 调用 `resources/read`，并得到：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe 沙盒

宿主在沙盒化的 `<iframe>` 中渲染HTML，具有以下限制：

- `sandbox="allow-scripts allow-same-origin"`（或根据服务器声明采用更严格的策略）
- 通过响应头应用服务器声明的CSP。
- 没有来自宿主源的cookies，没有localStorage。
- 网络访问仅限于CSP中的 `connectSrc`。

### postMessage 协议

iframe通过 `window.postMessage` 与宿主通信。一个小型的JSON-RPC 2.0方言：

务必始终将 `targetOrigin` 固定为对等方的确切源，并且在接收端处理任何有效负载前，根据允许列表验证 `event.origin`。永远不要在此通道的任何一端使用 `"*"` — 消息体承载工具调用和资源读取。

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

UI可调用的可用宿主端方法：

- `host.callTool(name, arguments)` — 调用服务器工具。
- `host.readResource(uri)` — 读取MCP资源。
- `host.getPrompt(name, arguments)` — 获取提示模板。
- `host.close()` — 关闭UI。

每个调用仍然经过MCP协议，并继承服务器的权限。

### 权限

`_meta.ui.permissions` 列表请求额外的能力：

- `camera` — 访问用户的摄像头（用于文档扫描UI）。
- `microphone` — 语音输入。
- `geolocation` — 位置信息。
- `network:*` — 比 `connectSrc` 单独允许的更宽的网络访问。

每项权限是用户在UI渲染前看到的一个提示。

### 安全风险

iframe中的HTML仍然是HTML。新的攻击面：

- **通过UI的提示注入。** 恶意服务器UI可以显示看起来像系统消息的文本并欺骗用户。宿主渲染应明显区分服务器UI和宿主UI。
- **通过 `connectSrc` 的数据泄露。** 如果CSP允许 `connect-src: *`，UI可以向任何地方发送数据。默认策略应严格。
- **点击劫持。** UI覆盖宿主界面。宿主必须防止z-index操纵并强制执行不透明度规则。
- **窃取焦点。** UI获取键盘焦点并捕获下一条消息。宿主必须拦截。

阶段 13 · 15 将作为MCP安全部分深入探讨这些内容；本课进行介绍。

### `ui/initialize` 握手

iframe加载后，它通过postMessage发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主响应以能力和会话令牌。UI在后续的每次宿主调用中使用该会话令牌。

### AppRenderer / AppFrame SDK 基元

ext-apps SDK 提供了两个便捷基元：

- `AppRenderer`（服务器端）— 包装React / Vue / Solid组件，并发出一个带有正确MIME和元数据的 `ui://` 资源。
- `AppFrame`（客户端）— 接收资源，挂载iframe，并协调postMessage。

你可以使用这些，或者自行编写HTML和JSON-RPC。

### 生态系统状态

MCP Apps于2026年1月26日发布。截至2026年4月的客户端支持情况：

- **Claude Desktop。** 自2026年1月起完全支持。
- **ChatGPT。** 通过Apps SDK完全支持（底层相同的MCP Apps协议）。
- **Cursor。** 测试版；通过设置启用。
- **VS Code。** 仅限Insider版本。
- **Goose。** 完全支持。
- **Zed, Windsurf。** 已列入路线图。

生产中的服务器：仪表盘、地图可视化、数据表、图表构建器、沙盒IDE预览。

## 使用它

`code/main.py` 扩展了笔记服务器，增加了一个 `visualize_timeline` 工具，该工具返回一个 `ui://notes/timeline` 资源，以及一个针对该URI的 `resources/read` 处理程序，该处理程序返回一个小型但完整的HTML包，其中包含一个SVG时间线。HTML是标准库模板化的——无需构建系统。postMessage在JS注释中进行了概述，因为标准库无法驱动浏览器。

需要关注的地方：

- 工具响应上的 `_meta.ui` 携带resourceUri、CSP、权限。
- HTML在没有网络访问的情况下渲染；所有数据都是内联的。
- JS通过 `window.parent.postMessage` 调用 `host.callTool`（在此标准库演示中有文档记录但处于非活动状态）。

## 部署它

本课产生 `outputs/skill-mcp-apps-spec.md`。给定一个可以从交互式UI中受益的工具，该技能产生完整的MCP Apps契约：`ui://` URI、CSP、权限、postMessage入口点和安全检查清单。

## 练习

1. 运行 `code/main.py` 并检查发出的HTML。在浏览器中直接打开该HTML；验证SVG是否渲染。然后勾勒出UI将用来调用 `host.callTool("notes_update", ...)` 的postMessage契约。

2. 收紧CSP：移除 `'unsafe-inline'` 并使用基于nonce的脚本策略。HTML生成代码需要做什么更改？

3. 添加第二个UI资源 `ui://notes/editor`，其中包含一个用于就地编辑笔记的表单。当用户提交时，iframe调用 `host.callTool("notes_update", ...)`。

4. 审计UI的攻击面。恶意服务器可以在哪里注入内容？iframe沙盒防御什么，不防御什么？

5. 阅读SEP-1724规范，并找出一个此玩具实现未使用的MCP Apps SDK功能。（提示：组件级状态同步。）

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| MCP Apps | "交互式UI资源" | 2026-01-26发布的SEP-1724扩展 |
| `ui://` | "应用URI方案" | UI包的资源方案 |
| `text/html;profile=mcp-app` | "MIME类型" | MCP应用HTML的内容类型 |
| Iframe沙盒 | "渲染容器" | 使用CSP和权限对UI进行浏览器沙盒隔离 |
| postMessage JSON-RPC | "UI到宿主线" | 用于宿主调用的小型JSON-RPC-over-postMessage方言 |
| `_meta.ui` | "工具-UI绑定" | 将工具结果链接到UI资源的元数据 |
| CSP | "Content-Security-Policy" | 声明允许的脚本、网络、样式来源 |
| AppRenderer | "服务器SDK基元" | 将框架组件转换为 `ui://` 资源 |
| AppFrame | "客户端SDK基元" | 协调postMessage的iframe挂载助手 |
| `ui/initialize` | "握手" | 从UI到宿主的第一个postMessage |

## 扩展阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 参考实现和SDK
- [MCP Apps规范 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式规范文档
- [MCP — Apps扩展概述](https://modelcontextprotocol.io/extensions/apps/overview) — 高级文档
- [MCP博客 — MCP Apps发布](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026年1月发布文章
- [MCP Apps API参考](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc风格的SDK参考