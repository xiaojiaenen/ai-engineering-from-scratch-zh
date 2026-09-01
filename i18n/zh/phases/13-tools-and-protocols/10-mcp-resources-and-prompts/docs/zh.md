# MCP 资源与提示词：有地址上下文的状态无服务器

> 工具用于执行操作，资源暴露可寻址的内容，提示词包装用户选定的消息模板。一个好的 MCP 服务器会把这些职责分离并保持可预测性。

**类型：** 构建
**语言：** Python
**前置条件：** 第 13 阶段第 7 课（构建 MCP 服务器）、第 13 阶段第 9 课（MCP 传输层）
**时间：** 约 60 分钟

## 学习目标

- 从消费者意图出发选择工具、资源与提示词。
- 通过强制的 `server/discover` 声明资源与提示词能力。
- 构建确定性排序的 `resources/list` 与 `prompts/list` 结果。
- 应用 `ttlMs` 和 `cacheScope` 而不泄漏用户特定数据。
- 对无效或未知的资源 URI 返回 JSON-RPC 错误 `-32602`。
- 打开 `subscriptions/listen` POST 响应流并按订阅 ID 关联每个事件。
- 将资源内容与提示词模板视为不可信的服务端输出。

## 从消费者出发

误用 MCP 最简单的方式是从实现代码开始。一个数据库查询被做成工具，因为函数更熟悉。一段可复用的流程被做成资源，因为它存储在文件里。一个提示词被隐藏为策略，因为宿主可以注入它。

要先想清楚谁来做选择，以及他们期望什么。

| 原语 | 主要意图 | 选择权归属 | 典型结果 |
|---|---|---|---|
| 工具 | 执行操作 | 模型或应用程序 | 结构化的操作结果 |
| 资源 | 读取 URI 处的内容 | 宿主、应用程序或用户 | 文本或二进制内容 |
| 提示词 | 启动可复用的消息流程 | 用户通过宿主 UI | 一条或多条提示词消息 |

`notes://note-1` 是资源，因为它是可寻址的内容。`delete_note` 是工具，因为它改变状态。`review_note` 是提示词，因为用户选择了一个准备好的审查流程。

不要因为看起来完整就把一个操作同时暴露为三种原语。每一种额外的暴露面都需要发现、授权、缓存、错误处理、测试与文档。

## 2026-07-28 状态无信封

本课时面向 MCP 协议版本 `2026-07-28`。在此配置中不存在初始化握手或协议会话。每条请求都在保留的 `_meta` 键中携带协议版本与客户端能力。

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

服务器必须实现 `server/discover`。其结果会声明支持的版本、资源与提示词能力、实现身份与缓存提示。客户端可以直接调用其他方法，但发现机制能在它构建 UI 之前提供一次稳定的快照。

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "resources": {"listChanged": true, "subscribe": true},
    "prompts": {"listChanged": true}
  },
  "ttlMs": 3600000,
  "cacheScope": "public"
}
```

正常结果会声明 `"resultType": "complete"`。响应 `_meta` 通过 `io.modelcontextprotocol/serverInfo` 标识服务实现，这有助于诊断，但不是认证身份。携带不支持版本的请求会返回 `-32022`，并包含请求版本与服务支持的版本。

状态无契约改变了你的设计直觉：列表不能依赖同一条连接上的前序调用。授权可能改变可见集合，因为凭据是请求输入，但连接历史不能。

## 资源是稳定的 URI 契约

资源是由 URI 标识的内容。应在处理器之前先设计 URI。

好的 URI 属性：

- 足够稳定，可以书签化或在请求之间传递。
- 属于服务器的命名空间。
- 独立于进程 ID 或连接。
- 在访问存储前进行校验。
- 每次读取都经过授权检查。

`notes://note-1` 优于 `note-1`，因为它的命名空间是显式的。文件服务器可以使用 `file://` URI，但在解析符号链接与相对路径段之后仍需检查配置的目录边界。

`resources/list` 返回调用者当前可见的资源。按稳定键（如 URI）排序。确定性顺序可避免频繁缓存未命中、快照抖动，以及在不同刷新间跳动的宿主 UI。

```json
{
  "resultType": "complete",
  "resources": [
    {
      "uri": "notes://note-1",
      "name": "Architecture decision",
      "description": "Why the service uses a stateless boundary",
      "mimeType": "text/markdown"
    }
  ],
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

`resources/read` 返回一个或多个内容项。未知 URI 不是成功的空读取。当前的 Resources 规范将无效或未知资源 URI 归类为 JSON-RPC 无效参数，错误码 `-32602`。

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "error": {
    "code": -32602,
    "message": "Unknown or invalid resource URI",
    "data": {
      "uri": "notes://missing"
    }
  }
}
```

这种区分让客户端能够区分“不存在”与“有效的空文档”。同时也防止意外回退到范围更广的查找。

### 资源模板

资源模板描述了一组参数化 URI。当枚举所有具体项代价高昂或不可穷举时使用。例如 `notes://projects/{project}/decisions/{decision}` 告诉客户端如何构造合法地址，而无需返回每一条决策。

模板不会弱化校验。解析变量、应用授权、强制执行长度与字符限制，并使用类型化参数构造存储查询。绝不能将任意 URI 尾部拼接到文件系统路径或数据库语句中。

### 内容不等于可信指令

资源文本可能包含提示词注入、密钥、误导性命令或格式错误的标记。宿主应保留来源信息并将资源内容视为数据。服务器应限制内容大小、返回准确的 MIME 类型、脱敏调用方无权访问的字段，并避免返回不相关的记录。

## 提示词是用户可控的模板

MCP 提示词专为显式用户选择而设计。宿主可以将它们渲染为斜杠命令、菜单项或工作流按钮。协议并不要求单一 UI。

对于相同的请求授权，`prompts/list` 应当是确定性的。每个提示词需要稳定的名称、有用的描述，以及允许宿主在 `prompts/get` 之前收集输入的参数声明。

```json
{
  "resultType": "complete",
  "prompts": [
    {
      "name": "review_note",
      "title": "Review a note",
      "description": "Review one note for a named concern",
      "arguments": [
        {
          "name": "uri",
          "description": "The note resource URI",
          "required": true
        }
      ]
    }
  ],
  "ttlMs": 600000,
  "cacheScope": "public"
}
```

`prompts/get` 将参数解析为消息。它不会替换宿主系统指令。宿主决定返回的消息如何进入模型上下文，并将受信任策略保留在更高优先级。

在服务器边界校验提示词参数。提示词 URI 应通过直接资源读取相同的授权检查。不要把提示词变成绕过资源访问的侧信道。

## 缓存提示是正确性的一部分

`ttlMs` 告诉客户端结果可以复用的时长。`cacheScope` 描述谁可以共享该缓存值。

| 作用域 | 含义 | 典型用法 |
|---|---|---|
| `public` | 在授权允许时可跨用户复用 | 公共提示词目录 |
| `private` | 绑定到请求用户或凭据上下文 | 用户拥有的笔记内容 |

TTL 应基于数据变更频率与过时带来的损害来选择。公共提示词目录可用五分钟，私有笔记读取可用一分钟。

MCP 仅将 `public` 与 `private` 作为 `cacheScope` 取值。对于包含密钥或快速变化的结果，返回 `cacheScope: "private"` 并设置 `ttlMs: 0`，然后在宿主缓存策略中应用更严格的 no-store 规则。`no-store` 本身不是 MCP 的 `cacheScope` 取值。

缓存提示不能替代授权。缓存键必须包含所有改变可见性的请求维度，包括租户、用户、作用域、区域与分页游标。若共享缓存无法安全表达这些维度，请使用 `private` 搭配零 TTL 并在宿主层级实施 no-store 策略。

## 订阅使用客户端打开的响应流

现代订阅模式取代了原有的 `resources/subscribe` RPC 与旧的 HTTP GET 事件端点。

客户端以普通 JSON-RPC 请求发送 `subscriptions/listen`。在 Streamable HTTP 上这是一次 POST，其响应保持为 SSE 流开放。`notifications` 对象是白名单，服务器不得投递未被请求的通知类型。

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "subscriptions/listen",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    },
    "notifications": {
      "resourcesListChanged": true,
      "promptsListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

请求 ID 即订阅 ID。在任意被请求事件之前，服务器会发送 `notifications/subscriptions/acknowledged`，其过滤器仅包含服务器接受的部分。

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "notifications": {
      "resourcesListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

该流上之后每个事件都携带相同的元数据。

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "uri": "notes://note-1"
  }
}
```

通知只说明资源已变更。客户端再通过 `resources/read` 读取，仍受当前授权约束。它不假设事件中包含新文档。

多个订阅可共享一条 stdio 通道。订阅 ID 让客户端能解复用它们。在 HTTP 上，关闭响应流即取消订阅。优雅结束流的服务器会返回一条与原始请求关联的 `resultType: "complete"` 最终响应。

不要把订阅流当作协议会话。后续读取仍然是完整请求，可以到达任何健康的服务实例。

```figure
t3-primitive-sort
```

## 交互实验

使用本图从项目追踪器中对五项能力进行分类：issue 详情、创建 issue、sprint 审查模板、项目策略与关闭 issue。随后判断哪些列表可以公共缓存、哪些读取必须保持私有、哪些资源值得变更通知。

对每次分类，指明选择者：模型执行动作时用工具，宿主读取 URI 寻址内容时用资源，用户启动已准备好的消息工作流时用提示词。

## 实践实验

从仓库根目录运行模拟器：

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

按以下顺序审查输出日志：

1. 确认 `server/discover` 声明了当前修订版与两项能力。
2. 确认两个列表结果均排序且使用 `resultType: "complete"`。
3. 确认列表与读取结果携带了有意的缓存提示。
4. 将读取 URI 改为 `notes://missing`，观察 `-32602`。
5. 确认订阅确认先于资源事件出现。
6. 确认事件与优雅关闭都携带订阅 ID `5`。

Python 模型不会建立真实的 HTTP 连接。它用消息表示 SDK 必须在请求范围内置于响应流上的数据。生产环境请使用官方 SDK 处理帧结构与传输。

## 交付产物

`outputs/skill-primitive-splitter.md` 是可复用的 MCP 原语选型设计审查模板，现已补充确定性发现、缓存作用域、无效 URI 行为与现代订阅过滤器的检查点。

本课时还交付 `assets/primitive-split.svg`，一份用于离线学习的原语与订阅边界静态图。

## 验证

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

预期结果：主程序打印 JSON 输出日志，测试命令报告至少十二项通过。

## 毕业项目衔接

当你的毕业项目服务器在动作旁边暴露可寻址知识时，使用本契约。包含一次确定性目录快照、一次授权资源读取、一次提示词解析、一次无效 URI 用例，以及一条订阅输出日志。

你的证据应展示：任何列表都不依赖连接历史，且订阅事件从不授予对底层资源的访问权限。

## 练习

1. 添加 `notes://projects/{project}/notes/{id}` 资源模板并校验两个变量。
2. 为 `resources/list` 添加分页，同时保持确定性顺序。
3. 将一项资源改为 `cacheScope: "private"` 且 `ttlMs: 0`，添加宿主级 no-store 策略，并解释支撑这两项控制的威胁模型。
4. 添加提示词列表变更订阅，并证明当过滤器省略 `promptsListChanged` 时不发送事件。
5. 创建两条并发订阅，并证明每个事件携带正确的请求 ID。
6. 在读取处理器中添加授权主体，并证明缓存条目不能跨主体。

## 关键术语

- **资源：** MCP 服务器暴露的 URI 寻址内容。
- **提示词：** MCP 服务器暴露的用户可控消息模板。
- **确定性列表：** 相同请求输入下具有稳定成员与顺序的发现结果。
- **`ttlMs`：** 缓存新鲜度持续时间，单位毫秒。
- **`cacheScope`：** 缓存结果的共享边界。
- **`subscriptions/listen`：** 长生命周期请求，其响应流投递显式过滤的通知。
- **订阅 ID：** 原始 listen 请求 ID，重复出现在通知元数据中。
- **无效参数：** JSON-RPC 错误 `-32602`，用于无效或未知资源 URI。
- **不支持的协议版本：** JSON-RPC 错误 `-32022`，包含 `supported` 与 `requested` 修订版。
- **`server/discover`：** 强制性服务器方法，返回支持的修订版、能力、身份与可选缓存提示。

## 延伸阅读

- [MCP 2026-07-28 Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP 2026-07-28 Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP 2026-07-28 Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Caching](https://modelcontextprotocol.io/specification/2026-07-28/basic/utilities/caching)
