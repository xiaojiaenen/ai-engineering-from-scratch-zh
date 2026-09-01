# FIPA-ACL 与言语行为的遗产

> 在 MCP 和 A2A 之前，有 FIPA-ACL。2000 年，IEEE 智能物理代理基金会认证了一种包含二十种行事语（performatives）、两种内容语言以及一组交互协议（契约网、订阅/通知、request-when）的代理通信语言。它之所以在工业界式微，是因为本体论开销对 Web 来说太重；但 LLM 驱动的多代理系统复兴正在悄然重新实现相同的思想，只是去掉了形式化语义：JSON 合约取代了行事语，自然语言取代了本体论。本课认真研读 FIPA-ACL，让你看清 2026 年的哪些协议决策是重蹈覆辙、哪些是真正创新，以及当前浪潮将在何处重新发现 2000 年代已经解决的问题。

**类型：** 学习
**语言：** Python（标准库）
**前置要求：** 阶段 16 · 01（为什么做多代理）
**时间：** 约 60 分钟

## 问题

2026 年的代理协议格局十分热闹：MCP 做工具、A2A 做代理、ACP 做企业审计、ANP 做去中心化信任、NLIP 做自然语言内容，再加上 CA-MCP 和二十多个研究提案。每个规范都自称是基础性的。

诚实的看法是：其中大多数只是在重新发现一棵非常古老的二十年代龄决策树。Austin（1962）和 Searle（1969）的言语行为理论给了我们"话语即行动"。KQML（1993）将其转化为网络协议。FIPA-ACL（2000 年认证）产生了参考标准：二十种行事语、SL0/SL1 内容语言、用于契约网和订阅-通知的交互协议。JADE 和 JACK 是 Java 参考平台。这套工作大约在 2010 年式微，因为本体论开销太重，而 Web 赢得了胜利。

当你审视 MCP 的 `tools/call`、A2A 的任务生命周期或 CA-MCP 的共享上下文存储时，你看到的是一种更温和的、JSON 原生的 FIPA 决策翻版。了解这段历史能告诉你两件事：哪些新的"创新"实际上是重新发明，哪些旧有的失败模式将被新规范重新发现。

## 概念

### 言语行为，一段话讲完

Austin 注意到有些句子不是在描述世界——它们改变世界。"我承诺。""我请求。""我宣布。"他称之为行事语。Searle 将其归纳为五类：断言类、指令类、承诺类、表情类、宣告类。KQML（Finin 等人，1993）让这对软件代理变得可操作：消息 = 行事语（行动类型）+ 内容（行动的客体）。FIPA-ACL 修补了 KQML 的缺陷，围绕二十种行事语进行了标准化。

### FIPA 二十种行事语（部分列表）

| 行事语 | 意图 |
|---|---|
| `inform` | "我告诉你 P 为真" |
| `request` | "我请求你做 X" |
| `query-if` | "P 是否为真？" |
| `query-ref` | "X 的值是什么？" |
| `propose` | "我建议我们做 X" |
| `accept-proposal` | "我接受提议" |
| `reject-proproposal` | "我拒绝该提议" |
| `agree` | "我同意做 X" |
| `refuse` | "我拒绝做 X" |
| `confirm` | "我确认 P 为真" |
| `disconfirm` | "我否认 P" |
| `not-understood` | "你的消息解析失败" |
| `cfp` | "征集关于 X 的提案" |
| `subscribe` | "当 X 变化时通知我" |
| `cancel` | "取消进行中的 X" |
| `failure` | "我尝试了 X 但失败了" |

完整列表见 `fipa00037.pdf`（FIPA ACL 消息结构）。要点不是背诵它——而是每一个行事语都对应 LLM 协议最终会重新引入的一个原语。

### 典型 FIPA-ACL 消息

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段构成协议信封；一个字段（`content`）承载负载。其余字段正是你每次给 JSON 协议外挂重试、线程和本体论时重新发明的那些东西。

### 两个遗留平台

**JADE**（Java Agent DEvelopment 框架，1999–2020 年代）是最常用的 FIPA 兼容运行时。代理继承基类，交换 ACL 消息，在容器内运行，并使用"行为"进行协调。交互协议库附带了契约网、订阅通知、request-when 和 propose-accept。

**JACK**（Agent Oriented Software，商业产品）在 FIPA 消息之上强调 BDI（信念-欲望-意图）推理。形式化程度更高，采用面更广。

一旦 Web 技术栈吞噬了多代理用例，两者都开始衰退。MCP 和 A2A 就是 2026 年的运行时"容器"。

### 为什么 FIPA 式微

- **本体论开销。** FIPA 需要共享本体才能解析 `content`。达成本体共识是一个历时多年的标准化过程。Web 直接使用了 HTTP + JSON。
- **没人用的形式语义。** SL（语义语言）提供了严谨的真值条件，但大多数生产系统使用自由格式内容，忽略形式化体系。
- **工具链锁定。** JADE 仅支持 Java；JACK 是商业产品。多语言团队绕开两者。
- **互联网赢下了传输层。** REST，然后 JSON-RPC，然后 gRPC 取代了 ACL 的传输。

### LLM 复兴是 FIPA 轻量版

将 FIPA 的 `request` 与 MCP 的 `tools/call` 对比：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

相同的信封，不同的语法。两者都携带：谁、向谁、意图、负载、相关 ID。它们彼此之间都不是革命——它们是在同一设计上的不同取舍。

Liu 等人 2025 年的调查（"A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP"，arXiv:2505.02279）明确指出了这条家谱：MCP 对应工具使用言语行为，A2A 对应代理对等言语行为，ACP 对应审计追踪言语行为，ANP 对应去中心化身份扩展。新规范是带有 JSON 语法和更宽松语义的 ACL 后代。

### 取舍，直白地说

**FIPA 给你而现代规范放弃的：**

- 形式语义——你可以证明 `inform` 意味着发送方相信该内容。
- 标准化的行事语目录——你无需重新争论"是否应该有 `cancel`"。
- 数十年的交互协议模式——契约网、订阅通知、提议-接受，以及已知的正确性属性。

**现代规范给你而 FIPA 没有的：**

- 与现代工具兼容的 JSON 原生负载。
- LLM 无需手编本体即可理解的自然语言内容。
- Web 栈传输（HTTP、SSE、WebSocket）。
- 通过实时 MCP `server/discover` 和 A2A 代理卡进行的能力发现。

用更宽松的意图语义换取更低的实现门槛。这就是确切的取舍。

### 值得移植的交互协议

FIPA 附带约 15 种交互协议。三种值得带入 LLM 多代理系统：

1. **契约网协议（CNP）。** 管理器发出 `cfp`（征集提案）；投标者回复 `propose`；管理器接受/拒绝。这是典型的任务市场模式（阶段 16 · 16 谈判）。
2. **订阅/通知。** 订阅者发送 `subscribe`；发布者在主题变化时发送 `inform`。这是 2026 年的每一个事件总线。
3. **Request-When。** "当条件 Y 成立时做 X。"带前置条件的延迟动作。2026 年的对应物是可维持工作流引擎中的延迟任务（阶段 16 · 22 生产扩展）。

它们都能干净地映射到现代消息队列、HTTP + 轮询或 SSE 流式传输。

### 去掉本体论后哪里会出问题

没有共享本体时，代理从自然语言内容中推断含义。2026 年记录在案的去向模式是**语义漂移**：两个代理用同一个词（`"customer"`）表示微妙不同的概念，接收方代理基于错误的理解行动，没有模式验证器能捕获它。FIPA 的本体要求会在解析阶段就拒绝该消息。

不走完整本体论的缓解措施：

- `content` 上使用 JSON Schema——在传输层拒绝结构性错误。
- 类型化工件（A2A）——拒绝错误模态。
- 在信封中显式行事语——即使内容是自然语言也能使意图无歧义。

### 2026 年规范映射到言语行为遗产

| 现代规范 | FIPA 对应物 | 保留内容 | 放弃内容 |
|---|---|---|---|
| MCP `tools/call` | `request` | 显式意图、相关 ID | 形式语义、本体论 |
| MCP `resources/read` | `query-ref` | 显式意图、相关 ID | 形式语义 |
| A2A 任务生命周期 | contract-net + request-when | 异步生命周期、状态转换 | 形式完备性保证 |
| A2A 流式事件 | subscribe/notify | 异步推送 | 类型谓词订阅 |
| CA-MCP 共享上下文 | 黑板（Hayes-Roth 1985） | 多写者共享内存 | 逻辑一致性模型 |
| NLIP | 自然语言内容 | LLM 原生 | 模式 |

从上到下阅读表格，模式是：保留结构原语，放弃形式化体系，让 LLM 弥补歧义。

```figure
sw-contract-net
```

## 动手实现

`code/main.py` 实现了一个纯标准库的 FIPA-ACL 翻译器。它编码和解码典型的 ACL 信封，并展示了每个 MCP / A2A 消息形状如何归约为相同的七个字段。演示：

- 将五种 MCP 风格和 A2A 风格消息编码为 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等价格式。
- 使用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 在一个管理器和三个投标者之间运行示例契约网协商。

运行：

```
python3 code/main.py
```

输出是一个并排追踪，展示每个现代消息同时以 2026 JSON 格式和 FIPA-ACL 格式呈现，然后是契约网投标的往返过程。相同的协议原语在往返中存活下来；只有语法不同。

## 使用方式

`outputs/skill-fipa-mapper.md` 是一个技能，读取任意代理协议规范并生成 FIPA-ACL 映射。在采用新协议之前使用它来回答："这真正是新的东西，还是只是带 JSON 语法的 `inform`？"

## 交付

不要重新引入 FIPA-ACL。带回来的是它的清单：

- 每条消息的意图原语（行事语）是什么？
- 是否存在用于请求-响应和取消的相关 ID？
- 是否有显式的内容语言（JSON-RPC、纯文本、结构化类型化工件）？
- 交互协议是一等公民，还是你在从零重新实现契约网？
- 当两个代理对内容含义产生分歧时（语义漂移），会发生什么？

在任何新协议投入生产之前，记录这五个问题。

## 练习

1. 运行 `code/main.py`。观察往返编码过程。找出与 `tools/call`、`resources/read` 和 A2A 任务创建对应的 FIPA 行事语。
2. 用 `cancel` 行事语扩展契约网演示，允许管理器在投标中途撤回任务。`cancel` 解决了重试单独无法解决的哪种故障场景？
3. 阅读 FIPA ACL 消息结构（http://www.fipa.org/specs/fipa00037/）第 4.1–4.3 节。挑选一个本课未涵盖的行事语，描述其现代 JSON-RPC 对应物。
4. 阅读 Liu 等人，arXiv:2505.02279。针对 MCP、A2A、ACP、ANP 每一个，列出它们保留和放弃的 FIPA 行事语家族。
5. 为自己系统中的 `request` 行事语的 `content` 字段设计一个最小 JSON Schema。这个模式为你带来了纯自然语言所不具备的什么能力，又付出了什么代价？

## 关键术语

| 术语 | 人们怎么说 | 实际上是什么意思 |
|------|----------------|------------------------|
| 言语行为 | "一种有行动效果的话语" | Austin/Searle：话语即行动。ACL 的理论祖先。 |
| FIPA | "那个旧的 XML 东西" | 智能物理代理 IEEE 基金会。2000 年标准化了 ACL。 |
| ACL | "代理通信语言" | FIPA 的信封格式：行事语 + 内容 + 元数据。 |
| 行事语 | "动词" | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | "FIPA 的前身" | 知识查询与操作语言（1993）。更简单、更窄。 |
| 本体论 | "共享词汇表" | 对内容语言所讨论的概念的形式化定义。 |
| SL0 / SL1 | "FIPA 内容语言" | 语义语言第 0 层和第 1 层——形式化内容语言家族。 |
| 契约网 | "任务市场" | 管理器发出 cfp；投标者提案；管理器接受。经典交互协议。 |
| 交互协议 | "消息序列模式" | 具有已知正确性的行事语序列：request-when、subscribe-notify 等。 |

## 延伸阅读

- [Liu 等人 — 《代理互操作协议调查：MCP、ACP、A2A、ANP》](https://arxiv.org/html/2505.02279v1) —— 将现代规范与 FIPA 遗产联系起来的 2025 年标准调查
- [FIPA ACL 消息结构规范（fipa00037）](http://www.fipa.org/specs/fipa00037/) —— 2000 年认证的信封格式
- [FIPA 交际行为库规范（fipa00037）](http://www.fipa.org/specs/fipa00037/) —— 完整行事语目录
- [MCP 规范 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) —— `request`/`query-ref` 的当前无状态工具使用等价物
- [A2A 规范](https://a2a-protocol.org/latest/specification/) —— 契约网和订阅通知的现代代理对等等价物
