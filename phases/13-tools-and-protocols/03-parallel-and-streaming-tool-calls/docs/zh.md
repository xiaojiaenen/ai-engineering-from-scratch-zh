# 并行工具调用与工具流式处理

> 三个独立的天气查询序列化执行相当于三次往返。将它们并行运行，总时间将缩短到最慢的单个调用的时长。现在，每个前沿的提供商都能在单次回合中发出多个工具调用。回报是实在的；但其中的实现细节很微妙。本课将涵盖两个部分：并行扇出和流式参数的重组，重点强调ID相关性陷阱。

**类型：** 构建
**语言：** Python（标准库，线程池 + 流式处理框架）
**前置要求：** 阶段13 · 02（函数调用深入探讨）
**时间：** 约75分钟

## 学习目标

- 解释 `parallel_tool_calls: true` 为何存在以及何时应禁用它。
- 在并行扇出期间，将流式传输的参数块关联到正确的工具调用ID。
- 在不提前解析的情况下，将部分 `arguments` 字符串重组为完整的JSON。
- 运行一个三城市天气基准测试，演示顺序执行与并行执行的延迟差异。

## 问题所在

如果没有并行调用，一个回答“班加罗尔、东京和苏黎世的天气是什么”的智能体将执行以下操作：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次LLM往返，每次往返还需要承担执行器的延迟。大约是理想墙上时间的4倍。

使用并行调用：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

一次LLM往返。执行器时间是三者中的最大值，而不是总和。在OpenAI、Anthropic和Gemini上进行的生产基准测试表明，在扇出工作负载上，墙上时间减少了60%到70%。

代价是相关性复杂性。当三个调用以乱序完成时，您的结果必须携带匹配的 `tool_call_id`，以便模型可以对它们进行对齐。当结果流式传输时，您必须在执行之前将部分参数片段组装成完整的JSON。Gemini 3添加了唯一ID，部分是为了解决一个现实世界的问题：对同一工具的两次并行调用无法区分。

## 核心概念

### 启用并行

- **OpenAI.** 默认开启 `parallel_tool_calls: true`。设置 `false` 以强制序列化。
- **Anthropic.** 通过 `disable_parallel_tool_use: false` 实现并行（Claude 3.5及更高版本默认开启）。设置 `true` 以实现序列化。
- **Gemini.** 始终具备并行能力；`tool_config.function_calling_config.mode = "AUTO"` 让模型自行决定。

当工具存在顺序依赖关系时（`create_file` 然后 `write_file`），当一个调用的输出是另一个调用的输入，或者当速率限制器无法处理扇出时，请禁用并行。

### ID相关性

模型发出的每个调用都有一个 `id`。主机返回的每个结果都必须包含相同的ID。没有这个，结果就会产生歧义。

- **OpenAI.** 每个工具角色消息上的 `tool_call_id`。
- **Anthropic.** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini.** 每个 `functionResponse` 上的 `id`（Gemini 3及更高版本；Gemini 2通过名称匹配，这在同名并行调用时会失效）。

### 并发运行调用

主机在各自的线程、协程或远程工作者上运行每个调用的执行器。最简单的框架使用线程池；生产环境使用带有 `asyncio.gather` 的 asyncio 或结构化并发。完成顺序是不可预测的——ID是标识符。

一个常见的错误是：按照调用列表顺序而不是完成顺序回复结果。这通常有效，因为模型只关心 `tool_call_id`，但如果结果丢失或重复，乱序提交会使调试变得更加困难。建议按完成顺序回复并使用显式ID。

### 流式工具调用

当模型进行流式传输时，`arguments` 会分片到达。三个并行调用的三个独立数据块流在网络上交错传输。你需要为每个ID维护一个累加器。

各提供商的形状：

- **OpenAI.** 每个数据块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。数据块携带 `index`（在调用列表中的位置）。你按索引累加，当首次出现时读取 `id`，并在 `finish_reason = "tool_calls"` 时解析JSON。
- **Anthropic.** 流事件是 `message_start`，然后每个块有一个 `content_block_start`，类型为 `tool_use`（包含id、名称、空输入）。`content_block_delta` 事件携带 `input_json_delta` 块。`content_block_stop` 关闭每个块。
- **Gemini.** `streamFunctionCallArguments`（Gemini 3及更高版本）发出带有一个 `functionCallId` 的数据块，因此调用可以清晰地交错。在Gemini 3之前，流式传输一次返回一个完整的调用。

### 部分JSON与提前解析的陷阱

你不能在 `arguments` 完成之前解析它。像 `{"city": "Beng` 这样的部分JSON是无效的，会引发错误。正确的触发条件是提供商的调用结束信号：OpenAI的 `finish_reason = "tool_calls"`、Anthropic的 `content_block_stop`，或Gemini的流结束事件。只有到那时才能尝试 `json.loads`。更健壮的方法是使用增量JSON解析器，当结构完成时产生事件；OpenAI的流式处理指南建议这样做以实现显示实时“思考”指示器的用户体验。大括号计数作为完整性测试是不可靠的（引号字符串或转义内容中的大括号会导致误报），应仅用作非正式的调试启发式方法。

### 乱序完成

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

主机回复仍然必须引用这些ID：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

在OpenAI或Anthropic上，回复中的顺序对于正确性无关紧要。Gemini接受任何顺序，只要ID匹配。

### 基准测试：顺序 vs 并行

`code/main.py` 中的框架模拟了三个执行器，延迟分别为400、600和800毫秒。顺序运行总共需要1800毫秒。并行运行需要 max(400, 600, 800) = 800 毫秒。差异是常数，而非比例性的，因此节省的时间随工具数量增加而增长。

现实世界的注意事项：并行调会给下游API带来压力。对速率限制的服务进行10路扇出将会失败。阶段13 · 17涵盖了网关级别的背压；重试语义计划在未来某个阶段实现。

### 流式扇出的墙上时间

如果模型本身进行流式传输，你可以在一个调用的参数完成时就开始执行，而不是等待所有调用最终确定。这是OpenAI文档中记录的一种优化，但并非所有SDK都暴露了这一功能。本课中的框架就实现了这一点：一旦模拟流产生一个完整的参数对象，主机就会启动该调用。

## 使用它

`code/main.py` 分为两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 顺序和并行运行三个模拟的天气调用，并打印墙上时间。第二部分重放一个伪造的流式响应——三个并行调用的 `arguments` 数据块在一个流上交错传输——并使用 `StreamAccumulator` 按ID重新组装它们。没有LLM，没有网络，只有重组逻辑。

需要观察的地方：

- 顺序计时器耗时1.8秒。并行计时器在相同的伪造延迟下耗时0.8秒。
- 累加器通过按ID缓冲来处理乱序到达的数据块，并且只有在每个调用的JSON完成时才进行解析。
- 一旦某个ID的参数最终确定，执行器就会启动，而不是等到所有流结束。

## 交付它

本课产出 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该技能会审计哪些工具可以安全地并行化，哪些存在顺序依赖关系，以及哪些会超出下游速率限制——返回一个修订后的注册表，其中包含每工具 `parallel_safe` 标志。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行与顺序的比率大约为 `max/sum`（实际运行会因线程调度、序列化和框架开销而略有偏离理想值）。在何种延迟分布下，并行变得不再重要？

2. 扩展累加器以处理“调用在流式传输中途被取消”的情况，通过丢弃其缓冲区并发出一个 `cancelled` 事件。哪个提供商明确记录了这种情况？检查Anthropic的 `content_block_stop` 语义和OpenAI的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者进行基准测试。你应该会看到异步方式有微小的性能提升，因为上下文切换成本较低，但前提是执行器执行真正的I/O操作。

4. 选择两个不应该并行化的工具（例如 `create_file` 然后 `write_file`）。向注册表添加一个 `ordering_dependency` 图，并根据该图控制并行扇出。这是实现依赖关系感知调度的最小机制，未来的智能体工程阶段将对此进行形式化。

5. 阅读OpenAI的并行函数调用章节和Anthropic的 `disable_parallel_tool_use` 文档。找出一种现实世界的工具类型，Anthropic建议禁用其并行性。（提示：对同一资源的有影响的修改。）

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| 并行工具调用 | “一次回合中的扇出” | 模型在单个助手消息中发出多个工具调用 |
| `parallel_tool_calls` | “OpenAI的标志” | 启用或禁用多调用发出 |
| `disable_parallel_tool_use` | “Anthropic的反向标志” | 选择退出标志；默认为并行启用 |
| 工具调用ID | “相关性句柄” | 结果消息必须回显的每调用标识符 |
| 累加器 | “流缓冲区” | 用于部分 `arguments` 块的每ID字符串缓冲区 |
| 乱序完成 | “最快的先到” | 并行调用以不可预测的顺序完成；ID是粘合剂 |
| 依赖图 | “顺序约束” | 其输出作为其他工具输入的工具；无法并行化 |
| 提前解析陷阱 | “JSON.parse崩溃” | 尝试解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | “Gemini 3功能” | 带有每调用唯一ID的流式参数块 |
| 完成顺序回复 | “不要等待全部” | 结果到达即回复，以ID为键 |

## 延伸阅读

- [OpenAI — 并行函数调用](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为和选择退出标志
- [Anthropic — 工具使用：实现工具使用](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 和结果批处理
- [Google — Gemini 函数调用并行部分](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3的ID相关并行调用
- [OpenAI — 带工具的流式响应](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI流的分块参数重组
- [Anthropic — 流式消息](https://docs.anthropic.com/en/api/messages-streaming) — 带有 `input_json_delta` 的 `content_block_delta`