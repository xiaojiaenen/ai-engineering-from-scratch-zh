# 基于角色的 Agent 团队 —— 角色、任务、流程

> 四个原语：Agent（智能体）、Task（任务）、Crew（团队）、Process（流程）。两种顶层形态：Crews（自主的、基于角色的协作）和 Flows（事件驱动的、确定性的）。CrewAI 是 2026 年的参考实现，其文档直言不讳："对于任何生产就绪的应用，请从 Flow 开始。"

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置知识：** 第 14 阶段 · 12（工作流模式），第 14 阶段 · 14（Actor 模型）
**时间：** 约 75 分钟

## 学习目标

- 说出 CrewAI 的四个原语（Agent、Task、Crew、Process）各自负责什么。
- 区分 Sequential、Hierarchical 和计划中的 Consensus 流程；为不同负载选择合适的类型。
- 区分 Crews（自主的基于角色的协作）与 Flows（事件驱动的确定性），并解释文档的生产建议。
- 使用 `@tool` 装饰器和 `BaseTool` 子类来连接工具；思考结构化输出与自由文本的差异。
- 说出 CrewAI 的四种内存类型及其适用场景。
- 实现一个 stdlib 三 Agent Crew（研究员、作家、编辑）来生成简报。
- 识别 CrewAI 的三个失败模式：prompt 膨胀、manager LLM 开销、脆弱的交接。

## 问题所在

采用多 Agent 框架的团队总会撞到同一面墙。"自主协作"在演示中听起来很棒。然后客户提交了一个 bug，你需要确定性的重放。或者财务问每轮运行 LLM 路由的 Crew 成本是多少。或者 on-call 需要知道是哪个 Agent 在凌晨 3 点卡住了。

自由形式的 LLM 路由 Crew 无法干净地回答这些问题。纯 DAG 可以全部回答，但失去了头脑风暴 Agent 所需的探索性结构。

CrewAI 的分裂方式诚实地面对了这个权衡。Crews 用于协作的、基于角色的、探索性工作。Flows 用于事件驱动的、代码主导的、可审计的生产环境。同一个框架，两种形态，根据场景选择。

## 概念

### 四个原语

CrewAI 的表层很小。记住这些，剩下的就是配置。

- **Agent。** `role + goal + backstory + tools + (可选) llm`。backstory 是承重的。它塑造语气、判断、以及 Agent 何时停止。Tools 是 Agent 可以调用的函数（见下文）。
- **Task。** `description + expected_output + agent + (可选) context + (可选) output_pydantic`。一个可重用的工作单元。`expected_output` 是契约。`context` 列出上游任务的输出会传入的地方。`output_pydantic` 强制结构化形状。
- **Crew。** 容器。持有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory`、`verbose`、`manager_llm` 设置。
- **Process。** 执行策略。Sequential、Hierarchical、Consensus（计划中）。决定运行的形状。

Agent 之间不能直接看到对方。Task 引用 Agent。Crew 对任务进行排序。Process 决定谁选择下一个任务。这就是整个心智模型。

> **已验证于** CrewAI 0.86（2026-05）。新版本可能重命名或合并流程类型；在具体依赖某种形状之前，请查阅 [CrewAI 流程文档](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential。** 任务按声明顺序执行。任务 N 的输出作为 `context` 提供给任务 N+1。成本最低。最可预测。当顺序固定时使用。
- **Hierarchical。** 一个 Manager Agent（单独的 LLM 调用）在专家之间进行路由。CrewAI 根据你的 `manager_llm` 配置或默认值生成 Manager。Manager 每轮选择下一个任务，可以拒绝或重新路由。当你有四个或更多专家且顺序确实依赖于先前输出时使用。
- **Consensus。** 计划中，当前未在公共 API 中实现。文档保留了这个名字用于未来的基于投票的流程。今天不要依赖它。

Hierarchical 在每个专家调用之上增加了一个每轮的 LLM 调用（Manager）。在五步运行中，令牌开销可能翻三倍。仅在需要路由时才为此付费。

### Crews vs Flows

这是 2026 年文档的首要框架。

- **Crew。** LLM 驱动的自主性。框架在运行时选择形状。适用于：研究、头脑风暴、初稿、路径本身就是答案的任何地方。难以重放。难以测试。原型开发成本低。
- **Flow。** 你拥有的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记当另一个步骤发出该 topic 时触发的步骤。每个步骤都是普通 Python（可以在内部调用 Crew）。适用于：生产环境。可观测。可测试。确定性。

2026 年的文档生产建议：从 Flow 开始。当自主性证明其成本合理时，将 Crews 折叠为 Flow 步骤内的 `Crew.kickoff()` 调用。Flow 给你审计轨迹，Crew 给你探索。组合使用，不要二选一。

### 工具集成

给 Agent 提供工具有三种方式。选择最适合你的最简单的那种。

1. **`@tool` 装饰器。** 纯函数变成工具。签名是 schema；docstring 是 LLM 看到的描述。最适合一次性辅助函数。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 具有显式参数 schema、异步支持、重试的类工具。当工具有状态（客户端、缓存）或需要结构化参数时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 附带官方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。通过一次导入即可连接。

结构化输出使用 Pydantic。在 Task 上传递 `output_pydantic=MyModel`。CrewAI 根据模型验证 LLM 响应，并进行转换或重试。配合严格的 `expected_output` 字符串使用。自由文本输出适合初稿；结构化输出是下游 Flow 可以消费的内容。

### 内存钩子

CrewAI 开箱即用地提供四种内存类型。它们可以组合：Crew 可以同时启用全部四种。

> **已验证于** CrewAI 0.86（2026-05）。最近的版本将所有内容通过统一的 `Memory` 系统路由，该系统包装了这四种存储。下面的概念模型仍然成立，但公共类表层可能在较新版本中收缩为单个 `Memory` 入口点；请查看 [CrewAI 内存文档](https://docs.crewai.com/concepts/memory) 了解当前 API。

- **短期。** 单次运行内的对话缓冲区。运行结束时清除。
- **长期。** 跨运行持久化。存储在向量数据库中（默认 Chroma，可替换）。通过与当前任务的相似度检索。
- **实体。** 每个实体的事实。"客户 X 在企业套餐上。" 按实体键控，而非按相似度。跨运行存活。
- **上下文。** 组装时检索。在 Agent 需要时拉取相关内存，而非预加载。

在 Crew 上使用 `memory=True` 或按类型配置启用。由你配置的嵌入提供商支持（默认 OpenAI，可替换为本地）。内存是 CrewAI 相对于更薄框架的价值所在之一；纯 LangGraph 需要你自行连接每一种。

### 基于角色的团队适合的场景

- 三到六个具有命名角色和协作工作流的 Agent。起草、审查、规划、头脑风暴。
- LLM 对下一步的判断本身就是价值的路由（Hierarchical）。
- 团队更喜欢阅读 `role + goal + backstory` 而不是阅读图定义的任何地方。

### 不适合的场景

- 具有严格顺序的确定性 DAG。使用 LangGraph（第 13 课）。图的形状是正确的抽象；CrewAI 的角色框架是摩擦。
- 亚秒级延迟预算。Hierarchical 增加了往返次数。即使是 Sequential 也会对包含 backstory 和先前输出的 prompt 进行序列化。
- 单 Agent 循环。跳过框架；一个 Agent 循环（第 1 课）加上工具注册表更简洁。

第 17 课（Agent 框架权衡）用矩阵展示了这些内容。简短版本：CrewAI 位于"协作基于角色"的角落。

### 依赖形状

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。Star 数：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（2026-05 快照）。AWS Bedrock 集成已有文档；厂商基准测试报告相比 LangGraph 在 QA 负载上有显著加速，但方法论（数据集、硬件、评估指标）未发布，因此将框架厂商数据视为仅具方向性。

### 此模式出错的地方

- **来自 backstory 的 prompt 膨胀。** 每个 Agent 一个 2000 词的 backstory，五个 Agent 的 Crew 在第一次工具调用前就烧完了上下文预算。保持 backstory 在 200 词以内。跨 Agent 复用短语；不要重复五次相同的风格指南。
- **Manager-LLM 令牌开销。** Hierarchical 流程在每个专家调用前增加一次 Manager LLM 调用。在一个五任务 Crew 中，这是六次 LLM 调用而非五次，且 Manager 调用携带完整的任务列表加上先前输出。除非路由依赖于输出，否则切换到 Sequential。
- **脆弱的交接。** 任务 N 的 `expected_output` 是"一个大纲"。任务 N+1 将其作为 `context` 读取并尝试解析三个部分。LLM 生成了四个。下游 Agent 即兴发挥。在任务 N 上使用 `output_pydantic` 修复，使任务 N+1 读取类型化对象而非自由文本。
- **Crew 用于生产。** 未加 Flow 包装的自由形式 Crew 直接推向生产。输出变异性高；无法重放；on-call 无法将失败运行与成功运行进行对比。用 Flow 包装。

```figure
ae-crew-vs-flow
```

## 构建它

`code/main.py` 实现了两种形状的 stdlib 版本以及一个三 Agent Crew。

形状：

- `Agent`、`Task` 数据类，匹配 CrewAI 的表层。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，将输出作为 `context` 传递。
- `HierarchicalCrew.kickoff(topic)` 添加一个 Manager Agent 每轮选择下一个专家，直到"done"停止。
- `Flow` 带有 `@start` 和 `@listen(topic)` 装饰器、一个小型事件循环和追踪。
- `tool(name)` 装饰器，镜像 CrewAI 的 `@tool` 形状。
- `Memory` 带有 `short_term`、`long_term`、`entity` 存储；模拟相似度使用 numpy。
- 模拟 LLM 响应是基于角色和输入前缀的硬编码字符串。无网络。确定性。

具体示例：研究员、作家、编辑 Crew 生成关于"agent engineering 2026"的简报。研究员提取（模拟的）来源。作家起草。编辑精简。同一个 Crew 通过 Flow 运行以展示确定性形状。

运行：

```bash
python3 code/main.py
```

追踪涵盖：Sequential Crew 通过 `context` 传递输出、Hierarchical Crew 带有 Manager 选择（研究员、作家、编辑，然后"done"）、Flow 使用显式主题（`researched`、`drafted`、`edited`）运行相同的三个步骤、工具调用通过 `@tool` 路由、以及长期内存在两次 kickoff 之间存活。

Crew 追踪是流动的；Manager 原则上可以重新排序。Flow 追踪是固定的。这个选择就是课程要点。

## 使用它

- **CrewAI Flow** 用于生产。即使 Flow 只是一个调用 `Crew.kickoff()` 的步骤。Flow 给你审计边界。
- **CrewAI Crew（Sequential）** 用于明确顺序的协作工作，尤其是初稿和审查循环。
- **CrewAI Crew（Hierarchical）** 当路由依赖于输出且你有四个或更多专家时。
- **LangGraph**（第 13 课）用于显式状态机、持久恢复、严格顺序。
- **AutoGen v0.4**（第 14 课）用于 Actor 模型并发和故障隔离。
- **OpenAI Agents SDK**（第 16 课）用于具有交接和护栏的 OpenAI 优先产品。
- **Claude Agent SDK**（第 17 课）用于具有子 Agent 和会话存储的 Claude 优先产品。

## 交付

`outputs/skill-crew-or-flow.md` 为任务选择 Crew 还是 Flow 并搭建最小实现。拒绝没有 backstory 的 Crew、没有显式主题的 Flow、少于三个专家的 Hierarchical。

## 陷阱

- **Backstory 作为装饰。** 它塑造输出。为每个 Agent 测试三个变体；变异性是真实的。选择一个，冻结它。
- **跳过 `expected_output`。** 没有每个任务的契约，下游任务会拾取 LLM 生成的任何内容。Crew 运行；审计失败。
- **内存始终开启。** 长期内存每次运行都写入。向量数据库增长。检索变得嘈杂。将写入范围限定在事实持久的任务上。
- **Manager prompt 漂移。** Hierarchical 的 Manager prompt 是隐式的。如果路由变得奇怪，以 verbose 模式转储并阅读。
- **Crew 中的工具副作用。** Crew 可能比预期更频繁地调用工具。POST、DELETE、支付属于 Flow 步骤，绝不属于 Crew 工具。

## 练习

1. 将 Sequential Crew 转换为 Flow。计算变异性下降的触点数量。注意可读性下降的地方。
2. 向 Crew 添加实体内存：关于客户的事实跨 kickoff 持久化。验证检索拉取正确的实体。
3. 实现一个 Hierarchical 流程，其中 Manager 在作家的输出至少有三个段落之前拒绝路由到编辑。追踪重试过程。
4. 为（模拟的）网络搜索实现一个 `BaseTool` 子类。比较追踪形状与 `@tool` 装饰器版本。
5. 向编辑任务添加 `output_pydantic=Brief`，其中 `Brief` 具有 `title`、`summary`、`sections`。让作家任务输出一段格式错误的 JSON；验证追踪中 CrewAI 的重试行为。
6. 阅读 CrewAI 的文档简介。将玩具程序移植到真正的 `crewai` API。stdlib 版本省略了哪些保证？
7. 将 AgentOps 或 Langfuse（第 24 课）连接到真实运行。stdlib 版本中你遗漏了哪些追踪？

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|-----------|---------|
| Agent | "角色" | 角色 + 目标 + backstory + 工具 |
| Task | "工作单元" | 描述 + 预期输出 + 分配者 + 可选结构化输出 |
| Crew | "Agent 团队" | Agent + Task + Process 的容器 |
| Process | "执行策略" | Sequential / Hierarchical / Consensus（计划中） |
| Flow | "确定性工作流" | 事件驱动、代码主导、可测试 |
| Backstory | "角色 prompt" | Agent 的语气和判断塑造器 |
| `@tool` | "函数工具" | 将函数变为 Agent 可调用的工具的装饰器 |
| `BaseTool` | "类工具" | 具有参数 schema、重试、异步支持的类工具 |
| Entity memory | "每个实体的事实" | 作用域限定为客户 / 账户 / 问题的内存 |
| Long-term memory | "跨运行内存" | 向量支持的在 kickoff 之间存活的内存 |
| Contextual memory | "即时检索" | 在 Agent 需要时拉取的内存 |
| Manager LLM | "路由 Agent" | Hierarchical 流程中选择下一个任务的额外 LLM |
| `expected_output` | "任务契约" | 告诉 Agent（和审计）返回什么形状字符串 |

## 进一步阅读

- [CrewAI 文档简介](https://docs.crewai.com/en/introduction)：概念和推荐的生产路径
- [CrewAI Flows 指南](https://docs.crewai.com/en/concepts/flows)：事件驱动形状、`@start`、`@listen`
- [CrewAI 工具参考](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、内置工具包
- [CrewAI 内存](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文
- [Anthropic，构建有效的 Agent](https://www.anthropic.com/research/building-effective-agents)：何时多 Agent 有帮助，何时没有帮助
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案
