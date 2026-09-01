# 长期运行的后台代理：持久化执行

> 生产环境的长周期代理不会在 `while True` 中运行。每次 LLM 调用都会变成一个带检查点、重试和重放的活动。Temporal 的 OpenAI Agents SDK 集成已于 2026 年 3 月 GA。Claude Code Routines（Anthropic）无需持久本地进程即可运行计划的 Claude Code 调用。会话在人机交互时暂停，部署后存活，并从最新的按 `thread_id` 键控的检查点恢复。在这些新特性背后，是一个古老的模式——工作流编排——加上一个新增输入：LLM 调用作为必须在恢复时确定性重放的非确定性活动。

**类型：** 学习
**语言：** Python（标准库，最小化持久执行状态机）
**前置知识：** 阶段 15 · 10（权限模式），阶段 15 · 01（长周期代理）
**时间：** 约 60 分钟

## 问题所在

考虑一个运行四小时的代理。它调用三个工具，提示用户两次，并执行四十次 LLM 调用。中途，运行它的宿主机重启了。会发生什么？

- 在朴素的 `while True` 循环中：一切都会丢失。运行从头重新开始。三次工具调用（带有真实副作用）会再次执行。用户会被再次提示已经批准过的事情。四十次 LLM 调用会被重新计费。
- 使用持久化执行：运行从最近的检查点恢复。已完成的 activity 不会重新执行；它们的结果会从持久日志中重放。用户不会重新批准已批准过的事情。已执行的 LLM 调用不会被重新计费。

这与工作流引擎过去十年提供相同的模式（Temporal、Cadence、Uber 的 Cherami）。新事物在于，LLM 调用现在成为一种 activity——非确定性、昂贵、有副作用——并且可以干净地适配这个模式。

本课程的运行主题：长周期可靠性会衰减（METR 观察到"35 分钟退化"——成功率随时间大致呈二次方下降）。持久化执行使得运行时间超过可靠性曲线支持的范围成为可能，这是在设计正确时的安全失败方式，或设计错误时的不安全失败方式。

## 概念

### Activity、工作流与重放

- **工作流（Workflow）**：确定性编排代码。定义 activity 的顺序、分支、等待。必须是确定性的，以便能够从事件日志中重放而不出现令人意外的分歧。
- **Activity**：一个非确定性、可能失败的工作单元。LLM 调用、工具调用、文件写入、HTTP 请求。每个 activity 都会被记录其输入，并在完成后记录其输出。
- **事件日志**：持久化的支撑存储。每次 activity 启动、完成、失败、重试，以及每次工作流决策都会被记录。
- **重放（Replay）**：在恢复时，工作流代码从头重新运行；所有已完成的活动返回其日志结果而不会重新执行。只有尚未完成的活动才会实际运行。

这与 React 针对虚拟 DOM 重新渲染，或 Git 从提交重建工作树是相同的形状。编排器的确定性是持久化成本低廉的原因。

### 为什么 LLM 调用适配此模式

LLM 调用是：
- 非确定性的（temperature > 0；即使 temperature 为 0 也会因模型版本漂移）。
- 昂贵的（金钱和延迟）。
- 可能失败的（速率限制、超时）。
- 有副作用的（如果调用工具）。

这正好符合 activity 的特征。将每次 LLM 调用包装为 activity，可以获得带指数退避的重试、跨重启的检查点，以及用于调试的可重放追踪。

### 按 `thread_id` 键控的检查点

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 和 Claude Code Routines 都收敛到了相同的 API 形态：`thread_id`（或等价物）标识会话；每个状态转换持久化到后端（默认 PostgreSQL，开发用 SQLite，缓存用 Redis）；恢复时读取最新检查点。

后端选择很重要：

- **PostgreSQL**：持久、可查询、部署存活。LangGraph 的默认选项。
- **SQLite**：仅本地开发；跨主机丢失数据。
- **Redis**：快速但易失，除非配置 AOF/快照。
- **Cloudflare Durable Objects**：透明分布式；按唯一键作用域；持续数小时到数周。

### 人机交互作为一等公民状态

提议-提交模式（第 15 课）需要持久的"等待人工介入"状态。工作流暂停，外部队列持有待处理请求，批准从精确的那一点恢复。没有持久性这只是尽力而为；有了它，隔夜的批准到达后，工作流在早上继续执行。

### 35 分钟退化

METR 观察到，每种被测量的代理类都在连续运行约 35 分钟后显示出可靠性衰减。任务持续时间翻倍，失败率大约翻四倍。持久化执行并不能解决这个问题；它只是让你能够在可靠性曲线支持的范围内运行更长时间。安全的模式是将持久性与要求重新进入时进行新的 HITL 的检查点相结合，并与预算杀开关（第 13 课）相结合，无论墙钟时间如何都能限制总计算量。

### 持久化执行不适用的场景

- 少于几分钟且无人工介入的运行。开销 > 收益。
- 严格只读的检索任务。
- 正确性要求在单个上下文窗口内完成端到端的任务（某些推理任务；某些一次性生成任务）。

```figure
memory-consolidation
```

## 使用它

`code/main.py` 在标准库 Python 中实现了一个最小化的持久化执行引擎。它支持：

- `@activity` 装饰器，将输入和输出记录到 JSON 事件日志。
- 编排 activity 序列的工作流函数。
- 一个 `run_or_replay(workflow, event_log)` 函数，在不重新执行的情况下重放已完成的活动。

驱动程序模拟了一个三 activity 工作流，中途崩溃，并展示（a）朴素重试重新执行所有内容，与（b）重放只运行缺失的 activity。

## 部署它

`outputs/skill-durable-execution-review.md` 审查了一个拟议的长周期代理部署是否正确具有持久化执行形态：activity、确定性、检查点后端、人机交互状态，以及恢复时的 HITL 策略。

## 练习

1. 运行 `code/main.py`。观察朴素重试与重放之间的 activity 执行次数差异。更改崩溃点，并显示重放次数相应变化。

2. 将玩具引擎转换为显式使用 `thread_id`。模拟两个并发会话共享该引擎，并确认它们的事件日志不会冲突。

3. 选取玩具引擎中的一个 activity。引入非确定性（工作流决策中的墙钟时间戳）。演示重放时的分歧。解释真实引擎如何处理这种情况（副作用注册、`Workflow.now()` API）。

4. 阅读 LangChain 的"生产级深度代理背后的运行时"文章。列出运行时持久化的每个状态，并命名每个状态覆盖的故障模式。

5. 为 6 小时自主编码任务设计检查点策略。在哪里检查点？崩溃恢复是什么样子？什么需要新的 HITL？

## 关键术语

| 术语 | 人们说的话 | 实际含义 |
|---|---|---|
| Workflow | "代理的脚本" | 确定性编排代码；可从事件日志重放 |
| Activity | "一个步骤" | 非确定性单元（LLM 调用、工具调用）；前后记录 |
| Event log | "支撑存储" | 每次状态转换的持久记录 |
| Replay | "恢复" | 重新运行工作流；已完成 activity 返回日志结果而不重新执行 |
| Checkpoint | "保存点" | 按 thread_id 键控的持久状态；恢复时取最新值 |
| thread_id | "会话密钥" | 作用域持久化标识符 |
| 35-minute degradation | "可靠性衰减" | METR：成功率随时间大致呈二次方下降 |
| Non-determinism | "重放漂移" | 墙钟时间、随机数、LLM 输出；必须注册为副作用 |

## 延伸阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 预算、轮次和恢复语义。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent 形态。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具体运行时需求。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — LLM 调用的 activity 形态。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 35 分钟退化参考。
