# 生产运行时：队列、事件、定时任务

> 生产级智能体运行于六种运行时形态：请求-响应、流式、持久化执行、基于队列的后台、事件驱动和调度。先选形态，再选框架。可观测性在任何形态下都是关键负载。

**类型：** 学习
**语言：** Python (stdlib)
**前置知识：** 第 14 阶段 · 13（LangGraph），第 14 阶段 · 22（语音）
**耗时：** 约 60 分钟

## 学习目标

- 说出六种生产运行时形态，并将其与对应的框架/产品模式匹配。
- 解释持久化执行（LangGraph）为何对长周期任务至关重要。
- 描述事件驱动运行时，以及何时适合使用 Claude Managed Agents。
- 阐述"可观测性作为关键负载"这一主张在多步智能体中的含义。

## 问题所在

生产级智能体的失败方式是 Jupyter Notebook 无法揭示的：第 37 步的网络超时、用户在语音通话中途挂断、定时任务在服务器重启后死亡、后台工作者内存溢出。运行时形态决定了哪些失败是可以恢复的。

## 概念讲解

### 请求-响应

- 同步 HTTP。用户等待完成。
- 仅适用于短任务（<30 秒）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准 HTTP 访问日志 + OTel spans。

### 流式

- SSE 或 WebSocket 实现渐进式输出。
- LiveKit 将其扩展到 WebRTC 用于语音/视频（第 22 课）。
- 技术栈：任何支持流式的框架 + 能处理 SSE/WS 的前端。
- 可观测性：每个 chunk 的时序、首 token 延迟、尾部延迟。

### 持久化执行

- 每个步骤后保存状态检查点；失败时自动恢复。
- AutoGen v0.4 的 actor 模型将故障隔离到单个智能体（第 14 课）。
- LangGraph 的核心差异化特性（第 13 课）。
- 当步骤数未知且恢复成本高昂时必不可少。

### 基于队列 / 后台

- 任务进入队列，工作者领取，结果通过 webhook 或 pub/sub 回传。
- 对长周期智能体至关重要（Anthropic 的 computer use 公告指出每个任务数十到数百步）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自研方案。
- 可观测性：队列深度、单任务延迟分布、DLQ 大小。

### 事件驱动

- 智能体订阅各类触发器：新邮件、PR 打开、定时触发。
- Claude Managed Agents 开箱即用地支持（第 17 课）。
- CrewAI Flows（第 15 课）构建了事件驱动的确定性工作流。
- 可观测性：触发源、事件到启动的延迟、智能体延迟。

### 定时任务

- 以 Cron 形态运行的智能体，定期执行。
- 与持久化执行结合，使失败任务的夜间运行可在下一个周期恢复。
- 技术栈：Kubernetes CronJob + 持久化框架；托管方案（Render cron、Vercel cron）。

### 2026 年部署模式

- **CrewAI Flows** 用于事件驱动生产场景。
- **Agno** 无状态 FastAPI 用于 Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）用于嵌入集成。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（第 22 课）。
- **Claude Managed Agents** 用于托管长周期异步任务。

### 可观测性是核心负载

没有 OpenTelemetry GenAI spans（第 23 课）配合 Langfuse/Phoenix/Opik 后端（第 24 课），你无法调试在第 40 步失败的多步智能体。这对生产环境不是可选项。它是"我们能快速调试"和"我们只能带更多日志从头重放"之间的区别。

### 生产运行时失败的场景

- **选错形态。** 用请求-响应去做一个 5 分钟的任务。用户挂断；工作者堆积；重试雪崩。
- **无 DLQ。** 没有死信队列的后台工作者。失败的任务静默消失。
- **后台工作不透明。** 后台智能体运行但不导出追踪。直到用户报障前，失败完全不可见。
- **跳过持久化状态。** 任何超过 30 秒且不能承受重启的任务都需要持久化执行。

```figure
wb-runtime-shapes
```

## 动手实践

`code/main.py` 是一个 stdlib 多形态演示程序：

- 请求-响应端点（普通函数）。
- 流式处理器（生成器）。
- 带 DLQ 的基于队列的工作者。
- 事件触发注册表。
- 定时任务调度器。

运行它：

```bash
python3 code/main.py
```

输出五条追踪，展示同一任务在各形态下的行为。相同的智能体逻辑，不同的外壳。持久化执行（第六种形态）故意留到第 13 课配合 LangGraph 检查点机制讲解。

## 使用指南

- **请求-响应** 用于类聊天 UX。
- **流式** 用于渐进式响应。
- **持久化** 用于长周期任务。
- **队列** 用于批处理 / 异步 / 长时运行。
- **事件驱动** 用于智能体响应性。
- **定时** 用于运维工作（内存整合、评估、成本报表）。

## 上线交付

`outputs/skill-runtime-shape.md` 为某个任务选择运行时形态，并配置好可观测性要求。

## 练习

1. 将你第 01 课的 ReAct 循环移植到你的技术栈中的全部六种形态。哪种形态对应哪个产品形态？
2. 为基于队列的演示添加 DLQ。模拟 10% 的任务失败；暴露 DLQ 大小。
3. 编写一个 cron 触发的评估智能体，每日夜间对你的当天 Top 20 追踪进行评估。
4. 实现带背压的流式传输：如果客户端处理缓慢，暂停智能体。这如何与回合预算交互？
5. 阅读 Claude Managed Agents 文档。什么情况下你应该将自托管的长周期智能体迁移到托管方案？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 请求-响应 | "同步" | 用户等待；仅适用于短任务 |
| 流式 | "SSE / WS" | 渐进式输出；更好的 UX；按 chunk 可观测延迟 |
| 持久化执行 | "从失败恢复" | 状态检查点；从最后一步重启 |
| 基于队列 | "后台任务" | 生产者 / 工作者池 / DLQ |
| 事件驱动 | "触发式" | 智能体对外部事件作出反应 |
| DLQ | "死信队列" | 失败任务的停车场 |
| Claude Managed Agents | "托管方案" | Anthropic 托管的长周期异步，带缓存和压缩 |

## 延伸阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 持久化执行详解
- [Claude Managed Agents 概述](https://platform.claude.com/docs/en/managed-agents/overview) — 托管长周期异步
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "每个任务数十到数百步"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor 模型故障隔离
