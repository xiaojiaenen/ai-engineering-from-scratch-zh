# Agno 与 Mastra：生产运行时

> Agno (Python) 和 Mastra (TypeScript) 是 2026 年的生产运行时组合。Agno 旨在实现微秒级智能体实例化和无状态的 FastAPI 后端。Mastra 基于 Vercel AI SDK 基础设施，提供智能体、工具、工作流、统一模型路由和复合存储。

**类型：** 学习
**语言：** Python, TypeScript
**前置要求：** 阶段 14 · 01 (智能体循环), 阶段 14 · 13 (LangGraph)
**时间：** ~45 分钟

## 学习目标

- 识别 Agno 的性能目标及其重要场景。
- 列举 Mastra 的三个基本构建块——智能体、工具、工作流——以及支持的服务器适配器。
- 解释为何无状态的会话作用域 FastAPI 后端是推荐的 Agno 生产部署路径。
- 根据给定技术栈（Python 优先 vs TypeScript 优先）选择 Agno 或 Mastra。

## 问题背景

LangGraph、AutoGen、CrewAI 框架较重。那些想要“只要智能体循环、要快、在我自己的运行时里”的团队会选择 Agno (Python) 或 Mastra (TypeScript)。两者都牺牲了一些框架内置的高级抽象，以换取原始速度和与技术栈更紧密的契合度。

## 核心概念

### Agno

- Python 运行时，前身是 Phi-data。
- “无图、无链、无复杂模式——只有纯粹的 python。”
- 其文档中的性能目标：约 2μs 智能体实例化时间，每个智能体约 3.75 KiB 内存，支持约 23 个模型提供商。
- 生产路径：无状态的会话作用域 FastAPI 后端。每个请求启动一个全新的智能体；会话状态存储在数据库中。
- 原生多模态支持（文本、图像、音频、视频、文件）和智能体 RAG。

当你每秒需要处理数千个短生命周期智能体（聊天扇入、评估流水线）时，这些速度目标很重要。当单个智能体运行 10 分钟时，它们就不太重要了。

### Mastra

- TypeScript 构建，基于 Vercel AI SDK。
- 三个基本构建块：**智能体**、**工具** (Zod 类型化)、**工作流**。
- 统一路由模型 — 截至 2026 年 3 月，支持 94 个提供商的 3,300+ 个模型。
- 复合存储：内存、工作流、可观测性可对接不同后端；推荐在规模可观测性方面使用 ClickHouse。
- 基于 Apache 2.0 许可证，但 `ee/` 目录在源代码可用的企业许可下提供。
- 支持 Express、Hono、Fastify、Koa 的服务器适配器；对 Next.js 和 Astro 有一流的集成支持。
- 附带 Mastra Studio (localhost:4111) 用于调试。
- 截至 1.0 版本（2026 年 1 月），GitHub 星标超过 22k，每周 npm 下载量超过 300k。

### 定位

两者都不试图成为 LangGraph。它们在以下方面竞争：

- **语言契合度。** Agno 面向 Python 优先的团队；Mastra 面向 TypeScript 优先。
- **运行时体验。** Agno = 几乎零开销；Mastra = 与 Vercel 生态系统深度集成。
- **可观测性。** 两者都与 Langfuse/Phoenix/Opik 集成（课程 24），但 Mastra Studio 是第一方工具。

### 何时选择哪个

- **Agno** — Python 后端，大量短生命周期智能体，强性能要求，FastAPI 技术栈。
- **Mastra** — TypeScript 后端，Next.js / Vercel 部署，统一的多提供商模型路由，Zod 类型化的工具。
- **LangGraph** (课程 13) — 当持久化状态和显式图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK** — 当你想要提供商产品化后的形态时（课程 16-17）。

### 此模式可能出错的地方

- **为性能而性能。** 当工作负载是每个请求仅一次慢智能体调用时，因为 “2μs” 听起来不错而选择 Agno。开销并非瓶颈。
- **生态系统锁定。** Mastra 与 Vercel 风格的集成在 Vercel 上是加分项，在其他地方则可能成为减分项。
- **企业许可混淆。** Mastra 的 `ee/` 目录是源代码可用，而非 Apache 2.0。如果你计划分叉，请务必阅读许可条款。

## 动手构建

本课程主要是对比性的——没有单一的代码实现能同时公正地展示这两个框架。请参阅 `code/main.py` 获取一个并排示例：一个实现了两次（一次 Agno 形态，一次 Mastra 形态）的最小化“运行一个智能体，流式输出，持久化会话”流程。

运行它：

```
python3 code/main.py
```

两个结构不同但功能等价的追踪。

## 实际使用

- **Agno** — 需要速度和 FastAPI 形态的 Python 后端。
- **Mastra** — 拥有众多提供商和工作流基本构建块的 TypeScript 后端。
- 两者都提供第一方可观测性钩子。两者都与 Langfuse 集成。

## 交付部署

`outputs/skill-runtime-picker.md` 根据技术栈、延迟预算和运维形态，选择 Agno、Mastra、LangGraph 或提供商 SDK。

## 练习题

1.  阅读 Agno 的文档。将标准库的 ReAct 循环（课程 01）移植到 Agno。哪些东西消失了？哪些保留了下来？
2.  阅读 Mastra 的文档。将同一个循环移植到 Mastra。工具类型化（Zod vs 无）方面有什么变化？
3.  基准测试：在你的技术栈上测量智能体实例化延迟。Agno 的 2μs 对你的工作负载重要吗？
4.  设计迁移：如果你一直在 Python 中运行 CrewAI，迁移到 Agno 会导致什么中断？
5.  阅读 Mastra 的 `ee/` 许可条款。哪些限制会影响开源分叉？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Agno | “快速的 Python 智能体” | 无状态的会话作用域智能体运行时 |
| Mastra | “Vercel AI SDK 上的 TypeScript 智能体” | 智能体 + 工具 + 工作流 + 模型路由 |
| 统一模型路由 | “多提供商访问” | 跨 94 个提供商访问 3,300+ 个模型的单一客户端 |
| 复合存储 | “多后端” | 内存/工作流/可观测性各自连接不同的存储 |
| Mastra Studio | “本地调试器” | 用于自检智能体的 localhost:4111 UI |
| 源代码可用 | “非开源” | 许可证允许查看源代码，但限制商业使用 |

## 延伸阅读

- [Agno 智能体框架文档](https://www.agno.com/agent-framework) — 性能目标、FastAPI 集成
- [Mastra 文档](https://mastra.ai/docs) — 基本构建块、服务器适配器、模型路由
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图的替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 集成引用的可观测性对比