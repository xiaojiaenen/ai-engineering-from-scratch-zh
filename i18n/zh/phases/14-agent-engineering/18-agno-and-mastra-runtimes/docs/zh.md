# 生产级 Agent 运行时——快速实例化与类型化工作流

> 生产级 agent 运行时优化了原型框架所忽略的方面：实例化成本、类型化的工作流接口以及可直接部署的后端。2026 年的搭配：Agno（Python）专注于微秒级 agent 实例化和无状态 FastAPI 后端。Mastra 在 Vercel AI SDK 基础上提供 agent、工具、工作流、统一模型路由和组合存储。

**类型：** 学习
**语言：** Python、TypeScript
**前置知识：** Phase 14 · 01（Agent 循环）、Phase 14 · 13（LangGraph）
**耗时：** 约 45 分钟

## 学习目标

- 识别 Agno 的性能目标及其适用场景。
- 列举 Mastra 的三个核心原语——Agents、Tools、Workflows——以及支持的服务器适配器。
- 解释为何无状态会话级 FastAPI 后端是推荐的 Agno 生产路径。
- 根据技术栈（Python 优先 vs TypeScript 优先）选择合适的方案（Agno 或 Mastra）。

## 问题所在

LangGraph、AutoGen、CrewAI 等框架较为沉重。希望“在我的运行时中快速实现纯粹的 agent 循环”的团队会转向 Agno（Python）或 Mastra（TypeScript）。两者都以牺牲部分框架原生原语为代价，换取更高的运行速度和与周围技术栈更紧密的契合度。

## 核心概念

### Agno

- Python 运行时，原名 Phi-data。
- “没有图、链或复杂的模式——只有纯粹的 Python。”
- 官方文档中的性能指标：约 2μs agent 实例化时间、每个 agent 约占用 3.75 KiB 内存、支持约 23 个模型提供商。
- 生产路径：无状态会话级 FastAPI 后端。每个请求启动一个全新的 agent；会话状态存储在数据库中。
- 原生支持多模态（文本、图像、音频、视频、文件）及 agentic RAG。

当你每秒需要处理数千个短生命周期 agent 时（如聊天并发接入、评估流水线），这些速度指标很重要。而当单个 agent 运行长达 10 分钟时，这些指标的重要性则较低。

### Mastra

- TypeScript 编写，基于 Vercel AI SDK 构建。
- 三个核心原语：**Agents**、**Tools**（使用 Zod 类型定义）、**Workflows**。
- 统一模型路由器——跨 94 个提供商接入 3,300+ 个模型（2026 年 3 月数据）。
- 组合存储：将内存、工作流、可观测性数据分别路由到不同的后端；大规模可观测性推荐搭配 ClickHouse。
- Apache 2.0 许可证，但 `ee/` 目录受开源可用（source-available）的企业许可证约束。
- 提供 Express、Hono、Fastify、Koa 的服务器适配器；原生支持 Next.js 和 Astro 集成。
- 附带 Mastra Studio（本地调试端口 localhost:4111）。
- 1.0 版本发布时（2026 年 1 月）已拥有 22k+ GitHub Star 和 300k+ 的周 npm 下载量。

### 定位对比

两者都不是 LangGraph 的替代品。它们的竞争维度在于：

- **语言适配。** Agno 适合 Python 优先的团队；Mastra 适合 TypeScript 优先的团队。
- **运行时体验。** Agno = 近乎零开销；Mastra = 深度集成 Vercel 生态。
- **可观测性。** 两者均集成 Langfuse/Phoenix/Opik（第 24 课），但 Mastra Studio 为官方自研。

### 选型建议

- **Agno** —— Python 后端，大量短生命周期 agent，性能要求严格，FastAPI 技术栈。
- **Mastra** —— TypeScript 后端，Next.js/Vercel 部署，统一多提供商模型路由，Zod 类型化工具。
- **LangGraph**（第 13 课）—— 当持久化状态和显式图推理比纯速度更重要时。
- **OpenAI / Claude Agent SDK** —— 当你需要采用提供商内置的产品化形态时（见第 16-17 课）。

### 常见误区

- **为性能而性能。** 仅因“2μs”听起来很酷就选用 Agno，但若工作负载是每次请求仅调用一次慢速 agent，则开销根本不是瓶颈。
- **生态锁定。** Mastra 对 Vercel 的原生适配在 Vercel 上是优势，在其他平台上反而是劣势。
- **企业许可证混淆。** Mastra 的 `ee/` 目录采用开源可用（source-available）协议，而非 Apache 2.0。若有分叉计划，请务必仔细阅读许可证。

```figure
wb-runtime-spawn
```

## 动手实践

本课主要为对比性质——任何单一代码示例都无法完全公平地展示两个框架。请参考 `code/main.py` 中的对照玩具示例：一个最小的“运行 agent、流式输出、持久化会话”流程被分别用两种方式实现（一次 Agno 风格，一次 Mastra 风格）。

运行方式：

```
python3 code/main.py
```

两种结构不同但功能等价的可观测轨迹。

## 用法

- **Agno** —— 需要高性能且采用 FastAPI 架构的 Python 后端。
- **Mastra** —— 接入多个模型提供商并需要工作流原语的 TypeScript 后端。
- 两者均提供官方可观测性钩子，且均可与 Langfuse 集成。

## 产出

`outputs/skill-runtime-picker.md` 将根据技术栈、延迟预算和运行形态，为 Agno、Mastra、LangGraph 或提供商 SDK 提供选型指南。

## 练习

1. 阅读 Agno 官方文档。将标准库 ReAct 循环（第 01 课）移植到 Agno。哪些部分消失了？哪些部分保留了？
2. 阅读 Mastra 官方文档。将同一个循环移植到 Mastra。工具的类型定义（Zod vs 无类型）发生了哪些变化？
3. 基准测试：在你的技术栈上测量 agent 实例化延迟。Agno 的 2μs 对你的工作负载有意义吗？
4. 设计迁移方案：如果你一直在 Python 中使用 CrewAI，迁移到 Agno 会有哪些地方不兼容？
5. 阅读 Mastra 的 `ee/` 目录许可证条款。哪些限制会影响开源分叉？

## 核心术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agno | "Fast Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | Single client for 3,300+ models across 94 providers |
| Composite storage | "Multiple backends" | Memory/workflows/observability each to a different store |
| Mastra Studio | "Local debugger" | localhost:4111 UI for introspecting agents |
| Source-available | "Not OSS" | License permits source reading but restricts commercial use |

## 延伸阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — performance targets, FastAPI integration
- [Mastra docs](https://mastra.ai/docs) — primitives, server adapters, Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — the stateful-graph alternative
- [Comet Opik](https://www.comet.com/site/products/opik/) — observability comparisons cited by Mastra integrations
