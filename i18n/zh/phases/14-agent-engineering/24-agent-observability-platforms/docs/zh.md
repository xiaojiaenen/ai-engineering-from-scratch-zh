# Agent 可观测性：Langfuse、Phoenix、Opik

> 2026 年，三家开源 Agent 可观测性平台占据主导地位。Langfuse (MIT) — 月安装量 600 万+，支持追踪 + 提示词管理 + 评估 + 会话回放。Arize Phoenix (Elastic 2.0) — 深度 Agent 专用评估、RAG 相关性分析、OpenInference 自动插桩。Comet Opik (Apache 2.0) — 自动提示词优化、护栏机制、LLM 裁判幻觉检测。

**类型:** 学习
**语言:** Python (stdlib)
**前置知识:** 第 14 阶段 · 第 23 章（OTel GenAI）
**耗时:** 约 45 分钟

## 学习目标

- 说出三家顶级开源 Agent 可观测性平台及其许可证。
- 区分各家擅长领域：Langfuse（提示词管理 + 会话）、Phoenix（RAG + 自动插桩）、Opik（优化 + 护栏）。
- 解释为何到 2026 年，89% 的组织已部署了 Agent 可观测性。
- 使用标准库实现追踪到仪表盘的管道，并集成 LLM 裁判评估。

## 问题

OTel GenAI（第 23 章）提供了 schema，但你仍需一个能摄入 span、运行评估、存储提示词版本并暴露回归的平台。这三大竞品各自侧重生命周期的不同环节。

## 概念

### Langfuse (MIT)

- 月 SDK 安装量 600 万+，GitHub 星数 1.9 万+。
- 功能：追踪、带版本控制和沙盒的提示词管理、评估（LLM 裁判、用户反馈、自定义）、会话回放。
- 2025 年 6 月：此前为商业模块（LLM 裁判、标注队列、提示词实验、沙盒）以 MIT 许可证开源。
- 最强场景：端到端可观测性 + 紧密的提示词管理闭环。

### Arize Phoenix (Elastic License 2.0)

- 更深入的 Agent 专用评估：追踪聚类、异常检测、RAG 检索相关性。
- 原生支持 OpenInference 自动插桩。
- 可与托管版 Arize AX 配合用于生产。
- 无提示词版本控制 — 定位为配合更广泛平台的漂移/行为回归工具。
- 最强场景：RAG 相关性、行为漂移、异常检测。

### Comet Opik (Apache 2.0)

- 通过 A/B 实验实现自动提示词优化。
- 护栏（PII 脱敏、主题约束）。
- LLM 裁判幻觉检测。
- Comet 自身发布的基准测试：Opik 日志 + 评估耗时 23.44 秒 vs Langfuse 327.15 秒（约 14 倍差距） — 供应商基准数据仅作参考。
- 最强场景：优化闭环、自动化实验、护栏执行。

### 行业数据

据 Maxim（2026 年实地分析）：89% 的组织已部署 Agent 可观测性；质量问题是最主要的生产障碍（32% 的受访者提及）。

### 如何选择

| 需求 | 选择 |
|------|------|
| 一体化 + 提示词管理 | Langfuse |
| 深度 RAG 评估 + 漂移检测 | Phoenix |
| 自动优化 + 护栏 | Opik |
| 开放许可，不含 ELv2 | Langfuse (MIT) 或 Opik (Apache 2.0) |
| Datadog / New Relic 集成 | 均可 — 均支持 OTel 导出 |

### 本模式常见陷阱

- **缺乏评估策略**。仅有追踪而无评估，只是昂贵的日志记录。
- **无外部依据的自研 LLM 裁判**。CRITIC 模式（第 05 章）适用 — 裁判需要外部工具进行事实核查。
- **提示词版本与追踪未关联**。当生产环境出现回归时，你无法二分定位导致问题的提示词。

```figure
wb-trace-ingest
```

## 动手实践

`code/main.py` 实现了基于标准库的追踪采集器 + LLM 裁判评估器：

- 摄入 GenAI 形态的 span。
- 按会话分组，标记失败运行（护栏触发、低置信度评估）。
- 基于评价标准的脚本化 LLM 裁判，对 Agent 响应进行评分。
- 类仪表盘摘要：失败率、主要失败原因、评估分数分布。

运行：

```
python3 code/main.py
```

输出：按会话的评估分数和失败分类，与 Langfuse/Phoenix/Opik 展示的内容匹配。

## 使用方式

- **Langfuse**：自托管或云端；通过 OTel 或其 SDK 接入。
- **Arize Phoenix**：自托管；自动插桩 OpenInference。
- **Comet Opik**：自托管或云端；自动优化闭环。
- **Datadog LLM 可观测性**：适合已使用 Datadog 的混合运维 + ML 团队。

## 交付物

`outputs/skill-obs-platform-wiring.md` 选择一家平台，将追踪 + 评估 + 提示词版本接入现有 Agent。

## 练习

1. 将一周的 OTel 追踪导出到 Langfuse 云端（免费层）。哪些会话失败了？为什么？
2. 为你的领域编写一份 LLM 裁判评分标准（事实正确性、语调、范围遵守）。在 50 条追踪上测试。
3. 对比 Langfuse 提示词版本控制与 Phoenix 的追踪聚类。哪个能更快定位故障？
4. 阅读 Opik 的护栏文档。为你的某次 Agent 运行接入 PII 脱敏护栏。
5. 在你的数据集上对三者进行基准测试。忽略厂商发布的数据；自行测量。

## 关键术语

| 术语 | 业内说法 | 实际含义 |
|------|---------|---------|
| Tracing | "Span 采集器" | 摄入 OTel / SDK span，按会话索引 |
| Prompt management | "提示词 CMS" | 与追踪关联的版本化提示词 |
| LLM-as-judge | "自动化评估" | 独立 LLM 根据评分标准对 Agent 输出打分 |
| Session replay | "追踪回放" | 逐步查看历史运行以调试 |
| RAG relevancy | "检索质量" | 检索到的上下文是否与查询匹配 |
| Trace clustering | "行为聚类" | 聚类相似运行以检测漂移 |
| Guardrail enforcement | "日志级策略执行" | 对已记录内容进行 PII/毒性/范围检查 |

## 延伸阅读

- [Langfuse 文档](https://langfuse.com/) — 追踪、评估、提示词管理
- [Arize Phoenix 文档](https://docs.arize.com/phoenix) — 自动插桩、漂移检测
- [Comet Opik](https://www.comet.com/site/products/opik/) — 优化 + 护栏
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 三家平台共同消费的基础 schema
