# Agent 可观测性：Langfuse、Phoenix、Opik

> 三大开源Agent可观测性平台主导2026年。Langfuse (MIT) — 每月安装量600万+，具备追踪、提示词管理、评估与会话回放功能。Arize Phoenix (Elastic 2.0) — 深入的Agent专用评估、RAG相关性分析、OpenInference自动插桩。Comet Opik (Apache 2.0) — 自动化提示词优化、防护栏、LLM-judge幻觉检测。

**类型：** 学习
**语言：** Python (标准库)
**先修课程：** 阶段 14 · 23 (OTel GenAI)
**时长：** 约45分钟

## 学习目标

- 说出三大顶级开源Agent可观测性平台及其许可证。
- 区分各自最强项：Langfuse (提示词管理 + 会话), Phoenix (RAG + 自动插桩), Opik (优化 + 防护栏)。
- 解释为何89%的组织报告称到2026年已具备Agent可观测性。
- 实现一个从标准库追踪到仪表盘的流水线，包含LLM-judge评估。

## 问题所在

OTel GenAI（第23课）提供了模式定义。你仍然需要一个平台来摄取span、运行评估、存储提示词版本并暴露回归问题。三个竞争者各自侧重于生命周期的不同部分。

## 核心概念

### Langfuse (MIT)

- 每月SDK安装量600万+，GitHub星标19k+。
- 功能：追踪、带版本控制与沙盒的提示词管理、评估（LLM-as-judge、用户反馈、自定义）、会话回放。
- 2025年6月：原商业模块（LLM-as-a-judge、注释队列、提示词实验、沙盒）以MIT协议开源。
- 最强领域：端到端可观测性与紧密的提示词管理循环。

### Arize Phoenix (Elastic License 2.0)

- 更深入的Agent专用评估：轨迹聚类、异常检测、RAG的检索相关性分析。
- 原生OpenInference自动插桩。
- 可与托管的Arize AX搭配用于生产环境。
- 无提示词版本控制——定位为与更广泛平台搭配的漂移/行为回归工具。
- 最强领域：RAG相关性、行为漂移、异常检测。

### Comet Opik (Apache 2.0)

- 通过A/B实验进行自动化提示词优化。
- 防护栏（PII编辑、主题约束）。
- LLM-judge幻觉检测。
- 来自Comet自身测量的基准测试：Opik日志与评估耗时23.44秒，而Langfuse为327.15秒（差距约14倍）——请将供应商基准视为参考。
- 最强领域：优化循环、自动化实验、防护栏执行。

### 行业数据

根据Maxim (2026年领域分析)：89%的组织已具备Agent可观测性；质量问题是首要的生产障碍（32%的受访者提及）。

### 如何选择

| 需求 | 选择 |
|------|------|
| 集提示词管理于一体的全能平台 | Langfuse |
| 深度RAG评估与漂移检测 | Phoenix |
| 自动化优化与防护栏 | Opik |
| 开放许可，无ELv2限制 | Langfuse (MIT) 或 Opik (Apache 2.0) |
| Datadog / New Relic 集成 | 任意 — 它们都导出OTel |

### 此模式常见的错误

- **缺乏评估策略。** 没有评估的追踪只是昂贵的日志记录。
- **自建LLM-judge但无依据。** CRITIC模式（第5课）适用——评判需要外部工具进行事实核查。
- **提示词版本未与追踪关联。** 当生产环境出现回归时，你无法定位到导致问题的提示词。

## 动手构建

`code/main.py` 实现了一个标准库的追踪收集器与LLM-judge评估器：

- 摄取GenAI格式的span。
- 按会话分组，标记失败运行（触发防护栏、低置信度评估）。
- 一个脚本化的LLM-judge，根据评分标准对Agent响应进行评分。
- 一个类似仪表盘的摘要：失败率、主要失败原因、评估分数分布。

运行它：

```
python3 code/main.py
```

输出：每个会话的评估分数与失败分类，与Langfuse/Phoenix/Opik显示的内容相匹配。

## 如何使用

- **Langfuse** 自托管或云；通过OTel或其SDK接入。
- **Arize Phoenix** 自托管；自动插桩OpenInference。
- **Comet Opik** 自托管或云；自动化优化循环。
- **Datadog LLM Observability** 适用于已运行Datadog的混合运维与ML团队。

## 部署实践

`outputs/skill-obs-platform-wiring.md` 选择一个平台，并将追踪、评估与提示词版本接入现有的Agent。

## 练习

1. 将一周的OTel追踪数据导出到Langfuse云（免费层）。哪些会话失败了？为什么？
2. 为你的领域编写一个LLM-judge评分标准（事实正确性、语气、范围遵守）。在50条追踪上测试。
3. 比较Langfuse的提示词版本控制与Phoenix的轨迹聚类。哪个能更快地告诉你哪里出了问题？
4. 阅读Opik的防护栏文档。将一个PII编辑防护栏接入到你的某个Agent运行中。
5. 在你的语料库上对这三者进行基准测试。忽略供应商发布的数据；测量你自己的。

## 关键术语

| 术语 | 人们常说什么 | 实际含义 |
|------|----------------|------------------------|
| Tracing | "Spans收集器" | 摄取 OTel / SDK spans；按会话索引 |
| Prompt management | "提示词CMS" | 与追踪关联的版本化提示词 |
| LLM-as-judge | "自动评估" | 单独的LLM根据评分标准对Agent输出进行评分 |
| Session replay | "轨迹回放" | 逐步回放过去的运行用于调试 |
| RAG relevancy | "检索质量" | 检索到的上下文是否与查询匹配 |
| Trace clustering | "行为分组" | 聚类相似运行以检测漂移 |
| Guardrail enforcement | "日志时策略" | 对记录的内容进行PII/毒性/范围检查 |

## 延伸阅读

- [Langfuse 文档](https://langfuse.com/) — 追踪、评估、提示词管理
- [Arize Phoenix 文档](https://docs.arize.com/phoenix) — 自动插桩、漂移
- [Comet Opik](https://www.comet.com/site/products/opik/) — 优化 + 防护栏
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 三者都遵循的模式