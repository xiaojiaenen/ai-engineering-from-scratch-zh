# Batch APIs — 50% 折扣成为行业标准

> 每家主流供应商都提供了异步 Batch API，享受 50% 折扣和约 24 小时的交付周期。OpenAI、Anthropic、Google 以及大多数推理平台（Fireworks batch tier、Together batch）都实现了相同的模式。将 Batch 与 prompt 缓存和夜间流水线结合使用，成本可降至同步未缓存方案的 ~10%。规则极其简单：只要不是交互式场景，就应该走 Batch。内容生成流水线、文档分类、数据提取、报告生成、批量标注、目录标签化——任何能容忍 24 小时延迟的任务，在迁移到 Batch 之前都在白白浪费预算。2026 年的生产模式是将每个新 LLM 工作负载分为三条轨道：交互式（带缓存的同步调用）、半交互式（带回退的异步队列）、Batch（夜间运行，缓存输入堆叠）。那些假装是交互式但实际能容忍几分钟延迟的工作负载，浪费最严重。

**类型：** 学习
**语言：** Python（标准库，用于对比 batch vs sync 成本的玩具模拟器）
**前置知识：** 第 17 阶段 · 第 14 课（Prompt 与语义缓存）
**时间：** 约 45 分钟

## 学习目标

- 说出三家供应商的 Batch API（OpenAI、Anthropic、Google）以及通用的 50% 折扣 + 24 小时交付承诺。
- 计算在夜间分类工作负载上叠加 batch + 缓存输入的成本，并与同步未缓存基线进行对比。
- 将工作负载划分为交互式 / 半交互式 / Batch，并说明理由。
- 说出两个陷阱：部分交互性（用户期望快于 24 小时）和输出模式漂移（不同供应商的 Batch 文件格式不同）。

## 问题背景

你的团队维护一个夜间报告生成流水线。50,000 份文档，每份 summarize，聚类摘要，撰写高管简报。同步运行时，耗时 4 小时，每晚费用 $2,000。你听说有 Batch API。

Batch 给你 50% 折扣。你同时为系统 prompt（跨所有 50k 次调用共享）启用 prompt 缓存。叠加之后，账单降至 $180/晚 —— 约为基线的 9%。同一个流水线，三个配置变更。

Batch 是 LLM 成本工具包中最便宜的杠杆，却鲜有人使用。主要原因在组织层面：团队想到的是"实时"，而实际的 SLA 是"次日早上"。本课的目的就是让你不要把 90% 的账单白白浪费掉。

## 概念

### 三个 Batch API

**OpenAI Batch API**：JSONL 文件上传，包含一个请求列表。承诺 24 小时内返回（实际通常 ~2-8 小时）。输入和输出 token 均享受 50% 折扣。端点为 `/v1/batches`。符合缓存条件的输入还可额外获得缓存输入定价。

**Anthropic Message Batches**：JSONL 上传。24 小时交付。50% 折扣。支持 `cache_control` —— 缓存写入是显式的，读取在 Batch 内自动发生。

**Google Vertex AI Batch Prediction**：支持 BigQuery 或 GCS 输入。Gemini 同样享受 50% 折扣。可与 Vertex pipelines 集成。

### 语义：异步，而非缓慢

Batch 是"我保证在 24 小时内返回"—— 而非"这需要 24 小时"。典型 P50 为 2-6 小时。供应商在你的 Batch 排入 GPU 资源利用率较低的错峰时段。

### 与缓存叠加

50,000 份文档 summarization，使用相同的 4K-token 系统 prompt：

- 同步未缓存：50000 × ($input × 4000 + $output × 200)，按全价计费。
- 同步缓存：首次写入后系统 prompt 被缓存；剩余 49999 次获得 10 倍更便宜的输入价格。
- Batch 缓存：以上全部叠加，读写均享受 50% 折扣。

叠加效果：batch + cache ≈ 同步未缓存账单的 ~10%。任何夜间运行且有共享系统 prompt 的工作负载都应使用此方案。

### 工作负载分流

**交互式** —— 用户在等待响应。TTFT 很关键。同步调用 + prompt 缓存。无法使用 Batch。

**半交互式** —— 用户提交任务，几分钟后回来查看结果。异步队列，配合 Batch 不可用时回退到同步的方案。例如中等规模的 RAG 索引。

**Batch** —— 用户期望"次日早上"或"一小时内"拿到结果。内容流水线、大规模分类、离线分析。永远用 Batch，永远叠加缓存。

常见错误：因为流水线属于"生产环境"就将一切归类为交互式。生产环境不是延迟规格——SLA 才是。

### 部分交互性陷阱

某些功能看起来是交互式的，但实际能容忍 5-10 分钟延迟。例如：带有"刷新"按钮的夜间客户健康报告。用户点击刷新，等 10 分钟没问题。团队却以同步方式实现。50 个并发刷新产生的费用是"Batch 处理并通过邮件发送"的 10 倍。

需要问的问题是："24 小时对这个用户意味着什么？" 如果答案是"他们根本不会察觉"，那就走 Batch。

### 输出模式陷阱

不同供应商的 Batch 文件格式各不相同：

- OpenAI：JSONL，每行一个请求。
- Anthropic：JSONL，每行一个消息；响应格式内嵌。
- Vertex：BigQuery 表或带 TFRecord 的 GCS 前缀。

编写一个"跨供应商通用 Batch 客户端"意味着每个供应商都需要适配器代码。宣称支持多供应商 Batch 的网关（Portkey、LiteLLM 部分付费档位）也仍是薄薄封装了原始格式。

### 需要记住的数字

- 各供应商 Batch 折扣：输入 + 输出统一 50%。
- 交付 SLA：24 小时保底，典型 P50 为 2-6 小时。
- 叠加 Batch + 缓存输入：约为同步未缓存成本的 ~10%。
- 工作负载分流规则：如果能接受 24 小时延迟，永远选 Batch。

```figure
batch-lane-triage
```

## 使用它

`code/main.py` 为一个 50,000 文档工作负载计算同步、同步+缓存、Batch、Batch+缓存四种场景的成本，并以美元和百分比报告节省金额。

## 实践产出

本课产出 `outputs/skill-batch-triager.md`。根据工作负载特征，将其划分为交互式 / 半交互式 / Batch，并估算节省金额。

## 练习

1. 运行 `code/main.py`。对于一个 10 万文档流水线，系统 prompt 3K-token，输出 500-token，计算完整叠加方案（Batch + 缓存）相对于同步基线的节省金额。
2. 在你熟悉的某个真实产品中挑选三个功能。将每个功能划分为交互式 / 半交互式 / Batch。
3. 一个用户抱怨他们的报告等了 3 个小时。这是 Batch 分流错误还是合法的交互式需求？写出判定标准。
4. 你的 Batch API 返回 SLA 是 24 小时，但 P99 是 20 小时。如何向用户沟通这一点——边缘情况下下游系统应该如何行为？
5. 计算盈亏平衡点：共享前缀长度为多少时，Batch + 缓存会比在自己的预留 GPU 上夜间运行的成本更低？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| Batch API | "异步折扣" | 50% 折扣 + 24 小时交付 |
| JSONL | "Batch 格式" | 每行一个 JSON 请求；OpenAI/Anthropic 标准 |
| Message Batches | "Anthropic Batch" | Anthropic 的 Batch API 产品名称 |
| Batch prediction | "Vertex Batch" | Vertex AI 的 Batch API 产品 |
| Turnaround SLA | "24 小时承诺" | 保底承诺，非典型值；典型值为 2-6 小时 |
| Workload triage | "交互性决策" | 交互式 / 半交互式 / Batch 分流决策 |
| Output schema | "响应格式" | 各供应商 JSONL 布局；不可移植 |
| Stacked discount | "Batch + 缓存" | 两者同时适用时 ≈ 未缓存同步账单的 ~10% |

## 延伸阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) —— JSONL 格式与 `/v1/batches` 语义。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) —— Batch 格式与 `cache_control` 交互方式。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini) —— Gemini Batch 语义。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
