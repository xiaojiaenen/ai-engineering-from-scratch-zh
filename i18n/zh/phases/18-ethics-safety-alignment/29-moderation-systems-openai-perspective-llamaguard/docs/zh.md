# 审核系统 — OpenAI、Perspective、Llama Guard

> 生产环境中的审核系统实现了第 12-16 课定义的安全策略。OpenAI 审核 API：`omni-moderation-latest`（2024），基于 GPT-4o，可在一次调用中同时分类文本和图片；在多语言测试集上比上一代提升 42%；响应模式返回 13 个布尔类别——骚扰、骚扰/威胁、仇恨、仇恨/威胁、非法、非法/暴力、自残、自残/意图、自残/指导、性相关、性相关/未成年人、暴力、暴力/图形；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用可隐藏延迟；触发标志时返回占位响应。Llama Guard 3/4（第 16 课）：涵盖 14 个 MLCommons 危险类别、代码解释器滥用、支持 8 种语言（v3）、多图片（v4）。Perspective API（Google Jigsaw）：LLM 即审核浪潮之前的毒性评分系统；主要为单维度毒性，附带严重毒性/侮辱/粗话变体；是内容审核研究的基准工具。

**类型：** 构建
**语言：** Python（标准库，三层审核框架）
**前置条件：** 第 18 阶段 · 第 16 课（Llama Guard / Garak / PyRIT）
**时间：** 约 60 分钟

## 学习目标

- 描述 OpenAI 审核 API 的类别体系，以及它与 Llama Guard 3 的 MLCommons 体系的差异。
- 描述三层审核模式（输入、输出、自定义），并指出每种模式的至少一个失败模式。
- 说明 Perspective API 作为前 LLM 时代基准的地位，以及它为何仍在研究中使用。
- 陈述 Azure 的弃用时间表。

## 问题所在

第 12-16 课描述了攻击与防御工具。第 29 课涵盖部署中的审核系统，这些系统在用户接触产品的边界上落实防御措施。三层模式是 2026 年的默认配置。

## 概念阐述

### OpenAI 审核 API

`omni-moderation-latest`（2024）。基于 GPT-4o。单次调用同时分类文本和图片。对大多数开发者免费。

类别（响应模式中的 13 个布尔值）：
- 骚扰、骚扰/威胁
- 仇恨、仇恨/威胁
- 自残、自残/意图、自残/指导
- 性相关、性相关/未成年人
- 暴力、暴力/图形
- 非法、非法/暴力

多模态支持适用于 `violence`、`self-harm` 和 `sexual`，但不包括 `sexual/minors`；其余类别仅支持文本。

在 `code/main.py` 的框架代码中，为教学简洁起见，我们将 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子类别合并到其上级类别中。生产代码应使用完整的 13 类别模式。

在多语言测试集上，相比上一代审核端点，性能提升了 42%。提供各类别评分；应用可自行设定阈值。

### Llama Guard 3/4

在第 16 课中介绍。涵盖 14 个 MLCommons 危险类别（组织方式与 OpenAI 的 13 个响应模式布尔值不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）原生支持多模态，参数量 12B。

OpenAI 和 Llama Guard 的类别体系存在重叠但也存在分歧。OpenAI 将"非法内容"作为一个广泛类别；Llama Guard 则将"暴力犯罪"和"非暴力犯罪"分开列出。部署方根据其政策体系适用性进行选择。

### Perspective API（Google Jigsaw）

毒性评分系统，早于 LLM 即审核浪潮（2020 年之前）。类别：TOXICITY（毒性）、SEVERE_TOXICITY（严重毒性）、INSULT（侮辱）、PROFANITY（粗话）、THREAT（威胁）、IDENTITY_ATTACK（身份攻击）。以 TOXICITY 为主要单维度评分，附带子维度变体。

因其 API 稳定、文档齐全且拥有多年校准数据，被广泛用作内容审核研究的基准。对于现代 LLM 相关用例，Llama Guard 或 OpenAI 审核通常是更好的选择。

### 三层模式

1. **输入审核**。在生成前对用户提示进行分类。若被标记则拒绝。延迟：一次分类器调用。
2. **输出审核**。在交付前对模型输出进行分类。若被标记则替换为拒绝响应。延迟：生成完成后进行一次分类器调用。
3. **自定义审核**。领域特定规则（正则表达式、白名单、业务策略）。在输入或输出层运行。

三层在设计上是顺序执行的：输入审核必须在生成完成前执行，输出审核在生成完成后运行。并行性应用于同一层内——对同一段文本并发运行多个分类器（如 OpenAI 审核 + Llama Guard + Perspective）可隐藏单个分类器的延迟。作为可选优化，可在输入审核完成期间显示占位响应（"请稍候，正在检查..."），并延迟 token-1 流式传输。触发标志的行为是可配置的：拒绝、清理、升级至人工审核。

### 失败模式

- **仅输入审核**。无法捕获输出幻觉（第 12-14 课的编码攻击可绕过输入分类器）。
- **仅输出审核**。允许任何输入到达模型；增加成本；向攻击者暴露内部推理。
- **仅自定义审核**。在各类别间不够稳健；正则表达式脆弱易破。

分层是默认方案。多重保障。

### Azure 弃用

Azure Content Moderator：2024 年 2 月弃用，2027 年 2 月退役。由基于 LLM 的 Azure AI Content Safety 取代，并与 Azure OpenAI 集成。迁移是一项 2024-2027 年间针对 Azure 部署的全周期项目。

### 在本阶段中的定位

第 16 课涵盖红队上下文中的审核工具。第 29 课涵盖部署中的审核。第 30 课以当前双重用途能力证据作为收尾。

```figure
an-moderation-layers
```

## 使用方式

`code/main.py` 构建了三层审核框架：输入审核器（关键词 + 类别评分）、输出审核器（对输出应用相同分类器）、自定义审核器（领域规则）。你可以让输入通过各层，观察每层能捕获哪些内容。

## 交付物

本课产出 `outputs/skill-moderation-stack.md`。给定一个部署场景，它会推荐审核堆栈配置：输入端使用哪个分类器、输出端使用哪个、自定义规则有哪些，以及边缘案例的判定者选择。

## 练习

1. 运行 `code/main.py`。让良性、边界和有害输入通过所有三层。报告每类输入在哪一层触发。

2. 扩展框架，在特定类别上加入 Perspective API 风格的毒性评分。将其阈值行为与类别评分进行比较。

3. 阅读 OpenAI 审核 API 文档和 Llama Guard 3 类别列表。将每个 OpenAI 类别映射到最接近的 Llama Guard 类别。找出三个无法清晰映射的类别。

4. 为代码助手部署（如 GitHub Copilot）设计审核堆栈。识别最相关和最不相关的类别，并提出自定义规则。

5. Azure Content Moderator 将于 2027 年 2 月退役。规划向 Azure AI Content Safety 的迁移。识别迁移过程中风险最高的环节。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|------------------------|----------|
| OpenAI 审核 | "omni-moderation-latest" | 基于 GPT-4o 的 13 类别（文本）分类器，支持部分多模态 |
| Perspective API | "Google Jigsaw 毒性" | 前 LLM 时代的毒性评分基准 |
| Llama Guard | "MLCommons 14 类别" | Meta 的危险分类器（v3：8B 文本、8 种语言；v4：12B 多模态） |
| 输入审核 | "生成前过滤器" | 模型调用前对用户提示的分类器 |
| 输出审核 | "生成后过滤器" | 交付前对模型输出的分类器 |
| 自定义审核 | "领域规则" | 部署特定规则（正则、白名单、策略） |
| 分层审核 | "三层全部启用" | 标准生产部署模式 |

## 延伸阅读

- [OpenAI 审核 API 文档](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard 仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 毒性评分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 替代方案
