# A/B 测试 LLM 特性 — GrowthBook、Statsig 与"感觉良好"陷阱

> 传统 A/B 测试并非为确定性较低的 LLM 而设计。关键区别：评估（evals）回答"模型能否完成任务？"A/B 测试回答"用户是否在意？"两者缺一不可；仅凭"感觉良好"就上线的时代已经结束。2026 年需要测试什么：提示工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs 开源模型；精度 vs 成本 vs 延迟）、生成参数（temperature、top-p）。真实案例：聊天机器人奖励模型变体带来 +70% 对话长度和 +30% 留存率；Nextdoor AI 标题实验在奖励函数优化后带来 +1% CTR；Khan Academy Khanmigo 在延迟与数学精度之间迭代优化。平台对比：**Statsig**（2025 年 9 月被 OpenAI 以 11 亿美元收购）— 序贯测试、CUPED、一站式方案。**GrowthBook** — 开源、仓库原生、支持贝叶斯+频率学派+序贯引擎、CUPED、SRM 检查、Benjamini-Hochberg + Bonferroni 校正。根据你的仓库 SQL 偏好以及组织对"被 OpenAI 收购"的态度来选择。

**类型：** 学习
**语言：** Python（标准库，序贯测试模拟器示例）
**前置知识：** Phase 17 · 13（可观测性）、Phase 17 · 20（渐进式部署）
**时间：** 约 60 分钟

## 学习目标

- 区分评估（evals，"模型能否完成任务"）与 A/B 测试（"用户是否在意"）
- 列举三个可测试维度（提示、模型、参数）并为每个选择指标
- 解释 CUPED、序贯测试和 Benjamini-Hochberg 多重比较校正
- 根据仓库 SQL 偏好和企业对收购的态度选择 Statsig 或 GrowthBook

## 问题所在

你手工调优了系统提示词，感觉更好了，然后上线了。转化率的变化只是噪声。你责怪指标。或者你上线了新模型但转化率没动——是模型退化还是变化太小无法检测？你不知道，因为你没有经过 A/B 测试就上线了。

评估回答模型能否在标注数据集上完成任务。它们不回答用户是否更喜欢输出。只有受控的在线实验才能回答这个问题，而且实验必须具有足够的统计功效、控制不确定性因素，并校正多重比较。

## 核心概念

### 评估 vs A/B 测试

**评估（Evals）** — 离线、标注数据集、评判者（评分标准、LLM 作为评判者或人类）。回答："输出在这个固定分布上是否正确/有用/安全？"

**A/B 测试** — 在线、真实用户、随机化。回答："新变体是否推动了用户层面的关键指标？"

两者都需要。评估在暴露前捕获回归问题；A/B 测试在上线后确认产品影响。

### 测试什么

1. **提示工程** — 措辞、系统提示结构、示例。指标：任务成功率、用户留存率、每次请求成本。
2. **模型选择** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：准确率（任务）+ 每次请求成本 + P99 延迟。多目标优化。
3. **生成参数** — temperature、top-p、max_tokens。指标：任务特定指标（输出多样性 vs 确定性）。

### CUPED — 方差缩减

Controlled Experiments Using Pre-Experiment Data（使用实验前数据的对照实验）。在比较实验期数据前先对预实验期方差进行回归消除。典型方差缩减幅度：30%-70%。有效样本量免费提升。

实现方式：Statsig 和 GrowthBook 均已支持。

### 序贯测试

经典 A/B 测试假设固定样本量。序贯测试（"多次观察决策"）在重复查看时控制假阳性率。始终有效的序贯程序（mSPRT、Howard 置信序列）允许你在明显胜出时提前停止。

### 多重比较校正

同时进行 20 次 95% 置信度的 A/B 测试，理论上会因随机性产生一次假阳性。Bonferroni 校正收紧每次测试的 α；Benjamini-Hochberg 控制错误发现率。GrowthBook 均实现了这两种方法。

### SRM — 样本比例不匹配

分配哈希将用户随机分配到各变体。如果 50/50 的分组实际呈现 47/53，说明有问题——SRM 检查会标记它。两个平台均支持此功能。

### Statsig vs GrowthBook

**Statsig：**
- 2025 年 9 月被 OpenAI 以 11 亿美元收购。托管 SaaS 服务。
- 支持序贯测试、CUPED、独立人群。
- 一站式方案：功能开关 + 实验 + 可观测性。
- 最适合：希望获得捆绑产品、不关心 OpenAI 归属的团队。

**GrowthBook：**
- 开源（MIT 协议）；仓库原生（直接从 Snowflake/BigQuery/Redshift 读取数据）。
- 多种引擎：贝叶斯、频率学派、序贯。
- 支持 CUPED、SRM、Bonferroni、BH 校正。
- 可自托管或托管云。
- 最适合：使用仓库 SQL 的团队、数据团队控制指标层、希望使用开源方案。

### 不确定性增加功效计算的复杂度

同一提示会产生不同输出。传统功效计算假设 IID（独立同分布）观测。在 LLM 不确定性下，有效样本量低于名义样本量。将所需样本量乘以约 1.3-1.5 倍作为安全边际。

### 真实案例结果

- 聊天机器人奖励模型变体：+70% 对话长度，+30% 留存率。
- Nextdoor 标题：奖励函数优化后 +1% CTR。
- Khan Academy Khanmigo：在延迟与数学精度之间迭代权衡。

### 反模式：凭"感觉良好"上线

每位资深工程师都能举出一个因"感觉更好"而上线、却没有经过 A/B 测试的功能。其中大多数最终导致产品指标下降，而团队数个月后才意识到。A/B 测试是强制性的把关机制。

### 你应该记住的数据

- Statsig 被 OpenAI 收购：11 亿美元，2025 年 9 月。
- GrowthBook：开源 MIT 协议；贝叶斯 + 频率学派 + 序贯。
- CUPED 方差缩减：30%-70%。
- LLM 不确定性 → +30%-50% 样本量缓冲。

```figure
mx-sequential-test
```

## 动手实践

`code/main.py` 模拟序贯 A/B 测试，包含固定边界和序贯边界。展示序贯测试如何允许提前停止。

## 上线交付

本课将生成 `outputs/skill-ab-plan.md`。给定功能变更、工作负载、基线值，选择平台、设置门禁、确定样本量。

## 练习

1. 运行 `code/main.py`。对于预期 5% 提升、基线 3% 转化率，80% 功效需要多少样本量？
2. 为受医疗监管的本地部署客户选择 Statsig 或 GrowthBook。
3. 设计一个 A/B 测试，比较 GPT-4 vs GPT-3.5 在"每次解决工单的成本"上的表现。主要指标、守卫指标、次要指标分别是什么？
4. 你的金丝雀发布通过了但 A/B 测试显示 -1.2% 转化率。是否上线？写出升级标准。
5. 对预实验期方差占实验期 60% 的情况应用 CUPED。计算有效样本量的提升幅度。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| 评估（Eval） | "离线测试" | 基于标注数据集的模型能力评估 |
| A/B 测试 | "实验" | 在真实用户上的在线随机化对比 |
| CUPED | "方差缩减" | 对预实验期数据进行回归以降低方差 |
| 序贯测试 | "可提前查看的测试" | 始终有效的程序，允许提前停止 |
| 多重比较 | "家族误差" | 多次运行测试会膨胀假阳性率 |
| Bonferroni | "严格校正" | 将 α 除以测试次数 |
| Benjamini-Hochberg | "BH FDR" | 错误发现率控制，不如 Bonferroni 保守 |
| SRM | "糟糕的分组" | 样本比例不匹配；分配逻辑存在 bug |
| Statsig | "OpenAI 旗下产品" | 商业一站式方案，2025 年被收购 |
| GrowthBook | "开源那个" | MIT 协议的仓库原生平台 |
| mSPRT | "序贯概率比检验" | 经典序贯程序 |

## 延伸阅读

- [GrowthBook — 如何对 AI 进行 A/B 测试](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — 超越提示：数据驱动的 LLM 优化](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook 对比](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng 等人 — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — 置信序列](https://arxiv.org/abs/1810.08240)
