# 红队工具链 — Garak、Llama Guard、PyRIT

> 三套生产级工具构成了 2026 年的红队工具栈。Llama Guard（Meta）—— 在 14 个 MLCommons 危害类别上微调的 Llama-3.1-8B 分类器；2025 年 Llama Guard 4 是一个 12B 原生多模态分类器，从 Llama 4 Scout 剪枝而来。Garak（NVIDIA）—— 开源 LLM 漏洞扫描器，提供静态、动态和自适应探针，覆盖幻觉、数据泄露、提示注入、毒性和越狱等场景。PyRIT（Microsoft）—— 支持 Crescendo、TAP 和自定义转换器链的多轮红队战役编排工具，用于深度利用。Llama Guard 3 文档见 Meta 的"Llama 3 Herd of Models"（arXiv:2407.21783）；Llama Guard 3-1B-INT4 见 arXiv:2411.17713；Garak 探针架构见 github.com/NVIDIA/garak。这些工具是 2026 年红队研究（第 12-15 课）与部署（第 17 课+）之间的生产接口。

**类型：** 构建
**语言：** Python（stdlib，工具架构模拟器及 Llama Guard 风格分类器模拟）
**前置条件：** 第 18 阶段 · 第 12-15 课（越狱与 IPI）
**预计时间：** 约 75 分钟

## 学习目标

- 描述 Llama Guard 3/4 在安全栈中的定位：输入分类器、输出分类器，还是两者兼有。
- 列出 14 个 MLCommons 危害类别，并指出其中一个非显而易见的类别（代码解释器滥用）。
- 描述 Garak 的探针架构：探针、检测器、 harness。
- 描述 PyRIT 的多轮战役结构及其与 Garak 探针的组合方式。

## 问题

第 12-15 课介绍了攻击面。生产部署需要可重复、可扩展的评估手段。三套工具主导了 2026 年：Llama Guard（防御分类器）、Garak（扫描器）、PyRIT（战役编排器）。每套工具针对红队生命周期的不同层次。

## 概念

### Llama Guard（Meta）

Llama Guard 3 是在 Llama-3.1-8B 模型上针对 MLCommons AILuminate 14 个类别进行微调的输入/输出分类器：
- 暴力犯罪、非暴力犯罪、性相关、儿童性虐待材料（CSAM）、诽谤
- 专业建议、隐私、知识产权、大规模杀伤性武器、仇恨言论
- 自杀/自残、色情内容、选举、代码解释器滥用

支持 8 种语言。用法：置于 LLM 之前（输入审核）、之后（输出审核），或两者兼有。这两种用法的训练分布不同——Llama Guard 3 以单一模型形式发布，同时处理两种任务。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，移动端 CPU 上约 30 tokens/s）是量化边缘变体。

Llama Guard 4（2025 年 4 月）为 12B，原生多模态，从 Llama 4 Scout 剪枝而来。它用一个分类器取代了之前的 8B 文本分类器和 11B 视觉分类器，可同时处理文本和图片。

### Garak（NVIDIA）

开源漏洞扫描器。架构：
- **探针（Probes）。** 针对幻觉、数据泄露、提示注入、毒性、越狱的攻击生成器。静态探针（固定提示）、动态探针（生成式提示）、自适应探针（响应目标输出）。
- **检测器（Detectors）。** 根据预期失败模式对输出打分——毒性、泄露、已越狱。
- **Harnesses。** 管理探针-检测器对，运行战役，生成报告。

TrustyAI 将 Garak 与 Llama-Stack 防护盾（Prompt-Guard-86M 输入分类器、Llama-Guard-3-8B 输出分类器）集成，实现端到端受防护目标的评估。基于分层的评分（TBSA）取代了二元通过/失败——同一探针下，模型可能在严重级别 3 通过，而在严重级别 5 失败。

### PyRIT（Microsoft）

Python 风险识别工具包。多轮红队战役。核心围绕：
- **转换器（Converters）。** 转换种子提示——改述、编码、翻译、角色扮演。
- **编排器（Orchestrators）。** 运行战役：Crescendo（升级）、TAP（分支）、RedTeaming（自定义循环）。
- **评分（Scoring）。** LLM 作为评判或分类器作为评判。

PyRIT 是 Garak 的重型同族。Garak 运行数千次单轮探针；PyRIT 运行深度多轮战役，旨在突破特定失败模式。

### 工具栈

将 Llama Guard 部署在模型两侧。每晚运行 Garak 进行回归测试。在发布前运行 PyRIT 进行战役测试。这是 2026 年大多数生产部署的默认配置。

### 评估陷阱

- **评判者身份。** 三个工具均可使用 LLM 作为评判；评判者校准决定了报告的 ASR（第 12 课）。需在使用工具时同时指定评判者。
- **探针过时。** 随着模型被修补，Garak 探针会逐渐失效。自适应探针（PAIR 型）比静态探针老化更慢。
- **Llama Guard 对良性内容的误报率。** 早期 Llama Guard 版本对政治和 LGBTQ+ 内容过度标记；Llama Guard 3/4 的校准有所改善，但仍需针对具体部署进行校准。

### 本阶段定位

第 12-15 课是攻击家族。第 16 课是生产工具链。第 17 课（WMDP）是双重用途能力的评估。第 18 课是将这些工具封装为政策框架的前沿安全框架。

```figure
al-guard-stack
```

## 动手实践

`code/main.py` 构建了一个玩具级 Llama Guard 风格分类器（基于关键词和语义特征覆盖 14 个类别）、一个玩具级 Garak harness（探针-检测器循环）和一个 PyRIT 风格多轮转换器链。你可以针对模拟目标运行这三个工具，并观察它们不同的覆盖特征。

## 交付成果

本课产出 `outputs/skill-red-team-stack.md`。给定一个部署描述，文件需指明哪三套工具适用、各工具的配置项以及回归测试频率。

## 练习

1. 运行 `code/main.py`，比较 Llama-Guard 风格分类器在单轮攻击与多轮攻击上的检测率。

2. 实现一个新的 Garak 探针：base64 编码的有害请求。测量其被 Llama-Guard 风格分类器检测到的效果。

3. 在 PyRIT 风格转换器链中新增一个"先翻译为法语，再改述"的转换器。重新测量攻击成功率。

4. 阅读 Llama Guard 3 的危害类别列表。找出两个在合法开发者内容上可能产生高误报率的类别，并说明原因。

5. 比较 Garak 与 PyRIT 的设计原则，论证在哪些部署场景下各自是最合适的工具。

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|----------|
| Llama Guard | "那个分类器" | 在 14 个危害类别上微调的 Llama-3.1-8B/4-12B 安全分类器 |
| Garak | "那个扫描器" | NVIDIA 开源漏洞扫描器；包含探针、检测器、harness |
| PyRIT | "那个战役工具" | Microsoft 多轮红队编排器；包含转换器、编排器、评分 |
| Prompt-Guard | "小型分类器" | Meta 的 86M 参数提示注入分类器，与 Llama Guard 配对使用 |
| TBSA | "分层评分" | Garak 的基于分层的通过/失败评分，取代二元结果 |
| Converter chain | "改述 + 编码 + …" | PyRIT 用于构建多步攻击的组合原语 |
| MLCommons hazard categories | "14 个分类体系" | Llama Guard 瞄准的行业标准分类法 |

## 延伸阅读

- [Meta — Llama Guard 3（收录于 Llama 3 Herd 论文，arXiv:2407.21783）](https://arxiv.org/abs/2407.21783) — 8B 分类器
- [Meta — Llama Guard 3-1B-INT4（arXiv:2411.17713）](https://arxiv.org/abs/2411.17713) — 量化移动端分类器
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — 扫描器仓库与文档
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — 战役工具包
