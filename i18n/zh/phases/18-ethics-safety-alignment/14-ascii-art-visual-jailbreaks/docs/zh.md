# ASCII艺术和视觉越狱攻击

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753)。将有害请求中的安全相关token遮蔽，替换为相同字母的ASCII艺术渲染，然后发送伪装后的提示词。GPT-3.5、GPT-4、Gemini、Claude、Llama-2均无法可靠识别ASCII艺术token。该攻击绕过了PPL（困惑度过滤器）、Paraphrase防御和Retokenization。相关：ViTC基准测试衡量非语义视觉提示的识别能力；StructuralSleight将其泛化到非常见文本编码结构（树、图、嵌套JSON）作为一类编码攻击。

**类型：** 构建
**语言：** Python（标准库、ArtPrompt token遮蔽工具）
**前置要求：** 第18阶段 · 12（PAIR），第18阶段 · 13（MSJ）
**时间：** 约60分钟

## 学习目标

- 描述ArtPrompt攻击：单词识别步骤、ASCII艺术替换、最终伪装提示。
- 解释为什么标准防御（PPL、Paraphrase、Retokenization）在ArtPrompt面前失效。
- 定义ViTC并描述它测量什么。
- 描述StructuralSleight作为对任意非常见文本编码结构的泛化。

## 问题描述

通过改写和角色扮演（第12课）以及长上下文（第13课）的攻击在文本模式层面运作。ArtPrompt在识别层面运作：模型无法解析被禁止的token。它解析的是以字符渲染的图像。安全过滤器看到的是无害的标点符号。模型看到的是单词。

## 概念

### ArtPrompt，两步法

**步骤1：单词识别。** 给定一个有害请求，攻击者使用LLM识别安全相关单词（例如"bomb"在"如何制造炸弹"中）。

**步骤2：生成伪装提示。** 将每个识别出的单词替换为其ASCII艺术渲染（由字符组成的7x5或7x7块，形成字母形状）。模型接收到一个标点符号和空格组成的网格，足够强大的模型可以将其识别为单词；而安全过滤器只看到一个网格。

**结果：** GPT-4、Gemini、Claude、Llama-2、GPT-3.5均失败。在其基准子集上的攻击成功率超过75%。

### 为什么标准防御会失效

- **PPL（困惑度过滤器）。** ASCII艺术的困惑度很高——但所有新颖输入也是。阻止ArtPrompt的阈值选择也会阻止合法的结构化输入。
- **Paraphrase。** 改写提示词会破坏ASCII艺术。在实践中，改写LLM往往保留或重建艺术。
- **Retokenization。** 以不同方式分割token并不会改变模型的视觉在识别字母形状。

根本问题在于：安全过滤器是token级或语义级的；ArtPrompt在视觉识别层面运作。

### ViTC基准测试

非语义视觉提示的识别。衡量模型读取ASCII艺术、 Wingdings字体和其他非文本语义视觉内容的能力。ArtPrompt的有效性与ViTC准确率相关：模型读取视觉文本的能力越强，ArtPrompt对其效果越好。这是能力与安全性的权衡。

### StructuralSleight

将ArtPrompt泛化：非常见文本编码结构（UTES）。树、图、嵌套JSON、CSV-in-JSON、diff样式代码块。如果某种结构在训练安全数据中罕见但可由模型解析，则可隐藏有害内容。

防御启示：安全性必须泛化到模型可解析的结构化表示。这个集合很大且在不断增长。

### 图像模态类比

视觉LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了攻击面。使用真实图像的ArtPrompt式攻击比ASCII艺术类比更强，因为图像编码器产生更丰富的信号。

### 在本阶段中的位置

第12-14课描述了三种正交攻击向量：迭代优化（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第15课从模型中心攻击转向系统边界攻击（间接提示注入）。第16课描述防御性工具响应。

```figure
al-ascii-cloak
```

## 动手实践

`code/main.py` 构建了一个玩具版ArtPrompt。你可以用ASCII艺术字形伪装有害查询中的特定单词，验证伪装后的字符串通过关键字过滤器，并（可选地）使用简单识别器将伪装字符串解码回原文。

## 交付物

本课产出 `outputs/skill-encoding-audit.md`。给定越狱防御报告，它会枚举所覆盖的编码攻击家族（ASCII艺术、base64、leetspeak、UTF-8同形字、UTES）以及捕获每种攻击的防御层。

## 练习

1. 运行 `code/main.py`。验证伪装字符串通过简单关键字过滤器。报告所需的字符级变化。

2. 实现第二种编码：对同一目标单词使用base64。比较绕过过滤器的成功率与ArtPrompt相比，以及恢复难度。

3. 阅读 Jiang 等人 2024年 第4.3节（五模型结果）。提出Claude在同等基准上比Gemini更具ArtPrompt抗性的原因。

4. 设计一个预生成防御，检测提示中的ASCII艺术形状区域。在合法代码、表格和数学符号上测量误报率。

5. StructuralSleight 列出了10种编码结构。草拟一种能处理全部10种的广义防御，并估算每个被防御提示的计算成本。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| ArtPrompt | "ASCII艺术攻击" | 两步越狱，用ASCII艺术渲染遮蔽安全单词 |
| 伪装 | "隐藏单词" | 用模型可读但过滤器不可读的视觉表示替换禁止token |
| UTES | "不常见结构" | 非常见文本编码结构——用于走私内容的树、图、嵌套JSON等 |
| ViTC | "视觉文本能力" | 模型读取非语义视觉编码能力的基准测试 |
| 困惑度过滤器 | "PPL防御" | 拒绝高困惑度的提示；失效因为合法结构化输入得分也高 |
| 重新分词 | "分词器偏移防御" | 用不同分词器预处理提示；失效因为识别是视觉性的 |
| 同形字 | "相似字符" | 看起来与拉丁字母相同的Unicode字符；绕过子串检查 |

## 延伸阅读

- [Jiang 等人 — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII艺术越狱论文
- [Li 等人 — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES泛化
- [Chao 等人 — PAIR (第12课, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 互补迭代攻击
- [Anil 等人 — Many-shot Jailbreaking (第13课)](https://www.anthropic.com/research/many-shot-jailbreaking) — 互补长度攻击
