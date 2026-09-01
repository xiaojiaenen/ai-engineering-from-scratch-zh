# Capstone 15 — 宪法安全护栏 + 红队测试范围

> Anthropic 的宪法分类器、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 内容安全以及用于多语言覆盖的 X-Guard 定义了 2026 年安全分类器技术栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准的对抗性评估工具。NeMo Guardrails v0.12 将它们集成到生产管道中。本终极项目将所有组件串联起来：围绕目标应用的分层安全护栏、运行 6+ 攻击家族的自主红队代理，以及产生可衡量无害化改进的宪法自我批判流程。

**类型：** 终极项目（Capstone）
**语言：** Python（安全管道、红队），YAML（策略配置）
**前置要求：** 阶段 10（从零构建 LLM）、阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（代理）、阶段 18（伦理、安全、对齐）
**涉及的阶段：** P10 · P11 · P13 · P14 · P18
**预计时间：** 25 小时

## 问题

2026 年 LLM 安全的 Frontier 不在于分类器是否有效（它们大致有效），而在于如何在生产应用周围正确编排它们，同时避免过度拒绝或留下明显的漏洞。Llama Guard 4 处理英文政策违规。X-Guard（支持 132 种语言）处理多语言越狱。ShieldGemma-2 捕获基于图像的提示注入。NVIDIA Nemotron 3 Content Safety 覆盖企业类别。Anthropic 的宪法分类器是一种在训练阶段而非服务阶段使用的不同方法。

攻击演进同样重要。PAIR 和 TAP 自动化越狱发现。GCG 运行基于梯度的后缀攻击。多轮攻击和代码切换攻击利用代理记忆。任何部署的 LLM 都需要一个红队测试范围——garak 和 PyRIT 是标准的驱动框架——以及文档化的缓解措施和 CVSS 评分的漏洞发现。

你将加固一个目标应用（可以是 8B 指令微调模型，也可以是其他终极项目中之一的 RAG 聊天机器人），对其运行 6+ 个攻击家族，并生成前后无害化度量结果。

## 概念

安全管道由五层组成。**输入清洗**：剥离零宽字符、解码 base64/rot13、规范化 Unicode。**策略层**：NeMo Guardrails v0.12 护栏（非域外、毒性、PII 提取）。**分类器门控**：对输入使用 Llama Guard 4，对非英文输入使用 X-Guard，对图像输入使用 ShieldGemma-2。**模型**：目标 LLM。**输出过滤**：对输出使用 Llama Guard 4、Presidio PII 脱敏，以及必要的引用检查。**人工审核层**：标记为高风险的输出进入 Slack 队列。

红队测试范围按调度器运行。PAIR 和 TAP 自主发现越狱。GCG 运行基于梯度的后缀攻击。ASCII / base64 / rot13 编码攻击。多轮攻击（角色 adopted、记忆利用）。代码切换攻击（英文与斯瓦希里语或泰语混合）。每次运行都会生成带有 CVSS 评分和披露时间表的结构性发现文件。

宪法自我批判运行是一种训练阶段的干预措施。选取 1k 个有害尝试提示，让模型起草响应，对照书面宪法（不伤害规则）对其进行批判，并在批判循环中重新训练。在保留的评估集上测量前后无害化改进。

## 架构

```
请求（文本 / 图像 / 多语言）
      |
      v
输入清洗（剥离零宽字符、解码、规范化）
      |
      v
NeMo Guardrails v0.12 护栏（非域外、策略）
      |
      v
分类器门控：
  Llama Guard 4（英文）
  X-Guard（多语言，132 种语言）
  ShieldGemma-2（图像提示）
  Nemotron 3 Content Safety（企业）
      |
      v（允许通过）
目标 LLM
      |
      v
输出过滤：Llama Guard 4 + Presidio PII + 引用检查
      |
      v
高危输出的人工审核层

并行：
  红队调度器
    -> garak（经典攻击）
    -> PyRIT（编排红队）
    -> 自主越狱代理（PAIR + TAP）
    -> GCG 后缀攻击
    -> 多语言 / 代码切换
    -> 多轮角色 adopted
```

输出：CVSS 评分的发现 + 披露时间表 + 前后无害化改进

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动框架：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱代理：PAIR（Chao 等，2023）、Tree-of-Attacks（TAP）、GCG 后缀
- 宪法训练：Anthropic 风格的自我批判循环 + 对批判数据的 SFT
- PII 脱敏：Presidio
- 目标：8B 指令微调模型或其他终极项目的 RAG 聊天机器人

```figure
cf-safety-stack
```

## 构建步骤

1. **目标设置。** 在 vLLM 上部署 8B 指令微调模型（或复用其他终极项目的 RAG 聊天机器人）。这是被测应用。

2. **安全管道封装。** 围绕目标连接五层管道。验证每层是否独立可观测（在 Langfuse 中每层有独立的 span）。

3. **分类器覆盖。** 加载 Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在每个分类器上运行小规模标注集以建立基线。

4. **红队调度器。** 调度 garak、PyRIT、PAIR 代理、TAP 代理、GCG 运行器、多轮攻击者和代码切换攻击者。每个运行在独立的队列上。

5. **攻击套件。** 六个攻击家族：（1）PAIR 自动越狱，（2）TAP 攻击树，（3）GCG 梯度后缀，（4）ASCII / base64 / rot13 编码，（5）多轮角色 adopted，（6）多语言代码切换。报告每个家族的成功率。

6. **宪法自我批判。** 整理 1k 个有害尝试提示。对于每个提示，目标模型起草响应。批判 LLM 对照书面宪法进行评分（"不造成伤害"、"引用证据"、"拒绝非法请求"）。批判者提出异议的提示会被重写；目标模型在批判改进的配对上进行微调。在保留的评估集上测量前后无害化改进。

7. **过度拒绝度量。** 在良性提示套件（如 XSTest）上追踪误报率。目标必须对良性问题保持有帮助。

8. **CVSS 评分。** 对每个成功的越狱，按 CVSS 4.0 评分（攻击向量、复杂度、影响）。生成披露时间表和缓解计划。

9. **测试范围自动化。** 以上内容全部通过 cron 运行；发现写入队列；过度拒绝回归告警触发到 Slack。

## 使用方式

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR 代理在目标上运行
[attack]     尝试 1/50：伪装查询为学术研究 ... 已拦截
[attack]     尝试 2/50：诉诸角色扮演 ... 已拦截
[attack]     尝试 3/50：思维链诱导 ... 成功
[finding]    CVSS 4.8 中危：目标上的角色扮演绕过
[range]      50 次尝试中 7 次成功（14% 成功率）
```

## 交付物

`outputs/skill-safety-harness.md` 是交付物。一个生产级分层安全管道，加上一个可复现的红队测试范围，包含前后无害化改进度量。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 运行 6+ 攻击家族，2+ 种语言 |
| 20 | 真阳性 / 假阳性权衡 | 攻击拦截率 vs XSTest 良性通过率 |
| 20 | 自我批判改进 | 保留评估集上的前后无害化对比 |
| 20 | 文档与披露 | CVSS 评分的发现与时间表 |
| 15 | 自动化与可重复性 | 全部内容通过 cron 运行并带告警 |
| **100** | | |

## 练习

1. 在 RAG 聊天机器人上运行 garak 的提示注入插件，比较有无输出过滤层的攻击成功率。

2. 添加第七个攻击家族：通过检索到的文档进行间接提示注入。衡量所需的额外防御措施。

3. 实现"拒绝但提供帮助"模式：当护栏拦截时，目标提供一个更安全的相关问答，而非直接拒绝。测量 XSTest 的改进。

4. 多语言覆盖缺口：找到一个 X-Guard 表现不佳的语言。提出针对该语言微调的数据集方案。

5. 在 30B 模型上运行宪法自我批判，衡量改进幅度是否随模型规模缩放。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 分层安全 | "纵深防御" | 在输入、门控、输出、人工审核各层部署多个护栏 |
| Llama Guard 4 | "Meta 的安全分类器" | 2026 年参考输入/输出内容分类器 |
| PAIR | "越狱代理" | Chao 等人关于 LLM 驱动越狱发现的论文 |
| TAP | "攻击树" | PAIR 的树搜索变体 |
| GCG | "贪心坐标梯度" | 基于梯度的对抗性后缀攻击 |
| 宪法自我批判 | "Anthropic 风格训练" | 目标起草 -> 批判者评分 -> 重写 -> 重新训练 |
| XSTest | "良性探测集" | 用于过度拒绝回归的基准 |
| CVSS 4.0 | "严重性评分" | 安全发现的标准漏洞评分 |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练阶段参考
- [Meta Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — 2026 年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像与多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132 种语言多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 护栏框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 越狱代理论文
