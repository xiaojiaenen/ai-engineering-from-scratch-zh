# 毕业项目 15 — 宪法安全基座 + 红队靶场

> Anthropic的宪法分类器、Meta的Llama Guard 4、Google的ShieldGemma-2、NVIDIA的Nemotron 3内容安全模型，以及用于多语言覆盖的X-Guard共同定义了2026年的安全分类器技术栈。garak、PyRIT、NVIDIA Aegis和promptfoo已成为标准对抗性评估工具。NeMo Guardrails v0.12将它们整合到生产流水线中。本毕业项目将所有组件串联起来：为目标应用构建分层安全基座，运行一个涵盖6种以上攻击类型的自主红队智能体，并通过宪法自我批判生成可量化的无害性增量。

**类型：** 毕业项目
**语言：** Python（安全流水线、红队）、YAML（策略配置）
**前置条件：** 阶段10（从零构建LLM）、阶段11（LLM工程）、阶段13（工具）、阶段14（智能体）、阶段18（伦理、安全、对齐）
**涵盖阶段：** P10 · P11 · P13 · P14 · P18
**时间：** 25小时

## 问题

2026年LLM安全的前沿问题不在于分类器是否有效（它们大致有效），而在于如何围绕生产应用正确组合它们，避免过度拒绝或留下明显漏洞。Llama Guard 4处理英语策略违规。X-Guard（覆盖132种语言）处理多语言越狱。ShieldGemma-2捕获基于图像的提示注入。NVIDIA Nemotron 3内容安全覆盖企业级类别。Anthropic的宪法分类器是训练阶段使用的独立方法。

攻击演化同样重要。PAIR和TAP自动化发现越狱。GCG运行基于梯度的后缀攻击。多轮对话和代码切换攻击利用智能体记忆。任何部署的LLM都需要红队靶场——garak和PyRIT是标准驱动工具——以及记录在案的缓解措施和CVSS评分的发现。

你将加固一个目标应用（可以是8B指令微调模型或其他毕业项目的RAG聊天机器人），对其运行6种以上攻击类型，并生成攻击前后的无害性测量结果。

## 概念

安全流水线分为五层。**输入净化**：剥离零宽字符，解码base64/rot13，标准化Unicode。**策略层**：NeMo Guardrails v0.12护栏（领域外、毒性、PII提取）。**分类器门控**：输入端使用Llama Guard 4，非英语使用X-Guard，图像输入使用ShieldGemma-2。**模型**：目标LLM。**输出过滤**：输出端使用Llama Guard 4，Presidio PII擦除，必要时执行引用强制。**人机协同层**：标记为高风险的输出进入Slack队列。

红队靶场在调度器上运行。PAIR和TAP自主发现越狱。GCG运行基于梯度的后缀攻击。ASCII / base64 / rot13编码攻击。多轮攻击（角色扮演、记忆利用）。代码切换攻击（混合英语与斯瓦希里语或泰语）。每次运行生成结构化的发现文件，包含CVSS评分和披露时间线。

宪法自我批判运行是训练时的干预。取1000个有害尝试提示，让模型草拟响应，根据书面宪法（不伤害规则）进行批判，并基于批判循环重新训练。在留出集上测量攻击前后的无害性增量。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全分类器：Llama Guard 4, ShieldGemma-2, NVIDIA Nemotron 3 Content Safety, X-Guard
- 护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动：garak (NVIDIA), PyRIT (Microsoft Azure), NVIDIA Aegis, promptfoo
- 越狱智能体：PAIR (Chao et al., 2023), Tree-of-Attacks (TAP), GCG 后缀攻击
- 宪法训练：Anthropic风格的自我批判循环 + 对批判结果进行SFT
- PII擦除：Presidio
- 目标：8B指令微调模型或其他毕业项目的RAG聊天机器人

## 构建指南

1.  **目标设置。** 在vLLM上部署8B指令微调模型（或复用其他毕业项目的RAG聊天机器人）。这是被测试的应用。

2.  **安全流水线封装。** 将五层流水线围绕目标进行串联。验证每层均可独立观测（在Langfuse中每层一个span）。

3.  **分类器覆盖。** 加载Llama Guard 4, X-Guard（多语言）, ShieldGemma-2（图像）。在小型标记集上运行每个分类器以建立基线。

4.  **红队调度器。** 调度garak, PyRIT, PAIR智能体, TAP智能体, GCG运行器, 多轮攻击者, 代码切换攻击者。每个运行在独立队列上。

5.  **攻击套件。** 六类攻击：(1) PAIR自动越狱, (2) TAP攻击树, (3) GCG梯度后缀, (4) ASCII / base64 / rot13编码, (5) 多轮角色扮演, (6) 多语言代码切换。报告每类攻击的成功率。

6.  **宪法自我批判。** 策划1000个有害尝试提示。对于每个提示，目标草拟响应。一个评判LLM根据书面宪法（“不伤害”、“引用证据”、“拒绝非法请求”）进行评分。对评判提出异议的提示进行重写；目标基于批判改进后的配对进行微调。在留出集上测量攻击前后的无害性。

7.  **过度拒绝测量。** 在良性提示套件（例如XSTest）上跟踪假阳性率。目标必须在良性问题上保持有用性。

8.  **CVSS评分。** 对每个成功的越狱，根据CVSS 4.0（攻击向量、复杂性、影响）进行评分。生成披露时间线和缓解计划。

9.  **靶场自动化。** 上述一切通过cron定时运行；发现写入队列；过度拒绝回归告警发送至Slack。

## 使用示例

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付成果

`outputs/skill-safety-harness.md`是交付物。一个生产级的分层安全流水线，加上一个可重现的红队靶场，包含攻击前后的无害性增量。

| 权重 | 标准 | 衡量方法 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 运行6种以上攻击类型，覆盖2种以上语言 |
| 20 | 真阳性/假阳性权衡 | 攻击拦截率 vs XSTest良性通过率 |
| 20 | 自我批判增量 | 留出集上攻击前后的无害性变化 |
| 20 | 文档与披露 | 带时间线的CVSS评分发现 |
| 15 | 自动化与可重复性 | 一切通过cron运行并带有告警 |
| **100** | | |

## 练习

1.  在RAG聊天机器人上运行garak的提示注入插件，比较开启和关闭输出过滤层时的攻击成功率。
2.  增加第七类攻击：通过检索文档进行间接提示注入。测量所需额外防御。
3.  实现“拒绝并提供帮助”模式：当护栏拦截时，目标提供更安全的相关回答，而非直接拒绝。测量XSTest变化。
4.  多语言覆盖缺口：找到X-Guard表现不佳的语言。提出针对该语言的微调数据集。
5.  在30B模型上运行宪法自我批判，测量增量是否随模型规模增长。

## 关键术语

| 术语 | 常用说法 | 实际含义 |
|------|-----------------|------------------------|
| 分层安全 | “纵深防御” | 在输入、门控、输出、人机协同层设置多重护栏 |
| Llama Guard 4 | “Meta的安全分类器” | 2026年参考的输入/输出内容分类器 |
| PAIR | “越狱智能体” | Chao等人关于LLM驱动越狱发现的论文 |
| TAP | “攻击树” | PAIR的树搜索变体 |
| GCG | “贪婪坐标梯度” | 基于梯度的对抗性后缀攻击 |
| 宪法自我批判 | “Anthropic式训练” | 目标草拟 -> 评判评分 -> 重写 -> 重训练 |
| XSTest | “良性探测集” | 用于过度拒绝回归的基准 |
| CVSS 4.0 | “严重性评分” | 安全发现的标准漏洞评分 |

## 延伸阅读

- [Anthropic 宪法分类器](https://www.anthropic.com/research/constitutional-classifiers) — 训练时参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像 + 多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业级参考
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132种语言的多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA红队工具包
- [PyRIT](https://github.com/Azure/PyRIT) — 微软红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 护栏框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 越狱智能体论文