# 奖励建模与基于人类反馈的强化学习（RLHF）

> 人类无法为"良好的助手回复"编写奖励函数，但可以比较两个回复并选出更好的那个。通过这些比较拟合奖励模型，然后使用强化学习对语言模型进行优化。——Christiano 2017，InstructGPT 2022。这套方法将 GPT-3 变成了 ChatGPT。到了 2026 年，它基本被 DPO 取代——但核心思想模型依然有效。

**类型：** 构建  
**语言：** Python  
**先决条件：** 阶段 5 · 05（情感分析），阶段 9 · 08（PPO）  
**时间：** 约 45 分钟

## 问题所在

你在一个基于下一词预测目标上训练了语言模型。它能写出语法正确的英语。但它也会撒谎、跑题，并且拒绝做它该做的事。你无法通过更多预训练来解决这个问题——网络文本是问题本身，而非解药。

你想要一个*标量奖励*，它能表明"对于指令X，回复A比回复B更好"。手动编写这样的奖励函数是不可能的。"有用性"不是关于token的闭合形式表达式。但人类可以比较两个输出并标记偏好。大规模收集这种偏好数据成本很低。

RLHF（Christiano 等人 2017；Ouyang 等人 2022）将偏好转化为奖励模型，然后通过PPO优化语言模型以对抗该奖励。分三步：SFT → RM → PPO。这是 ChatGPT、Claude、Gemini 以及 2023-2025 年间所有其他对齐大型语言模型的发布配方。

到了 2026 年，PPO 步骤基本被 DPO（阶段 10 · 08）取代，因为它成本更低，且在对齐调优方面效果几乎一样好。但*奖励模型*部分仍然是每个最佳N选1采样器、每个基于可验证奖励的强化学习流程，以及每个使用过程奖励模型的推理模型的基础。理解了 RLHF，你就理解了整个对齐技术栈。

## 核心概念

![三阶段RLHF：SFT、基于成对偏好的RM训练、带KL惩罚的PPO](../assets/rlhf.svg)

**阶段 1：有监督微调（SFT）。** 从一个预训练基础模型开始。在人类编写的、展示目标行为的数据（遵循指令的回复、有帮助的回复等）上进行微调。结果：得到一个模型 `π_SFT`，它*倾向于良好行为*，但其动作空间仍然是无界的。

**阶段 2：奖励模型训练。**

- 收集对同一提示 `x` 的回复对 `(y_+, y_-)`，由人类标注为"y_+ 优于 y_-"。
- 训练一个奖励模型 `R_φ(x, y)`，使其对 `y_+` 赋予更高分数。
- 损失函数：**Bradley-Terry 成对逻辑损失**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid 函数。奖励的差值意味着偏好的对数几率。BT 自 1952 年（Bradley-Terry）以来就是标准，并且是现代 RLHF 中的主导选择。

- `R_φ` 通常从 SFT 模型初始化，在其顶部加一个标量输出头。使用相同的 Transformer 主干网络；一个单独的线性层输出奖励值。

**阶段 3：使用 KL 惩罚，通过 PPO 针对 RM 进行优化。**

- 从 `π_SFT` 初始化可训练的策略 `π_θ`。保持一个冻结的*参考*模型 `π_ref = π_SFT`。
- 在回复 `y` 结束时的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意偏离 `π_SFT` ——它是一个*正则化器*，而非硬性信任域。`β` 通常为 `0.01`-`0.05`。
- 使用此奖励运行 PPO（课程 08）。优势函数在 token 级轨迹上计算，但 RM 只对完整回复进行评分。

**为什么需要 KL？** 如果没有 KL，PPO 会很乐意找到奖励黑客攻击的策略——RM 仅在分布内的数据上训练过。一个分布外的回复可能获得比任何人工撰写的回复更高的分数。KL 使 `π_θ` 保持在 RM 训练数据附近的流形上。它是 RLHF 中最重要的调节旋钮。

**2026 年现状：**

- **DPO**（Rafailov 2023）：闭式代数将阶段 2 和 3 合并为一个在偏好数据上的有监督损失。无需 RM，无需 PPO。在对齐基准测试上质量相当，但计算量小得多。在阶段 10 · 08 中介绍。
- **GRPO**（DeepSeek 2024–2025）：使用群组相对基线代替评价者的 PPO，奖励来自*验证器*（代码运行/数学答案匹配）而非人类训练的 RM。在推理模型中占主导地位。在阶段 9 · 12 中介绍。
- **过程奖励模型（PRMs）：** 对部分解决方案（每个推理步骤）评分，用于 RLHF 和 GRPO 的推理变体。
- **宪法式 AI / RLAIF：** 使用对齐的 LLM 代替人类来生成偏好。扩展了偏好数据的预算。

## 动手构建

本课使用微型的合成"提示"和"回复"（表示为字符串）。RM 是一个基于词袋表示的线性打分器。没有真实的 LLM——重要的是流程的*结构*，而非规模。参见 `code/main.py`。

### 步骤 1：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实的 RLHF 中，这由人类标注员完成。其结构——`(prompt, preferred_response, rejected_response)`——是相同的。

### 步骤 2：Bradley-Terry 奖励模型

线性打分：`R(x, y) = w · bag(y)`。训练以最小化 BT 成对对数损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

经过几百次更新后，`w` 会对积极词汇的 token 赋予正权重，对消极词汇的 token 赋予负权重。

### 步骤 3：基于 RM 的类 PPO 策略

我们的玩具策略从一个词汇表中产生一个 token。我们在 RM 下对该 token 评分，计算 `log π_θ(token | prompt)`，添加一个到参考模型的 KL 惩罚，并应用带截断的 PPO 替代目标。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### 步骤 4：监控 KL

每次更新追踪平均 `KL(π_θ || π_ref)`。如果它超过 `~5-10`，说明策略已大幅偏离 `π_SFT`——`β` 上升或奖励黑客攻击开始。这是真实 RLHF 中最重要的诊断指标。

### 步骤 5：使用 TRL 的生产级配方

一旦你理解了玩具流程，下面是一个真实库用户编写的相同循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——`RewardTrainer` 用于阶段 2，`PPOTrainer`（内置 KL 到参考模型的惩罚）用于阶段 3。

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

库为你做了三件事。`adap_kl_ctrl=True` 实现了自适应 β 调度：如果观察到的 KL 超过 `target_kl`，β 加倍；如果低于一半，β 减半。按照惯例，参考模型是冻结的——你绝不能意外地与 `policy` 共享参数。价值头与策略位于同一个主干网络上（`AutoModelForCausalLMWithValueHead` 附加了一个标量 MLP 头），这就是为什么 TRL 会分别报告 `policy/kl` 和 `value/loss`。

## 陷阱与对策

- **过度优化 / 奖励黑客攻击。** RM 是不完美的；`π_θ` 会找到分数很高但实际上很差的对抗性补全。症状：奖励无限攀升，而人类评估分数停滞或下降。解决方法：提前停止训练，提高 `β`，扩充 RM 训练数据。
- **长度黑客攻击。** 在有帮助的回复上训练的 RM 往往会隐式奖励长度。策略学会填充回复。补救措施：长度归一化奖励，或使用长度感知的 RM 进行 RLAIF。
- **RM 规模太小。** RM 的规模至少应与策略相当。一个微小的 RM 无法准确评估策略的输出。
- **KL 调优。** β 太低 → 漂移和奖励黑客攻击。β 太高 → 策略几乎不变。标准技巧是使用*自适应* β，以每步固定的 KL 为目标。
- **偏好数据噪声。** 约 30% 的人类标签是嘈杂或模糊的。通过在共识过滤后的数据上训练 RM 或对 BT 使用温度参数来进行校准。
- **离策略问题。** 第一个 epoch 后，PPO 数据是略微离策略的。如课程 08 中所述，监控截断比例。

## 应用场景

2026 年的 RLHF 是分层的：

| 层级 | 目标 | 方法 |
|------|------|------|
| 指令遵循、有用性、无害性 | 对齐 | DPO（阶段 10 · 08）优于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 使用验证器奖励的 GRPO（阶段 9 · 12）。 |
| 长程多步任务 | 智能体 | 使用过程奖励模型在步骤上进行 PPO / GRPO。 |
| 安全/拒绝行为 | 安全 | 使用单独安全 RM 的 RLHF-PPO，或宪法式 AI。 |
| 推理时的最佳 N 选 1 | 快速对齐 | 在解码时使用 RM；无需策略训练。 |
| 奖励蒸馏 | 推理计算量 | 在冻结的 LM 上训练一个小型"奖励头"。 |

RLHF 在 2022-2024 年*曾是*主流方法。到了 2026 年，生产级对齐流程以 DPO 为首选，仅在对 RM 要求高或安全关键的步骤中使用 PPO。

## 发布保存

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 练习

1. **简单。** 在 500 个合成偏好对上训练 `code/main.py` 中的 Bradley-Terry 奖励模型。在预留的 100 个样本对上测量成对准确率。应超过 90%。
2. **中等。** 使用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对于每种设置，绘制更新过程中 RM 分数与 KL 到参考模型的关系图。哪种设置会发生奖励黑客攻击？
3. **困难。** 在相同的偏好数据上实现 DPO（闭式偏好似然损失），并与 RLHF-PPO 流程在计算量和最终 RM 分数上进行比较。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|----------------|----------|
| RLHF | "对齐 RL" | 三阶段 SFT + RM + PPO 流程（Christiano 2017, Ouyang 2022）。 |
| 奖励模型 (RM) | "打分网络" | 通过 Bradley-Terry 拟合到成对偏好的标量函数。 |
| Bradley-Terry | "成对逻辑损失" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准的 RM 目标。 |
| KL 惩罚 | "保持靠近参考模型" | 奖励中的 `β · KL(π_θ || π_ref)`；防止奖励黑客攻击的正则化器。 |
| 奖励黑客攻击 | "古德哈特定律" | 策略利用 RM 的缺陷；症状：奖励上升，人类评估持平。 |
| RLAIF | "AI 标注的偏好" | 标签来自另一个 LM 而非人类的 RLHF。 |
| PRM | "过程奖励模型" | 对部分推理步骤评分；用于推理流程。 |
| 宪法式 AI | "Anthropic 的方法" | 由明确规则引导的 AI 生成的偏好。 |

## 延伸阅读

- [Christiano 等人 (2017). 从人类偏好进行深度强化学习](https://arxiv.org/abs/1706.03741) —— 启动 RLHF 的论文。
- [Ouyang 等人 (2022). InstructGPT —— 使用人类反馈训练语言模型遵循指令](https://arxiv.org/abs/2203.02155) —— ChatGPT 背后的配方。
- [Stiennon 等人 (2020). 通过人类反馈学习摘要](https://arxiv.org/abs/2009.01325) —— 更早的用于摘要任务的 RLHF。
- [Rafailov 等人 (2023). 直接偏好优化](https://arxiv.org/abs/2305.18290) —— DPO；2026 年后 RLHF 的默认选择。
- [Bai 等人 (2022). 宪法式 AI：来自 AI 反馈的无害性](https://arxiv.org/abs/2212.08073) —— RLAIF 和自我批判循环。
- [Anthropic RLHF 论文 (Bai 等人 2022). 训练一个有帮助且无害的助手](https://arxiv.org/abs/2204.05862) —— HH 论文。
- [Hugging Face TRL 库](https://huggingface.co/docs/trl) —— 生产级 `RewardTrainer` 和 `PPOTrainer`。阅读训练器源码以了解自适应 KL 和价值头的细节。
- [Hugging Face —— 图解基于人类反馈的强化学习](https://huggingface.co/blog/rlhf) 作者 Lambert, Castricato, von Werra, Havrilla —— 带有图表的三阶段流程权威介绍。
- [von Werra 等人 (2020). TRL: Transformer 强化学习](https://github.com/huggingface/trl) —— 该库；`examples/` 包含用于 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). 第 17.4 节 —— 设计奖励信号](http://incompleteideas.net/book/RLbook2020.pdf) —— 奖励假设视角；是思考奖励黑客攻击问题的必要前提。