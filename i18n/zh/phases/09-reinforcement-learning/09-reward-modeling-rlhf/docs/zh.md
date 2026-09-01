# 奖励建模与 RLHF

> 人类无法为"良好的助手回复"编写奖励函数，但可以对两个回复进行比较并挑选出更好的那个。基于这些比较训练一个奖励模型，然后用强化学习 (RL) 对该语言模型进行优化。Christian 2017。InstructGPT 2022。这是将 GPT-3 转变为 ChatGPT 的配方。到了 2026 年，它大部分已被 DPO 取代——但心智模型依然成立。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 5 · 05（情感分析）、阶段 9 · 08（PPO）
**时间：** 约 45 分钟

## 问题所在

你用下一个 token 预测的目标训练了一个语言模型。它能写出语法正确的英文。但它也会撒谎、啰嗦，而且拒绝拒绝。你无法通过更多的预训练来解决这个问题——网络文本本身就是问题所在，而非解药。

你想要一个*标量奖励*，它能告诉你"对于指令 X，回复 A 比回复 B 更好"。手工编写这个奖励函数是不可能的。"有用性"并非 token 上的闭式表达式。但人类可以对两个输出进行比较并标记偏好。这在规模上收集成本很低。

RLHF（Christiano 等人 2017；Ouyang 等人 2022）将偏好转换为奖励模型，然后通过 PPO 针对该奖励对 LM 进行优化。分为三个步骤：SFT → RM → PPO。这是打造 ChatGPT、Claude、Gemini 以及 2023–2025 年所有对齐 LLM 的配方。

到了 2026 年，PPO 步骤大部分已被 DPO（阶段 10 · 08）取代，因为它更便宜且在对齐调优方面几乎同样有效。但*奖励模型*这一环节仍然是每个 Best-of-N 采样器、每个基于可验证奖励的 RL 管道以及每个使用过程奖励模型的推理模型的基础。理解 RLHF 就理解了整个对齐栈。

## 概念

![三阶段 RLHF：SFT、基于成对偏好的奖励模型训练、带 KL 惩罚的 PPO](../assets/rlhf.svg)

**阶段 1：监督微调 (SFT)。** 从一个预训练的基础模型开始。在目标行为的人类示范数据（指令遵循回复、有帮助的回复等）上进行微调。结果得到一个模型 `π_SFT`，它*偏向良好行为*，但仍然具有无界的动作空间。

**阶段 2：奖励模型训练。**

- 收集针对提示 `x` 的回复对 `(y_+, y_-)`，由人类标记为"y_+ 优于 y_-。"
- 训练一个奖励模型 `R_φ(x, y)` 以给 `y_+` 分配更高的分数。
- 损失函数：**Bradley-Terry 成对逻辑回归**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid 函数。奖励差异隐含了偏好的对数几率。BT 自 1952 年（Bradley-Terry）以来一直是标准做法，也是现代 RLHF 中的主导选择。

- `R_φ` 通常从 SFT 模型初始化，并在顶部添加一个标量头。使用相同的 transformer 骨干网络；一个线性层输出奖励值。

**阶段 3：带 KL 惩罚的 PPO 针对 RM 进行优化。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保持一个冻结的*参考模型* `π_ref = π_SFT`。
- 在回复 `y` 结束时的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意偏离 `π_SFT` —— 它是一种*正则化*，而非硬性的信任区域。`β` 通常取 `0.01`-`0.05`。
- 使用此奖励运行 PPO（课程 08）。优势在 token 级轨迹上计算，但 RM 只对完整回复进行打分。

**为什么需要 KL？** 没有它，PPO 会欣然找到奖励黑客策略 —— RM 仅在分布内回复上训练过。一个分布外的回复可能比任何人类编写的回复得分都高。KL 将 `π_θ` 保持在 RM 训练所在的流形附近。这是 RLHF 中最重要的旋钮。

**2026 年现状：**

- **DPO**（Rafailov 2023）：闭式代数将阶段 2+3 合并为偏好数据上的单个监督损失。没有 RM，没有 PPO。在对齐基准测试上达到相同质量，但计算量仅为 fractions。在阶段 10 · 08 中有介绍。
- **GRPO**（DeepSeek 2024–2025）：用组相对基线替代 critic 的 PPO，奖励来自*验证器*（代码运行 / 数学答案匹配）而非人工训练的 RM。在推理模型中占主导地位。在阶段 9 · 12 中有介绍。
- **过程奖励模型 (PRM)：** 对部分解决方案（每个推理步骤）进行打分，用于推理模型中的 RLHF 和 GRPO 变体。
- **Constitutional AI / RLAIF：** 使用对齐的 LLM 生成偏好，而非人类。扩展偏好预算。

```figure
reward-model
```

## 动手构建

本课使用小型合成"提示"和"回复"，表示为字符串。RM 是基于词袋表示的线性评分器。没有真正的 LLM —— 管道的*形状*比规模更重要。参见 `code/main.py`。

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

在实际 RLHF 中，这由人类标注员替代。其结构 —— `(prompt, preferred_response, rejected_response)` —— 完全相同。

### 步骤 2：Bradley-Terry 奖励模型

线性得分：`R(x, y) = w · bag(y)`。训练以最小化 BT 成对对数损失：

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

经过数百次更新后，`w` 会对好词 token 分配正权重，对坏词 token 分配负权重。

### 步骤 3：在 RM 之上的类 PPO 策略

我们的玩具策略从词汇表中产生一个 token。我们在 RM 下对该 token 进行打分，计算 `log π_θ(token | prompt)`，加上到参考模型的 KL 惩罚，并应用截断的 PPO 代理。

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

跟踪每次更新的平均 `KL(π_θ || π_ref)`。如果它超过 `~5-10`，则策略已远离 `π_SFT` —— β 正在上升或奖励黑客开始出现。这是实际 RLHF 中的首要诊断指标。

### 步骤 5：使用 TRL 的生产配方

一旦理解了玩具管道，以下是真实库用户编写的相同循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现 —— `RewardTrainer` 用于阶段 2，`PPOTrainer`（内置到参考模型的 KL）用于阶段 3。

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

库为你做了三件事。`adap_kl_ctrl=True` 实现了自适应 β 调度：如果观测到的 KL 超过 `target_kl`，β 翻倍；如果低于一半，β 减半。参考模型按惯例是冻结的 —— 你必须不能意外地与 `policy` 共享参数。价值头与策略位于同一骨干网络上（`AutoModelForCausalLMWithValueHead` 附加了一个标量 MLP 头），这就是为什么 TRL 分别报告 `policy/kl` 和 `value/loss` 的原因。

## 陷阱

- **过度优化 / 奖励黑客。** RM 是不完美的；`π_θ` 找到了得分高但质量差的对抗性完成。症状：奖励无限攀升，而人工评估分数停滞或下降。修复：提前停止，提高 `β`，扩充 RM 训练数据。
- **长度黑客。** 在有帮助的回复上训练的 RM 往往隐式地奖励长度。策略学会填充回复。补救措施：长度归一化奖励，或使用具有长度感知 RM 的 RLAIF。
- **RM 太小。** RM 至少应与策略一样大。一个微小的 RM 无法忠实地评估策略的输出。
- **KL 调优。** β 过低 → 漂移和奖励黑客。β 过高 → 策略几乎不变。标准技巧是使用*自适应* β 来针对每个步骤的固定 KL。
- **偏好数据噪声。** 约 30% 的人类标签存在噪声或歧义。通过在与协议过滤的数据上训练 RM 来校准，或在 BT 中使用温度参数。
- **离策问题。** PPO 数据在第一轮之后略有离策。像课程 08 那样监控截断比例。

## 应用

2026 年的 RLHF 是分层的：

| 层级 | 目标 | 方法 |
|-------|--------|--------|
| 指令遵循、有用性、无害性 | 对齐 | DPO（阶段 10 · 08）优于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 带验证器奖励的 GRPO（阶段 9 · 12）。 |
| 长程多步任务 | 智能体 | 带过程奖励模型的 PPO / GRPO。 |
| 安全 / 拒绝行为 | 安全 | 带独立安全 RM 的 RLHF-PPO，或 Constitutional AI。 |
| 推理时的 Best-of-N | 快速对齐 | 在解码时使用 RM；无需策略训练。 |
| 奖励蒸馏 | 推理计算 | 在冻结的 LM 上训练一个小型"奖励头"。 |

RLHF 是 2022–2024 年的*主要*方法。在 2026 年，生产对齐管道优先使用 DPO，仅在对 RM 密集或安全关键的步骤中使用 PPO。

## 交付

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: 为语言模型设计 RLHF / DPO / GRPO 对齐管道，包括 RM、KL 和数据处理策略。
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

给定一个基础 LM、目标行为（对齐 / 推理 / 拒绝 / 智能体）以及偏好或验证器预算，输出：

1. 阶段。SFT？RM？DPO？GRPO？附理由。
2. 偏好或验证器来源。人类、AI 反馈、基于规则、单元测试通过或奖励蒸馏。
3. KL 策略。固定 β、自适应 β 或 DPO（隐式 KL）。
4. 诊断指标。平均 KL、奖励稳定性、过度优化防护（保留人工评估集）。
5. 安全门。红队测试集、拒绝率、安全 RM 与有用性 RM 分离。

拒绝在没有 KL 监控的情况下交付 RLHF-PPO。拒绝使用比目标策略小的 RM。拒绝仅基于长度的奖励。标记任何未保留盲盒人工评估集的管道为缺乏过度优化防护。
```

## 练习

1. **简单。** 在 `code/main.py` 中的 500 个合成偏好对上训练 Bradley-Terry 奖励模型。在保留的 100 个对上测量成对准确率。应超过 90%。
2. **中等。** 使用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对每种情况，绘制更新过程中 RM 分数与到参考模型的 KL 的图表。哪种情况发生了奖励黑客？
3. **困难。** 在同一偏好数据上实现 DPO（闭式偏好似然损失），并比较其与 RLHF-PPO 管道在计算使用和最终 RM 分数方面的差异。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|-----------------|-----------------------|
| RLHF | "对齐 RL" | 三阶段 SFT + RM + PPO 管道（Christiano 2017，Ouyang 2022）。 |
| 奖励模型 (RM) | "评分网络" | 通过 Bradley-Terry 拟合到成对偏好的学习标量函数。 |
| Bradley-Terry | "成对逻辑损失" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标。 |
| KL 惩罚 | "保持在参考附近" | 奖励中的 `β · KL(π_θ \|\| π_ref)`；反奖励黑客的正则化器。 |
| 奖励黑客 | "古德哈特定律" | 策略利用 RM 缺陷；症状：奖励上升，人工评估持平。 |
| RLAIF | "AI 标记的偏好" | 标签来自另一个 LM 而非人类的 RLHF。 |
| PRM | "过程奖励模型" | 对部分推理步骤进行打分；用于推理管道。 |
| Constitutional AI | "Anthropic 的方法" | 由显式规则引导的 AI 生成偏好。 |

## 延伸阅读

- [Christiano 等人 (2017)。深度强化学习从人类偏好中](https://arxiv.org/abs/1706.03741) — 开启 RLHF 的论文。
- [Ouyang 等人 (2022)。InstructGPT — 训练语言模型以通过人类反馈遵循指令](https://arxiv.org/abs/2203.02155) — ChatGPT 背后的配方。
- [Stiennon 等人 (2020)。从人类反馈中学习总结](https://arxiv.org/abs/2009.01325) — 早期用于总结的 RLHF。
- [Rafailov 等人 (2023)。直接偏好优化](https://arxiv.org/abs/2305.18290) — DPO；2026 年后 RLHF 的默认方法。
- [Bai 等人 (2022)。Constitutional AI：从 AI 反馈中实现无害性](https://arxiv.org/abs/2212.08073) — RLAIF 和自我批判循环。
- [Anthropic RLHF 论文 (Bai 等人 2022)。训练有帮助且无害的助手](https://arxiv.org/abs/2204.05862) — HH 论文。
- [Hugging Face TRL 库](https://huggingface.co/docs/trl) — 生产级 `RewardTrainer` 和 `PPOTrainer`。阅读训练器源码以了解自适应 KL 和价值头细节。
- [Hugging Face — 图解从人类反馈中的强化学习](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — 带有图示的三阶段管道的典型 walkthrough。
- [von Werra 等人 (2020)。TRL：Transformer 强化学习](https://github.com/huggingface/trl) — 该库；`examples/` 中有 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018)。第 17.4 节 — 设计奖励信号](http://incompleteideas.net/book/RLbook2020.pdf) — 奖励假设视角；思考奖励黑客的必要前提。
