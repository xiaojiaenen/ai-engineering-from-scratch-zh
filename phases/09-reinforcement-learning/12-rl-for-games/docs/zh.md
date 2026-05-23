# 游戏中的强化学习 — AlphaZero、MuZero 与大模型推理时代

> 1992年：TD-Gammon 通过纯粹的时序差分学习在西洋双陆棋上击败人类冠军。2016年：AlphaGo 击败李世石。2017年：AlphaZero 从零开始主导了国际象棋、将棋和围棋。2024年：DeepSeek-R1 证明了相同的配方，用 GRPO 取代 PPO，在推理任务上同样有效。游戏是推动该阶段每一次突破的基准。

**类型:** 构建
**语言:** Python
**先决条件:** 阶段 9 · 05 (DQN), 阶段 9 · 08 (PPO), 阶段 9 · 09 (RLHF), 阶段 9 · 10 (MARL)
**时间:** 约 120 分钟

## 问题

游戏拥有强化学习所需要的一切。清晰的奖励（胜负）。无限的回合（自博弈可重置）。完美的仿真（游戏*本身*就是模拟器）。离散或小型的连续动作空间。多智能体结构迫使其具备对抗鲁棒性。

游戏也是每一次重大强化学习突破的验证舞台。TD-Gammon（西洋双陆棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（星际争霸 II，2019）。MuZero（学习得到的模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）——最新证明表明游戏强化学习技术在文本上同样有效。

本综合项目通过一个统一视角——**自博弈 + 搜索 + 策略改进**——来审视三个里程碑式的架构：AlphaZero、MuZero 和 GRPO。每一个都在前者基础上进行推广；特别是 GRPO，它是将 AlphaZero 的配方应用于大模型推理，其中 token 作为动作，数学验证作为胜利信号。

## 核心概念

![AlphaZero ↔ MuZero ↔ GRPO: 相同循环，不同环境](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017).** Silver 等人。给定一个规则已知的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：一个塔结构 `f_θ(s) → (p, v)`。`p` 是合法动作的先验概率。`v` 是预期游戏结果。
- 蒙特卡洛树搜索 (MCTS)：在每一步，展开一棵可能后续的树。使用 `(p, v)` 作为先验 + 引导。通过 UCB (PUCT) 选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自博弈：智能体 vs 智能体进行游戏。在动作 `t`，MCTS 的访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是游戏结果（+1 / 0 / -1）。

零人类知识。零手工启发式。单一配方，在每项数千万局自博弈后分别精通了国际象棋、将棋和围棋。

**MuZero (2019).** Schrittwieser 等人。移除了规则已知的要求。

- 不再依赖固定环境，而是学习一个*潜态动力学模型* `(h, g, f)`：
  - `h(s)`：将观测编码为潜态。
  - `g(s_latent, a)`：预测下一个潜态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学习的潜态空间*中运行。相同的搜索，相同的训练循环。
- 适用于围棋、国际象棋、将棋*和* Atari —— 一个算法，无需规则知识。

**Stochastic MuZero (2022).** 增加了随机动力学和机会节点；扩展到西洋双陆棋类游戏。

**Muesli, Gumbel MuZero (2022-2024).** 在样本效率和确定性搜索方面的改进。

**GRPO (2024-2025).** DeepSeek-R1 的配方。与 AlphaZero 形状相同的循环，应用于语言模型推理：

- "游戏"：回答一个数学/编程/推理问题。"胜利" = 验证器（测试用例通过，数值答案匹配）返回 1。
- 策略：大模型。动作：token。状态：提示 + 已生成回复。
- 无评论家（PPO 风格的 V_φ）。相反，对于每个提示，从策略中采样 `G` 个完成序列。计算每个的奖励。使用**组相对优势** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- 对参考策略的 KL 惩罚以防止漂移（类似 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

没有奖励模型，没有评论家，没有 MCTS。组相对基准取代了三者。在推理基准上匹配或超越 PPO-RLHF 的质量，且计算成本只是其一小部分。

**完整的 R1 配方。** DeepSeek-R1 (DeepSeek 2025) 在一篇论文中包含两个模型：

- **R1-Zero.** 从 DeepSeek-V3 基础模型开始。无 SFT。直接应用 GRPO，包含两个奖励成分：*准确性奖励*（基于规则——最终答案是否解析为正确数字/代码是否通过单元测试）和*格式奖励*（完成序列是否将其思维链包装在 `<think>…</think>` 标签中）。经过数千步训练，平均响应长度从约 100 增长到约 10,000 个 token，数学基准分数攀升至接近 o1-preview 水平。该模型从零开始学习推理。缺点是：其思维链通常难以阅读，混合语言，且缺乏风格润色。
- **R1.** 通过四阶段流程解决 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集数千条格式清晰的长思维链演示。在基础模型上进行有监督微调。这提供了一个可读的起点。
  2. **面向推理的 GRPO。** 应用 GRPO，使用准确性+格式奖励，并附加*语言一致性*奖励以防止语言切换。
  3. **拒绝采样 + 第二轮 SFT。** 从 RL 检查点采样约 60 万条推理轨迹，仅保留那些最终答案正确且思维链可读的轨迹，并与约 20 万条非推理 SFT 示例（写作、问答、自我认知）结合。再次微调基础模型。
  4. **全谱 GRPO。** 再进行一轮 RL，覆盖推理（基于规则的奖励）和通用对齐（基于有用性/无害性偏好的奖励）。

结果在 AIME 和 MATH-500 上与 o1 相当，且模型足够小可以蒸馏。同一篇论文还通过 SFT'ing R1 的推理轨迹，发布了六个蒸馏的稠密模型（从 Qwen-1.5B 到 Llama-70B）——学生模型无需 RL。对强大 RL 教师进行蒸馏，在学生模型规模上始终优于从头开始的 RL。

**为何选择 GRPO 而非 PPO 进行推理。** DeepSeekMath 论文（2024 年 2 月）中给出三个原因：(1) 无需训练价值网络，节省一半内存；(2) 组基准自然地处理了推理任务产生的稀疏轨迹末端奖励；(3) 每提示归一化使得优势值在难度迥异的问题间可比，这是 PPO 单个评论家无法做到的。

**无搜索 vs 有搜索。** 游戏领域已分化：

- *长视野的完全信息游戏*（围棋、国际象棋）：仍基于搜索。AlphaZero / MuZero 占主导地位。
- *大模型推理*：生产环境中尚无 MCTS；GRPO 进行完整 rollout，推理时使用 best-of-N。过程奖励模型 (PRMs) 暗示步骤级搜索可能被重新加入。

## 构建它

`code/main.py` 中的代码实现了**微缩版 GRPO** —— 一个多组采样的多臂老虎机。算法与在大模型上使用的相同；只是策略和环境更简单。它教授*损失*和*组相对优势*，这是 2025 年的创新。

### 步骤 1：一个微小的验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实的 GRPO 中，验证器运行单元测试或检查数学等式。

### 步骤 2：策略：每个提示上 K 个答案 token 的 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等同于大模型在给定提示下的最后一层输出。

### 步骤 3：组采样和组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是 2024 年 DeepSeek 的技巧。无需评论家。"基准"是组均值，归一化使用组标准差。

### 步骤 4：与 REINFORCE 基准（无价值）比较

相同设置，相同计算量，朴素 REINFORCE。GRPO 收敛更快更稳定。

### 步骤 5：观察熵和 KL

与 RLHF 相同的诊断指标：对参考策略的平均 KL、策略熵、奖励随时间变化。一旦这些稳定，训练就完成了。

## 陷阱

- **通过验证器博弈进行奖励黑客攻击。** GRPO 继承了 RLHF 的风险：如果验证器错误或可被利用，大模型会找到利用方法。健壮的验证器（多测试用例、形式化证明）至关重要。
- **组大小过小。** 组基准的方差与 `1/√G` 成正比。低于 `G = 4` 时，优势信号会有噪声；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的大模型完成序列具有不同的对数概率。按 token 数归一化，或使用序列级对数概率，或截断至最大长度。
- **纯粹的自博弈循环。** AlphaZero 风格的训练可能在一般和博弈中陷入主导循环。通过多样化的对手池（联盟对弈，课程 10）来缓解。
- **搜索-策略不匹配。** AlphaZero 训练策略以模仿搜索输出。如果策略网络太小，无法表示搜索的分布，训练会停滞。
- **计算下限。** MuZero / AlphaZero 需要大量计算。一次消融实验通常需要数百 GPU 小时。存在用于学习的微缩演示（例如，在四子连珠上的 AlphaZero）。
- **验证器覆盖范围。** 对有 bug 的解决方案通过的单元测试会强化该 bug。设计能捕获边界情况的验证器。

## 使用它

2026 年游戏强化学习格局，按领域：

| 领域 | 主导方法 |
|------|----------|
| 双人零和棋盘游戏（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完全信息卡牌游戏（扑克） | CFR + 深度学习 (DeepStack, Libratus, Pluribus) |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略游戏（Dota、星际争霸） | PPO + 自博弈 + 联盟 (OpenAI Five, AlphaStar) |
| 大模型数学/代码推理 | GRPO (DeepSeek-R1, Qwen-RL, 开源复现) |
| 大模型对齐 | DPO / RLHF-PPO (非 GRPO；验证器是偏好而非可验证的) |
| 机器人 | PPO + DR (非游戏强化学习，但使用相同的策略梯度工具) |
| 组合问题 | AlphaZero 变体 (AlphaTensor, AlphaDev) |

*配方*——自博弈、搜索增强改进、策略蒸馏——跨越文本、像素和物理控制。GRPO 是最新的实例；更多即将到来。

## 交付它

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO 多臂老虎机。在 2 个提示 × 每个提示 4 个答案 token 上进行训练。在少于 1,000 次更新内用 `G=8` 收敛。
2. **中等。** 加入 PPO（带裁剪）和朴素 REINFORCE。在相同的老虎机上比较样本效率和奖励方差与 GRPO 的差异。
3. **困难。** 扩展到长度为 2 的"推理链"：智能体生成两个 token，验证器奖励这对 token。测量 GRPO 如何处理跨两步序列的信用分配。（提示：计算每条*完整序列*的组优势，传播到两个 token 位置。）

## 关键术语

| 术语 | 人们常说 | 其实际含义 |
|------|----------|------------|
| MCTS | "使用学习网络的树搜索" | 蒙特卡洛树搜索；使用学习得到的 `(p, v)` 先验进行 UCB1/PUCT 选择。 |
| AlphaZero | "自博弈 + MCTS" | 训练以匹配 MCTS 访问分布和游戏结果的策略-价值网络。 |
| MuZero | "学习模型的 AlphaZero" | 相同的循环，但在通过学习的动力学构建的潜态空间中。 |
| GRPO | "无评论家的 PPO" | 组相对策略优化；带有组均值基准 + KL 的 REINFORCE。 |
| PUCT | "AlphaZero 的 UCB" | `Q + c · p · √N / (1 + N_a)` —— 平衡价值估计与先验。 |
| 自博弈 | "智能体 vs 过去的自己" | 零和博弈的标准；对称的训练信号。 |
| 联盟对弈 | "基于种群的自博弈" | 采样过去、当前和利用者作为对手。 |
| 验证器奖励 | "可验证的 RL" | 奖励来自确定性检查器（测试通过，答案匹配）。 |
| 过程奖励 | "PRM" | 为每个推理步骤评分，不仅仅是最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 引入 GRPO 和组相对基准的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整的四阶段 R1 配方以及 R1-Zero 消融实验。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 大规模的 CFR + 深度学习。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 开启一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 使用自定义奖励函数应用 GRPO 的生产参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 在多规模上对 R1 配方的开源复现。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书对自博弈、搜索以及 R1 在大模型规模上实例化的"设计奖励"的框架。