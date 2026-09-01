# 游戏强化学习 —— AlphaZero、MuZero 与 LLM 推理时代

> 1992 年：TD-Gammon 凭借纯 TD 算法击败人类西洋双陆棋冠军。2016 年：AlphaGo 击败李世石。2017 年：AlphaZero 从零开始统治国际象棋、将棋和围棋。2024 年：DeepSeek-R1 证明同一套配方（用 GRPO 替代 PPO）在推理任务上同样有效。游戏是这个阶段推动每项突破的基准。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 9 · 05（DQN）、Phase 9 · 08（PPO）、Phase 9 · 09（RLHF）、Phase 9 · 10（MARL）
**时间：** 约 120 分钟

## 问题所在

游戏拥有强化学习所渴望的一切：干净的奖励（赢/输）、无限集数（自对弈重置）、完美仿真（游戏本身就是模拟器）、离散或小连续动作空间、以及迫使对抗鲁棒性的多智能体结构。

而且，每一项重大强化学习突破都是通过游戏来测试的：TD-Gammon（西洋双陆棋，1992 年）。Atari-DQN（2013 年）。AlphaGo（2016 年）。AlphaZero（2017 年）。OpenAI Five（Dota 2，2019 年）。AlphaStar（星际争霸 II，2019 年）。MuZero（学习模型，2019 年）。AlphaTensor（矩阵乘法，2022 年）。AlphaDev（排序算法，2023 年）。DeepSeek-R1（数学推理，2025 年）——最新证明游戏强化学习技术在文本上也有效。

本综合项目通过一个统一视角考察三项标志性架构——AlphaZero、MuZero 和 GRPO——**自对弈 + 搜索 + 策略改进**。每一项都是前一项的推广；特别是 GRPO，是将 AlphaZero 配方应用于 LLM 推理：token 作为动作，数学验证作为赢的信号。

## 概念

![AlphaZero ↔ MuZero ↔ GRPO：相同的循环，不同的环境](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # 与自己下棋
    policy_target = search.improved_policy(trajectory) # 搜索改进原始策略
    policy_net.update(policy_target, value_target)     # 以搜索输出为监督信号训练
```

**AlphaZero（2017）。** Silver 等人。给定已知规则的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：单一塔 `f_θ(s) → (p, v)`。`p` 是合法动作上的先验分布。`v` 是期望的博弈结果。
- 蒙特卡洛树搜索（MCTS）：在每一步，展开可能延续的树。使用 `(p, v)` 作为先验 + 引导值。通过 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自对弈：智能体之间下棋。在第 `t` 步，MCTS 访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是博弈结果（+1 / 0 / -1）。

零人类知识。零手工启发式规则。一个简单的配方，在各自数百万场自对弈后精通了国际象棋、将棋和围棋。

**MuZero（2019）。** Schrittwieser 等人。消除了需要已知规则的依赖。

- 不用固定环境，而是学习一个**潜在动力学模型** `(h, g, f)`：
  - `h(s)`：将观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在**学习的潜在空间**中运行。相同的搜索，相同的训练循环。
- 在围棋、国际象棋、将棋和 Atari 上都有效——一个算法，无需规则知识。

**随机 MuZero（2022）。** 加入随机动力学和概率节点；扩展到西洋双陆棋类游戏。

**Muesli、Gumbel MuZero（2022-2024）。** 提高样本效率和确定性搜索的改进版本。

**GRPO（2024-2025）。** DeepSeek-R1 配方。与 AlphaZero 相同的循环，应用于语言模型推理：

- "游戏"：回答数学/编程/推理问题。"胜利" = 验证器（测试用例通过、数值答案匹配）返回 1。
- 策略：LLM。动作：token。状态：提示 + 已生成的响应。
- 没有价值网络（PPO 风格的 V_φ）。相反，对于每个提示，从策略采样 `G` 个完成序列。计算每个的奖励。使用**组内相对优势** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- 对参考策略的 KL 惩罚以防止漂移（类似 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无奖励模型，无价值网络，无 MCTS。组内基线替代了所有三项。在推理基准上匹配或超越 PPO-RLHF 质量，但计算量小得多。

**R1 完整配方。** DeepSeek-R1（DeepSeek 2025）在一篇论文中提出了两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基础模型开始。不做 SFT。直接应用 GRPO，包含两个奖励组件：**准确性奖励**（基于规则——最终答案是否解析为正确数字/代码是否通过单元测试）和**格式奖励**（完成序列是否用 `<think>…</think>` 标签包裹其思维链）。经过数千步，平均响应长度从约 100 token 增长到约 10,000 token，数学基准分数攀升至接近 o1-preview 水平。该模型从零学会推理。缺点：其思维链往往难以阅读、混合语言、缺乏风格打磨。
- **R1。** 通过四阶段流水线修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集数千条格式良好的长思维链示范。在它们上监督微调基础模型。这提供了一个可读的起点。
  2. **推理导向 GRPO。** 应用 GRPO，奖励包含准确性+格式奖励，加上**语言一致性**奖励以防止代码切换。
  3. **拒绝采样 + SFT 第 2 轮。** 从 RL 检查点采样约 600K 推理轨迹，只保留最终答案正确且思维链可读的，并与约 200K 非推理 SFT 示例（写作、问答、自我认知）结合。再次微调基础模型。
  4. **全谱 GRPO。** 再进行一轮 RL，覆盖推理（基于规则的奖励）和通用对齐（有用性/无害性偏好奖励）。

结果在 AIME 和 MATH-500 上匹配 o1，且足够小以蒸馏。同一篇论文还发布了六个蒸馏稠密模型（Qwen-1.5B 至 Llama-70B），通过在 R1 的推理轨迹上 SFT 实现——学生模型不涉及 RL。强 RL 教师的蒸馏在学生规模上一致优于从零开始的 RL。

**为什么推理用 GRPO 而非 PPO。** DeepSeekMath 论文（2024 年 2 月）的三个原因：（1）无需训练价值网络，内存减半；（2）组内基线自然处理推理任务产生的稀疏终局奖励；（3）逐提示归一化使不同难度问题的优势可比，而 PPO 的单价值网络做不到这一点。

**无搜索 vs 有搜索。** 游戏已分化为：

- *完全信息长视野游戏*（围棋、国际象棋）：仍有搜索。AlphaZero / MuZero 主导。
- *LLM 推理*：生产中尚未引入 MCTS；GRPO 用于完整 rollout，推理时采用 best-of-N。过程奖励模型（PRM）暗示步骤级搜索将重新引入。

```figure
f3-selfplay-ladder
```

## 动手构建

`code/main.py` 中的代码实现了**微型 GRPO**——一个多组样本的赌博机。算法与 LLM 相同；只是策略和环境更简单。它教授了**损失**和**组内相对优势**，这是 2025 年的创新。

### 步骤 1：小型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在实际 GRPO 中，验证器运行单元测试或检查数学等式。

### 步骤 2：策略：每个提示的 K 个答案 token 上的 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

相当于条件于提示的 LLM 最后一层输出。

### 步骤 3：组采样与组内相对优势

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
    # KL 惩罚：将 theta 拉向参考
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组内相对优势是 2024 年 DeepSeek 的技巧。无需价值网络。"基线"是组均值，归一化使用组标准差。

### 步骤 4：与 REINFORCE 基线（无价值）对比

相同设置，相同计算量，普通 REINFORCE。GRPO 收敛更快更稳定。

### 步骤 5：观察熵和 KL

与 RLHF 相同的诊断：相对于参考的均值 KL、策略熵、奖励随时间变化。这些稳定后，训练即完成。

## 常见陷阱

- **通过验证器作弊导致的奖励黑客。** GRPO 继承了 RLHF 的风险：如果验证器有误或可被利用，LLM 会找到漏洞。鲁棒的验证器（多个测试用例、形式化证明）很重要。
- **组大小过小。** 组内基线的方差约为 `1/√G`。当 `G < 4` 时，优势信号噪声大；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的 LLM 完成序列具有不同的对数概率。按 token 数归一化，或使用序列级对数概率，或截断到最大长度。
- **纯自对弈循环。** AlphaZero 式训练可能在一般和博弈中陷入支配循环。通过多样化对手池（联赛模式、第 10 课）缓解。
- **搜索-策略不匹配。** AlphaZero 训练策略以模仿搜索输出。如果策略网络太小无法表示搜索的分布，训练将停滞。
- **计算门槛。** MuZero / AlphaZero 需要大量计算。单个消融实验通常需要数百 GPU 小时。迷你演示版（如 Connect Four 上的 AlphaZero）可用于学习。
- **验证器覆盖率。** 对有缺陷方案通过的单元测试会强化该缺陷。设计能捕获边缘情况的验证器。

## 应用场景

2026 年的游戏 RL 格局，按领域分类：

| 领域 | 主导方法 |
|------|----------|
| 两人零和棋类（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完全信息纸牌游戏（扑克） | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略（Dota、星际争霸） | PPO + 自对弈 + 联赛（OpenAI Five、AlphaStar） |
| LLM 数学/代码推理 | GRPO（DeepSeek-R1、Qwen-RL、开源复现） |
| LLM 对齐 | DPO / RLHF-PPO（不是 GRPO；验证器是偏好而非可验证） |
| 机器人 | PPO + DR（非游戏 RL，但使用相同的策略梯度工具） |
| 组合问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

**配方**——自对弈、搜索增强改进、策略蒸馏——跨越文本、像素和物理控制。GRPO 是最年轻的实例；更多正在涌现。

## 交付物

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: 为给定领域设计游戏 RL 或推理 RL 训练流程（AlphaZero / MuZero / GRPO）。
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

给定目标（完全信息游戏/不完全信息/Atari/LLM 推理/组合问题），输出：

1. 环境适配性。已知规则？马尔可夫？随机？多智能体？决定 AlphaZero vs MuZero vs GRPO。
2. 搜索策略。MCTS（带学习先验的 PUCT）、Gumbel 采样、best-of-N、或无。
3. 自对弈方案。对称自对弈 / 联赛 / 离线数据 / 验证器生成。
4. 目标信号。博弈结果 / 验证器奖励 / 偏好 / 学习模型。包含鲁棒性方案。
5. 诊断指标。相对于基线的胜率、ELO 曲线、验证器通过率、相对于参考的 KL。

拒绝在不信息游戏上使用 AlphaZero（转 CFR）。拒绝在无可信验证器时使用 GRPO。拒绝在没有固定基线对手集的情况下使用任何游戏 RL 流程（否则自对弈 ELO 无法校准）。
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO 赌博机。在 2 个提示 × 每个 4 个答案 token 上训练。在 `G=8` 时 1,000 步更新内收敛。
2. **中等。** 接入 PPO（裁剪版）和普通 REINFORCE。在同一赌博机上比较样本效率和奖励方差与 GRPO 的差异。
3. **困难。** 扩展到长度 2 的"推理链"：智能体发出两个 token，验证器对配对给予奖励。衡量 GRPO 如何处理两步序列上的信用分配。（提示：按**完整序列**计算组内优势，传播到两个 token 位置。）

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| MCTS | "带学习网络的树搜索" | 蒙特卡洛树搜索；带学习 `(p, v)` 先验的 UCB1/PUCT 选择。 |
| AlphaZero | "自对弈 + MCTS" | 策略-价值网络训练以匹配 MCTS 访问和博弈结果。 |
| MuZero | "带学习模型的 AlphaZero" | 相同循环但在潜在空间中通过学习的动力学实现。 |
| GRPO | "无价值网络的 PPO" | 组内相对策略优化；带组均值基线和 KL 的 REINFORCE。 |
| PUCT | "AlphaZero 的 UCB" | `Q + c · p · √N / (1 + N_a)`——平衡价值估计与先验。 |
| 自对弈 | "智能体对抗过去的自己" | 零和博弈的标准；对称训练信号。 |
| 联赛模式 | "基于种群的自对弈" | 对手从历史 + 当前 + 利用者中采样。 |
| 验证器奖励 | "可验证 RL" | 奖励来自确定性检查器（测试通过、答案匹配）。 |
| 过程奖励 | "PRM" | 对每个推理步骤打分，不仅看最终答案。 |

## 延伸阅读

- [Silver 等人 (2017)。Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)。
- [Silver 等人 (2018)。A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser 等人 (2020)。Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals 等人 (2019)。Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024)。DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300)——引入 GRPO 和组内基线的论文。
- [DeepSeek-AI (2025)。DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)——完整的四阶段 R1 配方加上 R1-Zero 消融。
- [Brown 等人 (2019)。Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400)——大规模 CFR + 深度学习。
- [Tesauro (1995)。Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343)——开创一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)——使用自定义奖励函数应用 GRPO 的生产参考。
- [Qwen Team (2024)。Qwen2.5-Math — GRPO 复现](https://github.com/QwenLM/Qwen2.5-Math)——多尺度的 R1 配方开源复现。
- [Sutton & Barto (2018)。第 17 章——强化学习的前沿](http://incompleteideas.net/book/RLbook2020.pdf)——自对弈、搜索和"设计奖励"的教科书框架，R1 在 LLM 规模上实例化了这一框架。
