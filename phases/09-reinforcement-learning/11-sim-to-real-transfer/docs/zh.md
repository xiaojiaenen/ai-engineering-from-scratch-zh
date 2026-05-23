# 仿真到现实的迁移

> 在仿真器中训练但在硬件上失败的策略，本质上是记忆了仿真器的策略。域随机化、域适应和系统辨识是让学习到的控制器跨越现实差距的三大工具。

**类型：** 学习
**语言：** Python
**先决条件：** 阶段9 · 08 (PPO)， 阶段2 · 10 (偏差/方差)
**时间：** ~45 分钟

## 问题

训练真实的机器人缓慢、危险且昂贵。双足机器人需要数百万个训练轮次才能学会行走；而真实的双足机器人哪怕只跌倒一次就可能损坏硬件。仿真为你提供了无限次重置、确定性可复现性、并行环境且不会造成物理损坏。

但仿真器是错误的。轴承的摩擦力比MuJoCo模型中的要大。相机存在仿真器未包含的镜头畸变。电机具有延迟、齿隙和饱和现象，而99%的仿真模型都忽略了这些。风力、灰尘和变化的光照条件会破坏在无菌渲染环境下训练的策略。**现实差距**——仿真分布与真实分布之间的系统性差异——是机器人强化学习部署的核心问题。

你需要一个对仿真到现实分布偏移*鲁棒的*策略。历史上有三种方法：使仿真器随机化（域随机化）、用少量真实数据适应策略（域适应/微调）、或辨识真实系统的参数并进行匹配（系统辨识）。到2026年，主导的配方是将这三者与大规模并行仿真（Isaac Sim, Isaac Lab, Mujoco MJX on GPU）相结合。

## 概念

![三种仿真到现实的范式：域随机化、域适应、系统辨识](../assets/sim-to-real.svg)

**域随机化 (DR)。** Tobin等人 (2017)，Peng等人 (2018)。在训练过程中，随机化每个可能与真实机器人不同的仿真参数：质量、摩擦系数、电机PD增益、传感器噪声、相机位置、光照、纹理、接触模型。策略学习一个关于"今天它处于哪种仿真中"的条件分布，并在整个范围内泛化。如果真实机器人处于训练包络内，策略就能工作。

- **优点：** 不需要真实数据。一套方案，多种机器人。
- **缺点：** 过度随机化训练会产生"通用"但过于谨慎的策略。噪声过多 ≈ 正则化过多。

**系统辨识 (SI)。** 在训练前，将仿真器的参数拟合到真实世界的数据。如果你能测量真实机器人手臂关节的摩擦力，就将其代入仿真中。然后训练一个期望这些数值的策略。需要访问真实系统，但能直接减少现实差距。

- **优点：** 精确、低噪声的训练目标。
- **缺点：** 残留的模型误差对策略来说是不可见的；微小的未辨识效应（例如电机死区）仍然会在部署时导致失败。

**域适应。** 在仿真中训练，用少量真实数据进行微调。两种风格：

- **Real2Sim2Real：** 使用真实轨迹学习一个残差仿真器 `f(s, a, z) - f_sim(s, a)`，在修正后的仿真中进行训练。在无需大量真实数据的情况下缩小差距。
- **观察适应：** 训练一个策略，通过一个学习的特征提取器（例如GAN像素到像素映射）将真实观测映射到类似仿真的观测。控制器仍保持在仿真中。

**特权学习 / 教师-学生。** Miki等人 (2022) (ANYmal四足机器人)。在仿真中训练一个*教师*策略，该策略可以访问特权信息（地面真实摩擦、地形高度、IMU漂移）。蒸馏出一个*学生*策略，该策略只看到真实传感器的观测。学生学会从历史观测中推断特权特征，对物理参数具有鲁棒性。

**大规模并行仿真。** 2024–2026年。Isaac Lab, Mujoco MJX, Brax 都能在单块GPU上运行成千上万的并行机器人。使用4,096个并行类人机器人进行PPO训练，几小时内就能收集到数年的经验。随着训练分布的扩大，"现实差距"在缩小；当这4,096个环境中的每一个都具有不同的随机化参数时，DR几乎变得免费。

**2026年的现实世界配方（以四足行走为例）：**

1. 在域随机化重力、摩擦、电机增益、负载的大规模并行仿真中进行。
2. 使用特权信息（地形图、身体速度地面真值）训练教师策略。
3. 仅使用本体感觉（腿部关节编码器）从教师蒸馏学生策略。
4. 通过真实IMU上的自编码器进行可选的观察适应。
5. 部署。在10种以上环境零样本测试。如果失败，使用安全约束下的PPO进行几分钟的真实世界微调。

## 构建它

本课的代码是在一个具有*噪声*转移的网格世界上进行域随机化的一个小型演示。我们训练一个在"仿真"中经历随机化滑动概率的策略，并在"真实"环境中评估，该环境使用了训练期间从未见过的滑动水平。其形态直接映射到MuJoCo到硬件的迁移。

### 步骤1：参数化仿真

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是仿真器暴露的一个参数。在实际机器人学中，它可以是摩擦力、质量、电机增益——任何在仿真和真实之间会发生偏移的东西。

### 步骤2：使用DR进行训练

在每个轮次开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练PPO/Q-learning/任何算法。进行多个轮次。

### 步骤3：在"真实"滑动下零样本评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上进行评估。前四个在训练支持范围内；`0.5` 和 `0.7` 在其外。经过DR训练的策略在支持范围内应保持接近最优，在其外应优雅地退化。在固定滑动下训练的策略在其训练滑动之外会非常脆弱。

### 步骤4：与狭窄训练进行比较

仅使用 `slip = 0.0` 训练第二个策略。在相同的 `slip` 范围上进行评估。你应该会看到，一旦真实滑动 > 0，就会出现灾难性的性能下降。

## 陷阱

- **过度随机化。** 在 `slip ∈ [0, 0.9]` 上训练，你的策略会变得非常风险规避，以至于从不尝试最优路径。匹配*预期的*真实世界分布，而不是"任何事情都可能发生"。
- **随机化不足。** 在狭窄的范围内训练，策略完全无法泛化。使用自适应课程（自动域随机化），随着策略改进扩大分布。
- **误识别的参数空间。** 随机化错误的东西（当真实差距是电机延迟时却随机化相机色调），DR不会起作用。首先对真实机器人进行分析。
- **特权信息泄露。** 一个教师策略如果使用全局状态（而不仅仅是观测）来执行动作，会产生一个无法追赶的学生策略。确保教师的策略在给定观测历史的情况下对学生来说是可实现的。
- **仿真到仿真迁移失败。** 如果你的策略对更难的仿真变体不鲁棒，那么它对真实世界也不会鲁棒。在部署前总是在一个保留的仿真变体上进行测试。
- **缺乏真实世界安全包络。** 一个在仿真中有效、在"真实"中也有效但没有底层安全屏蔽的策略，仍然可能损坏硬件。在一个非学习的控制器中添加速率限制、扭矩限制、关节限制。

## 使用它

2026年的仿真到现实技术栈：

| 领域 | 技术栈 |
|------|--------|
| 腿足运动 (ANYmal, Spot, 类人) | Isaac Lab + DR + 特权教师/学生 |
| 操作 (灵巧手, 抓取放置) | Isaac Lab + DR + 用于视觉的DR-GAN |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + 真实微调 |
| 无人机竞速 | RotorS / Flightmare + DR + 在线适应 |
| 手指/手内操作 | OpenAI Dactyl (前所未有的大规模DR) |
| 工业机械臂 | MuJoCo-Warp + SI + 少量真实微调 |

对于所有尺度的控制，工作流程是一致的：尽可能好地拟合仿真，随机化你无法拟合的部分，训练庞大的策略，蒸馏，部署时加上安全屏蔽。

## 保存它

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 练习

1. **简单。** 在固定滑动的GridWorld (slip=0.0) 上训练一个Q-learning智能体。在滑动 ∈ {0.0, 0.1, 0.3, 0.5} 上进行评估。绘制回报与滑动的关系图。
2. **中等。** 训练一个采样 `slip ~ Uniform[0, 0.3]` 的DR Q-learning智能体。评估相同的滑动范围。DR在滑动=0.5（分布外）时带来了多少收益？
3. **困难。** 实现一个课程：从slip=0.0开始，每当策略达到最优的90%时就扩大DR范围。测量达到零样本处理slip=0.3所需的总环境步数，并与固定DR基线进行比较。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 现实差距 | "仿真到现实的差异" | 训练与部署物理/感知之间的分布偏移。 |
| 域随机化 (DR) | "在随机仿真中训练" | 在训练期间随机化仿真参数，使策略能够泛化。 |
| 系统辨识 (SI) | "测量真实并拟合仿真" | 估计真实的物理参数；将仿真设置为匹配。 |
| 域适应 | "在真实数据上微调" | 仿真训练后在真实世界进行小规模微调；可能适应观测或动力学。 |
| 特权信息 | "教师的地面真值" | 只有仿真拥有的信息；学生必须从观测历史中推断它。 |
| 教师/学生 | "蒸馏特权->可观察" | 教师使用捷径训练；学生学会在没有这些捷径的情况下模仿。 |
| ADR | "自动域随机化" | 一种随着策略改进而扩大DR范围的课程。 |
| Real2Sim | "用真实数据缩小差距" | 学习一个残差以使仿真模仿真实轨迹。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — DR的开创性论文（机器人视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — 动力学DR，四足运动。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl，大规模ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal的教师-学生方法。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 推动2025–2026年部署的大规模并行仿真。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR课程方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Dyna框架（使用模型进行规划+轨迹），这是现代仿真到现实流水线的基础。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — 仿真到现实方法的分类学及基准测试结果。