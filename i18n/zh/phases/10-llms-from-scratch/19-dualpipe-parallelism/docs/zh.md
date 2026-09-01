```markdown
# DualPipe 并行

> DeepSeek-V3 在 2,048 块 H800 GPU 上训练，MoE 专家分散在各个节点上。跨节点专家 all-to-all 通信成本为：每 1 GPU 小时的计算就对应 1 GPU 小时的通信开销。GPU 有一半时间是空闲的。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线，它将前向和后向计算与它们触发的 all-to-all 通信重叠。气泡减少了，吞吐量上升了，而保留两份模型参数副本（"dual" 即由此得名）的开销，在专家并行已经跨 rank 分散专家的情况下是微不足道的。本课以 Learn 类型带你逐步了解 DualPipe 的实际机制及其设计原理，以及 Sea AI Lab 的 DualPipeV 改进如何在略微增加气泡的情况下消除这份 2x 参数成本。

**类型：** Learn
**语言：** Python（标准库，调度模拟器）
**前置知识：** 阶段 10 · 05（分布式训练、FSDP、DeepSpeed）、阶段 10 · 14（开放模型架构与 MoE）
**预计时间：** 约 60 分钟

## 学习目标

- 说出 DualPipe 前向-后向 chunk 的四个组成部分，以及为什么每个部分都有各自的重叠窗口。
- 解释大规模下的流水线气泡问题，以及"无气泡"在实际场景与营销话术中的差异。
- 手动追踪一个 8 个 PP rank、16 个 micro-batch 的 DualPipe 调度，验证前向与反向流互相填补了对方的空闲槽位。
- 阐述 DualPipeV（Sea AI Lab，2025）所做的权衡：以 EP 未激活时略微更大的气泡为代价，消除了 2x 参数复制开销。

## 问题背景

在 2k H800 GPU 上训练 671B 参数的 MoE 模型会遇到三个叠加瓶颈：

1. **显存压力。** 每块 GPU 持有模型的一个分片。序列长度 8k、61 层、128 个 attention head 的激活显存极其巨大。
2. **流水线气泡。** 传统流水线并行（GPipe、1F1B）会让 GPU 在等待本 stage 的输入或梯度时空转。在 8 个 stage 下，即便使用 1F1B 调度，GPU 时间中约有 12% 也会浪费在气泡上。
3. **跨节点 all-to-all 通信。** 带有专家并行的 MoE 将专家分散到不同节点。每次前向传播都会触发一次 all-to-all 把 token 分发到对应专家，另一次 all-to-all 把专家输出聚合回来。在 2k GPU 规模下，计算与通信的比例很容易达到 1:1。

这三个问题各自有独立解法：梯度检查点解决显存，Zero Bubble（Sea AI Lab，2023）解决流水线气泡，专家并行通信算子解决 all-to-all。DualPipe 的价值在于让它们协同工作：它在一个前向-后向 chunk 内将计算与通信重叠，从流水线两端同时注入 micro-batch，并用这种调度把 all-to-all 隐藏进计算窗口中。

报道结果：近乎消除流水线气泡，在 DeepSeek-V3 14.8T token 的训练中实现超过 95% 的 GPU 利用率。

## 核心概念

### 流水线并行回顾

将 N 层的模型切分到 P 个设备上。设备 `i` 持有第 `i * N/P .. (i+1) * N/P - 1` 层。一个 micro-batch 从前到后流经设备 0 到 P-1，再反向从 P-1 回到 0。每个设备只能在收到前一设备发送的输出后才能开始自己的前向阶段，也只能在收到下游设备发来的上游梯度后才能开始后向。

GPipe（Huang 等，2019）一次只调度一个 micro-batch，浪费了大部分 GPU 时间。1F1B（Narayanan 等，2021）对多个 micro-batch 交错执行前向和后向。Zero Bubble（Qi 等，2023）将后向分为两部分——输入梯度（B）和权重梯度（W）——并以填充气泡的方式调度它们。经过 Zero Bubble 之后，流水线已接近紧凑。

DualPipe 是下一步改进。它在上述基础上引入两个关键思想：

### 思想 1：chunk 分解

每个前向 chunk 拆成四个组件：

- **Attention。** Q/K/V 投影、注意力计算、输出投影。
- **All-to-all dispatch。** 跨节点通信，把 token 发送到对应的专家。
- **MLP。** MoE 专家计算。
- **All-to-all combine。** 跨节点通信，将专家输出聚合回来。

后向 chunk 则对应每个组件的梯度版本。DualPipe 的调度让 all-to-all dispatch 与下一个 chunk 的 attention 计算并行，让 all-to-all combine 与再下一个 chunk 的 MLP 计算并行。

### 思想 2：双向调度

大多数流水线调度从 stage 0 注入 micro-batch 并向 stage P-1 流动。DualPipe 从**两端**同时注入 micro-batch：stage 0 看到从自身出发的前向 micro-batch；stage P-1 也看到从自身出发的前向 micro-batch。两条流在中间汇合。

为了实现这一点，设备 `i` 必须同时持有**早段流水线层 `i`** 与**晚段流水线层 `P - 1 - i`**。这就是 DualPipe 中 "dual" 的来源：每个设备保留两份所需模型的层副本（服务于两个方向）。在 DeepSeek-V3 的规模下，这是 2x 的参数复制成本。但由于专家并行已经把 MoE 专家分散得足够稀疏，重复两份非专家层代价很小。

关键之处在于，一个方向的前向流与另一个方向的后向流正好重叠在单方向调度会出现气泡的位置。气泡由此消失。

### 手动追踪调度示例

考虑 P = 4 个 rank、8 个 micro-batch，分为 4 个前向 / 4 个反向。时间从左到右流动；行对应设备 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

解读 "F4/F5R" 标记：rank 1 在同一时间槽内既执行 micro-batch 4 的前向（在流水线中从左向右），也执行 micro-batch 5 的前向（从右向左）。这就是操作意义上的 "双向"。

在 rank 2，交叉流更早重叠；在 rank 0 和 P-1，重叠最晚。在调度的稳定阶段，每个 rank 都在运行：前向-X 方向 与 后向-Y 方向 的重叠组合。计算持续繁忙。前向 pass 的 all-to-all dispatch 隐藏在反向计算中，all-to-all combine 隐藏在前向计算中。气泡被挤出。

### 气泡核算

标准 1F1B 流水线气泡（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 有所改善，但无法归零。DualPipe 在稳定阶段，当 micro-batch 数量可被 2 倍流水线深度整除时，气泡为零。在稳定阶段之外（warmup 和 cooldown），仍存在少量气泡，但它不随 micro-batch 数量增长——这是论文强调的一个关键性质。

在营销语境中称为 "bubble-free"。在技术语境中则是：气泡不随 micro-batch 数量增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）表明，只有当专家并行不是瓶颈时才能实现完全零气泡；当 EP 驱动的 all-to-all 成为主导因素时，总存在一定的调度妥协。

### DualPipeV —— 改进版

Sea AI Lab（2025）观察到，当 EP 通信重叠并非核心诉求时，2x 参数复制是浪费。他们的 DualPipeV 调度将双向注入折叠为 "V 形" 调度，只需单份参数副本即可运行。气泡比 DualPipe 略大，但显存节省显著。DeepSeek 在他们的开源 DualPipe 实现中将 DualPipeV 作为 EP-off 模式采用。

权衡对比：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|------|---------|-----------|------|------------|
| 每设备参数副本数 | 2 | 1 | 1 | 1 |
| 气泡 vs micro-batch 数量 | 恒定 | 小幅增长 | 增长 | 增长 |
| 计算-通信重叠 | 充分 | 部分 | 最小 | 部分 |
| 适用场景 | EP 重负载 MoE | 稠密模型或 EP 轻负载 | 基线 | 任意流水线 |

### 对 14.8T token 训练的影响

DeepSeek-V3 预训练在约 2.8M GPU 小时内消耗了 14.8T token，使用了 2,048 块 H800 GPU。若使用朴素的 1F1B，他们将因流水线气泡损失 12-15% 的时间——约 340-420K GPU 小时，足够完整训练一个 70B 模型。DualPipe 挽回了大部分损失。在没有内部日志的情况下难以精确量化其贡献，但论文声称平均 GPU 利用率超过 95%。

对于较小规模的训练（少于 1k GPU），DualPipe 有些过度设计——相对于总成本，流水线气泡更小，而稠密模型训练很少触及 all-to-all 瓶颈。但在千卡以上规模的前沿 MoE 训练中，它几乎是必备的。

### 在技术栈中的位置

- 与 **FSDP**（阶段 10 · 05）互补。FSDP 跨 rank 分片模型参数；DualPipe 跨 rank 调度计算。两者结合使用。
- 与 **ZeRO-3** 梯度分片兼容。双副本参数的簿记需要与 ZeRO 的分片梯度协调。
- 需要针对具体集群拓扑调优的**自定义 all-to-all 算子**。DeepSeek 的开源算子即为此参考实现。

```figure
expert-capacity
```

## 动手实践

`code/main.py` 是一个流水线调度模拟器。它接收 `(P, n_micro_batches, schedule)` 并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 四种调度在稳定阶段的利用率。这是一个教学工具——数值与论文中的定性结论一致，并不代表对生产环境实测加速的声明。

模拟器的价值：用不同的 P 和 micro-batch 数量运行，观察 1F1B 的气泡比例如何增长而 DualPipe 不会。

实际训练集成的注意事项：

- 选择能够整除 micro-batch 数量的流水线并行深度。
- 确保你的专家并行 mesh 支持双向 all-to-all。DeepSeek 的算子是参考实现。
- 第一次使用时，预计要花一周调试调度本身。簿记工作相当繁琐。
- 监控每个 rank 的 GPU 利用率，而非仅看 aggregate。DualPipe 的收益正来自收紧那些短板 rank。

## 成果输出

本课产出 `outputs/skill-dualpipe-planner.md`。给定训练集群规格（GPU 数量、拓扑、互联、模型形状），它推荐一种流水线并行策略、所选调度算法以及在目标规模下的预期气泡比例。

## 练习

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 与 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU 利用率的差异，并将其表示为每百万 token 训练中恢复的 GPU 小时数。

2. 手工绘制 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表格。在每个时间槽中标注 micro-batch ID 和方向。找出气泡首次消失的时间槽。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的图 5。找出 DualPipe 前向 chunk 中 all-to-all dispatch 的重叠窗口。解释计算调度是如何将其隐藏的。

4. 计算 P=8 流水线 stage 下 70B 稠密模型的 DualPipe 2x 参数开销，以及 P=16 流水线 stage 下 671B MoE 模型的对应开销。说明为什么 MoE 案例的相对开销更小（大多数参数是专家，分布在大规模 EP 组中）。

5. 将 DualPipe 与 Chimera（2021 年出现的另一种双向调度器）进行对比。以论文第 3.4 节为参考，指出 DualPipe 具备而 Chimera 所没有的两个具体特性。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Pipeline bubble | "每个 rank 的空闲时间" | GPU 周期因流水线 stage 等待输入或梯度而被浪费 |
| 1F1B | "默认流水线调度" | 前向/后向交错调度；DualPipe 的基准对比对象 |
| Zero Bubble | "Sea AI Lab 2023" | 将后向分为 B（输入梯度）和 W（权重梯度）；几乎完全压缩流水线 |
| DualPipe | "DeepSeek-V3 调度" | 双向流水线 + 计算-通信重叠；气泡不随 micro-batch 数量增长 |
| DualPipeV | "Cut-in-half" | V 形改进，以略微更大的气泡为代价消除 2x 参数复制 |
| Chunk | "流水线工作单位" | 一个 micro-batch 经过一个流水线 stage 的前向或后向传递 |
| All-to-all dispatch | "发送 token 到专家" | 跨节点通信，将 token 路由到其对应的 MoE 专家 |
| All-to-all combine | "从专家聚合输出" | 跨节点通信，在 MLP 完成后收集专家输出 |
| Expert Parallelism (EP) | "专家跨 GPU 分布" | 将 MoE 专家跨 rank 分片，不同 GPU 持有不同专家 |
| Pipeline Parallelism (PP) | "层跨 GPU 分布" | 将模型层跨 rank 分片；DualPipe 所调度的维度 |
| Bubble fraction | "浪费的 GPU 时间" | （气泡时间 / 总时间）；DualPipe 将其趋近于零的指标 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)，第 3.3.2 节与图 5](https://arxiv.org/abs/2412.19437) — DualPipe 的主要参考
- [DeepSeek — DualPipe GitHub 仓库](https://github.com/deepseek-ai/DualPipe) — 开源参考实现，包含 DualPipeV (Cut-in-half) 模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241，Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble 的前身工作
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — DualPipeV 分析，启发了 DeepSeek 的 EP-off 模式
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377，2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe 对比的 1F1B 调度
- [Huang et al. — GPipe (arXiv:1811.06965，2018)](https://arxiv.org/abs/1811.06965) — 原始流水线并行论文与气泡问题
```
