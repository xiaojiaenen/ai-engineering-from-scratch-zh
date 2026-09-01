# 多Token预测 (MTP)

> 从 GPT-2 到 Llama 3，每一个自回归 LLM 在每个位置都只训练一个损失：预测下一个 token。DeepSeek-V3 在每个位置增加了第二个损失：预测再下一个 token。额外的 140 亿参数（在 6710 亿参数的模型上）通过梯度流被蒸馏回主模型，训练好的 MTP head 在推理时被重新用作推测解码的草稿生成器，接受率达到 80% 以上。1.8 倍的生成吞吐量就这样免费获得了。本教程将基于 DeepSeek 技术报告构建序列式 MTP 模块，计算损失与共享 head 的参数布局，并解释 MTP 如何保持因果链，而 Gloeckle 等人原始的并行 MTP 为何打破了它。

**类型：** 构建
**语言：** Python (标准库)
**前置要求：** 第 10 阶段 · 04（预训练迷你 GPT），第 10 阶段 · 15（推测解码）
**时间：** 约 60 分钟

## 学习目标

- 阐述 MTP 训练目标，并推导跨预测深度的联合损失。
- 解释 Gloeckle 等人（2024）的并行 MTP head 与 DeepSeek-V3 的序列式 MTP 模块之间的差异，以及为何序列式设计能保留因果链。
- 计算向预训练流程添加 MTP 模块所产生的参数与显存开销。
- 从零实现一个 MTP 模块：共享 embedding、各深度 transformer 块、投影层以及共享输出 head。

## 问题背景

下一个 token 预测是 LLM 的标准训练目标。每个 hidden state 都被监督去预测恰好一件事：紧随其后的那个 token。这是一个出乎意料地微弱的信号。序列中的大部分信息都超越了一个 token 的范畴——结构、连贯性、事实准确性、算术逻辑流。模型必须通过累计数万亿 token 上的大量单 token 信号来学习这些。

MTP 提出一个问题：如果每个 hidden state 被监督同时预测多个未来 token 会怎样？Gloeckle 等人（Meta，2024）证明了这确实有帮助。他们的实现方式是在 backbone 之上放置多个独立的输出 head，每个 head 预测不同的偏移量。并行、简单，但各 head 看到的是相同的 hidden state，没有任何层级上的细化——且预测之间不存在因果依赖链，因此无法用于推测解码。

DeepSeek-V3（2024年12月）将 MTP 重新设计为序列式模块，在每个预测深度上保持因果链。模型从 `h_i^(0)` 预测 `t+1`，然后从一个新的 hidden state `h_i^(1)` 预测 `t+2`，该状态融合了 `h_i^(0)` 与 `E(t+1)` 的 embedding，依此类推。每个深度都是独立的小型 transformer 块。共享的 embedding 和共享的输出 head 将参数开销控制在合理范围。在 DeepSeek-V3 的规模下，MTP 模块在 6710 亿主模型权重之上额外增加了 140 亿参数。这 2% 的开销换来了更密集的训练信号，以及在推理时现成的推测解码草稿。

本教程将从零构建单个 MTP 模块与 D 层深度损失。数学推导简洁清晰。实现代码约 150 行。

## 核心概念

### 序列式 MTP 实现方案

DeepSeek-V3 在主模型之上添加了 `D` 个 MTP 模块。每个模块 `k`（`k = 1..D`）预测深度 `k` 处的 token——即在 position `i` 之前的前缀条件下预测 `t_{i+k}`。

模块 `k` 由以下部分组成：

- 具有独立 attention 和 MLP 的 transformer 块 `T_k`。
- 投影矩阵 `M_k`，用于融合前一深度的 hidden state 与下一深度 ground truth token 的 embedding。
- 共享 embedding `E`（与主模型相同）。
- 共享输出 head `Out`（与主模型相同）。

在训练时，对于到 position `i` 的前缀，各深度的 hidden state 为：

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

各深度的预测为：

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

各深度损失是对 ground truth `t_{i+k}` 的交叉熵：

```
L_k = CE(logits_{i+k}, t_{i+k})
```

跨深度的联合损失：

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` 是一个较小的加权系数——DeepSeek-V3 在前 10% 训练阶段使用 0.3，之后使用 0.1。总训练损失为 `L_main + L_MTP`。

### 为何是序列式而非并行式

Gloeckle 原始的并行 MTP 包含 D 个输出 head，每个都直接作用于 `h_i^(0)`。每个 head 都从相同的 backbone hidden state 预测 `t_{i+k}`。这种方案训练起来没问题，但预测之间互不依赖。你无法用 `head_1` 的输出辅助 `head_2`——各 head 是并行触发的。

DeepSeek-V3 的序列式设计从 `h_i^(k-1)` 加上实际的下一 token embedding `E(t_{i+k})` 构建 `h_i^(k)`。这保留了因果链：要预测 `t_{i+k+1}`，深度 `k+1` 的模块可以看到 `t_{i+k}` 处的内容。这在结构上与自回归解码器消费自身输出的过程完全一致——使得 MTP 模块可以直接用作推测解码的草稿生成器。

在推理时：将 `h_i^(k-1)` 和草稿生成的 `t_{i+k}` 输入模块 `k+1`，得到对 `t_{i+k+1}` 的预测。重复此过程。这正是 EAGLE 风格的草稿生成流程，将训练好的 MTP 模块作为草稿网络使用。DeepSeek-V3 报告第一个 MTP 模块的接受率超过 80%，速度提升约 1.8 倍。

### 参数核算

对于一个 hidden dimension 为 `h`、词表大小为 `V` 的模型：

- 主模型：数十亿参数，外加一个大小为 `V * h` 的输出 head。
- 共享输出 head：复用主模型的 head。无额外参数。
- 共享 embedding：复用主模型的 embedding。无额外参数。
- 单个 MTP 模块：
  - 投影 `M_k`：`(2h) * h = 2h^2`。
  - Transformer 块 `T_k`：attention（MHA 为 `4h^2`）加上 MLP（SwiGLU 比例 8/3 时为 `8h^2`）。每个块约 `12h^2`。

每个模块的额外参数总计：`~14h^2`。对于 DeepSeek-V3 的 `h = 7168`，D = 1 个模块：理论上 `~14 * 7168^2 ≈ 7.2 亿` 参数。但 DeepSeek-V3 报告的是 140 亿——差异主要在于 MTP 模块中的 expert 层也采用了 MoE 结构。

### 推测解码的收益

在预训练期间，MTP 模块会使训练速度降低约 10%（更多前向计算与额外损失）。但收益是双重的：

1. 更密集的训练信号。每个 hidden state 面对 D+1 个监督目标。在 MMLU、GSM8K、MATH、HumanEval 上的实测效果：DeepSeek-V3 的消融实验显示稳定提升数个百分比点。

2. 推理时免费获得推测解码草稿。MTP 模块已经被训练用于预测接下来几个 token。将其重新用作草稿网络后，可实现 80% 以上的接受率。达到这一水平时，N=3 或 N=5 的推测解码可带来 1.8 倍的吞吐量提升。这 10% 的训练时间成本在第一次运行推理时即可收回。

### 与 EAGLE 的关系

EAGLE 在预训练完成后**单独**训练一个小型草稿模型。MTP 则将草稿生成能力**内嵌**到预训练中。这两种方法最终收敛到相似的接受率，但走的是不同的技术路线：

| 维度 | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| 训练时机 | 预训练后 | 预训练期间 |
| 与现有权重向后兼容 | 是 | 否（需重新训练） |
| 草稿参数 | 1-2 层 transformer | 1 个 transformer 块 + 投影层 |
| 接受率 | 0.88-0.92 | 深度 1 时 >0.80 |
| 除加速外的收益 | 仅限推测解码 | 更密集的训练信号 + 加速 |

```figure
multi-token-predict
```

## 动手实现

`code/main.py` 端到端构建单个 MTP 模块：共享 embedding、投影层、transformer 块、共享输出头。随后在一段简短的合成序列上计算各深度的交叉熵损失，并按组件打印参数数量。使用 32 个 token 的玩具词表以保持数值可读。

### 步骤 1：共享 embedding 表

一个单独的 `vocab_size x hidden` 表由主模型**以及**每个深度上的每个 MTP 模块共同使用。不是第二份副本——字面意义上是同一个张量。

### 步骤 2：逐深度融合

```python
def combine(prev_hidden, next_token_embed, M_k):
    # 沿特征维度拼接，然后投影回 hidden 维度
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # 用向量加法作为占位
    projected = matvec(M_k, concat)
    return projected
```

真实的 DeepSeek-V3 将两个 RMSNorm 向量拼接为 `[2h]`，再用 `h x 2h` 矩阵进行投影。本玩具实现为了标准库简洁起见使用了向量加法。

### 步骤 3：深度 k 的 transformer 块

自注意力加上 MLP。在玩具实现中，单层线性注意力块与 SwiGLU MLP 在不依赖 numpy 的情况下保持了结构清晰可见。

### 步骤 4：共享输出头

复用主模型的输出投影。输出词表维度的 logits。

### 步骤 5：逐深度损失

softmax(logits) 与偏移 `k` 处的 ground truth token 之间的交叉熵。使用 `lambda / D` 缩放因子在各深度间聚合。

### 步骤 6：参数核算

打印总参数数量、共享部分（embedding、head）的数量以及每个模块的额外数量。展示 MTP 额外参数相对于主模型规模的占比。

## 应用场景

MTP 已集成于 DeepSeek-V3（2024年12月）与 DeepSeek-R1 系列中。在推理时：

- DeepSeek 自有的推理服务栈开箱即用地将 MTP 模块作为推测解码器使用。
- 截至 2026 年 4 月，vLLM 与 SGLang 已具备 DeepSeek-V3 MTP 的集成路径。
- AMD 的 ROCm SGLang 教程展示了一种特定的 MTP 推测解码配置，在 V3 checkpoint 上实测速度提升 1.8 倍。

在新预训练流程中使用 MTP 的适用场景：

- 你掌控完整的预训练流水线，并希望换取更密集的训练信号。
- 你明确知道将在大规模环境下部署该模型，并希望免费获得推测解码能力。
- 你的 hidden size 至少为 4096。在 1B 参数规模下，开销带来的负面影响会大于收益。

不适用场景：

- 微调已有的预训练稠密模型。MTP 模块并未在此类模型上训练。
- 需要干净基线用于对比的研究模型。MTP 会改变模型架构。

## 交付成果

本教程将产出 `outputs/skill-mtp-planner.md`。给定预训练流程规格（模型规模、数据、算力），它会返回一份 MTP 集成计划：深度数量 D、`lambda` 调度策略、显存开销，以及推理时的推测解码接线方案。

## 练习

1. 运行 `code/main.py`。展示随着合成信号增强，各深度损失单调下降。修改合成数据以使用固定模式，并验证深度 1 与深度 2 的损失均收敛。

2. 计算 D=1 个 MTP 模块应用于稠密 70B 模型（hidden 8192，80 层）时的参数开销。与 DeepSeek-V3 报告的 140 亿额外参数进行对比。解释为何 DeepSeek 的数值更高：MTP transformer 块继承了相同的 MoE 结构，放大了每个模块的参数数量。

3. 在玩具实现中支持 D=2：添加第二个 MTP 模块，以 h^(1) 为输入并预测 `t_{i+2}`。验证联合损失与参数核算是否符合 DeepSeek 论文中的公式 19-21。

4. 将玩具实现切换为并行 MTP（Gloeckle 风格）：在主 hidden state 之上添加 D 个输出 head，每个 head 预测不同的偏移量。在相同合成信号下，测量各深度损失与序列式版本的对比。序列式版本在 k > 1 时应产生更低的深度 k 损失，因为它会依赖中间预测结果。

5. 将训练好的 MTP 模块用作 EAGLE 风格的草稿网络：在推理时调用模块 k 以提议 `t_{i+k}`。在留出序列上，测量这些草稿 token 相对于主模型预测的接受率。若在玩具任务上达到 50% 以上的接受率，则说明你已成功复现了"MTP 作为草稿"的经验特性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MTP 模块 | “额外损失块” | 一个小尺寸 transformer 块加投影层，用于预测主模型超前 `k` 个位置的 token |
| 预测深度 | “哪个偏移量” | 整数 `k`，表示模块 `k` 从到 position `i` 为止的前缀中预测 `t_{i+k}` |
| 并行 MTP | “Gloeckle 风格” | 在相同 backbone hidden state 上部署 D 个独立 head，无条件依赖链 |
| 序列式 MTP | “DeepSeek-V3 风格” | 每个模块均依赖前一深度的 hidden state 加上下一个 token 的 embedding；保留因果链 |
| 共享输出头 | “复用主 head” | MTP 模块直接调用主模型的 LM head，而非独立的输出投影 |
| 共享 embedding | “复用主词表” | 全局使用同一份词汇 embedding 表；无重复参数 |
| 投影矩阵 M_k | “融合 hidden + 下一 token” | 一个 `h x 2h` 线性层，将前一 hidden state 与目标 token embedding 折叠为下一深度的输入 |
| 联合损失 L_MTP | “平均额外损失” | 各深度交叉熵损失的算术平均值，乘以 `lambda` 缩放 |
| 深度 1 接受率 | “MTP 草稿对了多少次” | D=1 MTP 模块 top-1 预测与主模型 top-1 预测一致的比例；DeepSeek-V3 上超过 80% |
| Lambda 加权系数 | “额外损失的重要性” | 各深度的缩放因子；DeepSeek-V3 在训练初期为 0.3，后期为 0.1 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) —— 完整的序列式 MTP 描述（第 2.2 节），包含联合损失公式与推理时 1.8 倍加速的说明
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) —— DeepSeek 设计所改进的并行 MTP 基线
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) —— 总计 6850 亿参数（6710 亿主模型 + 140 亿 MTP），含部署说明
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) —— MTP 所属的推测解码框架
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) —— EAGLE 2025 年的草稿架构，即 MTP 的直接竞争方案
