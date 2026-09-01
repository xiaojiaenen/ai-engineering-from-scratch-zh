# GPT — 因果语言建模

> BERT 能看到两侧。GPT 只能看到过去。三角形掩码是现代 AI 中最关键的一行代码。

**类型：** 构建
**语言：** Python
**前置知识：** 第7阶段 · 02（自注意力），第7阶段 · 05（完整 Transformer），第7阶段 · 06（BERT）
**时间：** 约75分钟

## 问题

语言模型回答一个问题：给定前 `t-1` 个 token，token `t` 的概率分布是什么？对这个信号进行训练——下一个 token 预测——就能得到一个能一次一个 token 生成任意文本的模型。

为了对整个序列并行地进行端到端训练，每个位置的预测必须只依赖于更早的位置。否则模型会通过窥视答案来作弊。

因果掩码做到了这一点。它是一个单一的 `-inf` 值上三角矩阵，在 softmax 之前加到注意力分数上。softmax 之后，这些位置的权重变为 0。每个位置只能关注自身及更早的位置。而且因为一次应用到整个序列，你得到 N 个并行的下一个 token 预测，只需一次前向传递。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2025）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是具有相同核心循环的仅解码器因果 Transformer。将它们区分开的是数据质量、规模、架构改进以及后训练（SFT、RLHF、DPO 及其后继方法）。

## 概念

![因果掩码创建三角形注意力矩阵](../assets/causal-attention.svg)

### 掩码

给定长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```
M[i, j] = 0       如果 j <= i
M[i, j] = -inf    如果 j > i
```

在 softmax 之前将 `M` 加到原始注意力分数上。`exp(-inf) = 0`，所以被掩码的位置贡献零权重。注意力矩阵的每一行都只是对之前位置的概率分布。

实现成本：一次 `torch.tril()` 调用。计算时间：纳秒级。对领域的影响：一切。

### 三角形的来源

掩码通常被呈现为打在注意力上的补丁。换个方向推导，它就变得不再神秘：注意力是前缀平均的第三次改进，三角形就是那个平均值的循环边界以矩阵形式写出的样子。

**阶段 1——前缀平均。** 序列最朴素的因果摘要：位置 `i` 变为位置 `0…i` 的均值。用循环表示就是 `out[i] = X[:i+1].mean(0)`。同样的计算是一次矩阵乘法。取一个全 1 的下三角矩阵，每行除以其元素个数，然后相乘：

```python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
```

`A` 的第 `i` 行是 `[1/(i+1), …, 1/(i+1), 0, …, 0]`。对角线以上的零就是因果性。并不是未来被屏蔽了；未来从未出现在求和中。

**阶段 2——学习权重。** 均匀平均将所有过去的 token 视为同等重要。用学习到的分数矩阵 `S` 替换全 1。现在行不再天然和为 1，所以用 softmax 归一化每行而不是除以计数。softmax 永远不会输出精确的零，这会破坏因果性——除非将未来分数设为 `-inf`，因为 `exp(-inf) = 0`：

```python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
```

同样的三角形，同样的行随机矩阵，同样的单次矩阵乘法。`-inf` 掩码不是新机器。它是阶段 1 的零条目，转换到 softmax 的输入域。

**阶段 3——内容依赖权重。** 在阶段 2 中，`S` 训练后固定不变：位置 7 始终以相同方式加权位置 3，不管 token 是什么。让分数依赖于 token 本身：`S = Q @ K.T / sqrt(d_k)`。其他什么都不变。掩码、softmax、矩阵乘法——完全相同。

三个阶段，一个不变量：下三角行随机矩阵乘以序列。均匀平均、学习到的静态权重、内容依赖权重。掩码从未被添加到注意力上。它从平均值中存活下来。

```figure
mask-derivation
```

### 并行训练，串行推理

训练：一次性前向传递整个 `(N, d_model)` 序列，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。沿序列维度并行。这就是 GPT 训练能扩展的原因——你在一个 GPU 批次中一次处理 1M token。

推理：你逐个 token 生成。喂入 `[t1, t2, t3]`，得到 `t4`。喂入 `[t1, t2, t3, t4]`，得到 `t5`。喂入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV 缓存（第12课）保存 `t1…tn` 的隐藏状态，避免每一步重新计算。但推理时的串行深度 = 输出长度。这就是自回归税，也是为什么解码是每个 LLM 的延迟瓶颈。

### 损失函数——偏移一位

给定 token `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这就是整个序列的交叉熵。

你听说过的每个 Transformer LM 都在训练这种损失。预训练、微调、SFT——相同的损失，不同的数据。

### 解码策略

训练之后，采样选择的重要性被人低估了。

| 方法 | 做什么 | 何时使用 |
|------|--------|----------|
| 贪婪 | 每步取 argmax | 确定性任务、代码补全 |
| 温度 |  logits 除以 T，采样 | 创意任务，更高 T = 更多样 |
| top-k | 仅从 top-k token 中采样 | 消灭低概率尾部 |
| top-p（核采样） | 从累积概率 ≥ p 的最小集合中采样 | 2020+ 默认；自适应分布形状 |
| min-p | 保留满足 `p > min_p * max_p` 的 token | 2024+；比 top-p 更好地拒绝长尾 |
| 投机解码 | 草稿模型提议 N 个 token，大模型验证 | 同等质量下延迟降低 2–3 倍 |

2026年，min-p + 温度 0.7 是开放权重模型的合理默认值。投机解码是任何生产推理栈的入场券。

### 什么让"GPT 配方"生效

1. **仅解码器。** 无编码器开销。每层一次注意力 + FFN 传递。
2. **规模化。** 124M → 1.5B → 175B → 万亿。Chinchilla 缩放定律（第13课）告诉你如何分配算力。
3. **上下文学习。** 大约在 6B–13B 时涌现。模型无需微调就能跟随少样本示例。
4. **RLHF。** 在人类偏好上后训练将原始预训练文本转化为聊天助手。
5. **预归一化 + RoPE + SwiGLU。** 大规模稳定训练。

核心架构自 GPT-2 以来变化不大。所有有趣的事都发生在数据、规模和后训练中。

```figure
causal-mask
```

## 构建

### 步骤 1：因果掩码

见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前将它加到注意力分数上。这就是整个机制。

### 步骤 2：两层 GPT 式模型

堆叠两个解码器块（掩码自注意力 + FFN，无交叉注意力）。添加 token 嵌入、位置编码和 unembedding（与 token 嵌入矩阵共享——自 GPT-2 以来的标准技巧）。

### 步骤 3：端到端下一个 token 预测

在 20 token 的玩具词表上，在每个位置生成 logits。对照偏移一位的目标计算交叉熵损失。不需要梯度——这是前向传递的合理性检查。

### 步骤 4：采样

实现贪婪、温度、top-k、top-p、min-p。在固定 prompt 上运行每种方法并比较输出。采样函数 10 行搞定。

## 使用

PyTorch，2026 惯用法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 运行前向传递，取出最终位置的 logits，采样下一个 token，追加它，然后重复。每个生产 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都以大量优化实现相同的循环——批量预填充、连续批处理、KV 缓存分页、投机解码。

**GPT 与 BERT，各一行：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。损失函数决定模型能否生成。

## 交付

见 `outputs/skill-sampling-tuner.md`。该技能为新生成任务选择采样参数，并标记何时需要确定性解码。

## 练习

1. **简单。** 运行 `code/main.py` 并验证 softmax 后因果注意力矩阵是下三角的。抽查：第 3 行应只在列 0–3 有权重。
2. **中等。** 实现宽度为 4 的束搜索。比较束宽 4 与贪婪在 10 个短 prompt 上的困惑度。束搜索总是赢吗？（提示：通常对翻译赢，对开放式聊天不赢。）
3. **困难。** 实现投机解码：用小型 2 层模型作为草稿，用 6 层模型作为验证器。在 100 次长度为 64 的补全上测量墙钟加速。确认输出与验证器的贪婪输出匹配。

## 关键术语

| 术语 | 人们怎么说 | 实际上是什么意思 |
|------|-----------|-----------------|
| 因果掩码 | "那个三角形" | 加到注意力分数上的上三角 `-inf` 矩阵，使位置 `i` 只能看到 `≤ i` 的位置。 |
| 下一个 token 预测 | "损失" | 模型分布与每个位置真实下一个 token 之间的交叉熵。 |
| 自回归 | "一个一个生成" | 将输出反馈为输入；只在训练时并行，不在生成时并行。 |
| Logits | "softmax 前的分数" | LM 头在 softmax 之前的原始输出；采样在这些上发生。 |
| 温度 | "创意旋钮" | logits 除以 T；T→0 = 贪婪，T→∞ = 均匀。 |
| Top-p | "核采样" | 将分布截断到总和 ≥p 的最小集合；对剩余部分采样。 |
| Min-p | "比 top-p 更好" | 保留满足 `p ≥ min_p × max_p` 的 token；根据分布尖锐度自适应截止点。 |
| 投机解码 | "草稿 + 验证" | 便宜模型提议 N 个 token；大模型并行验证。 |
| 教师强制 | "训练技巧" | 训练时喂入真实的前一个 token，而非模型预测。每个 seq2seq LM 的标准做法。 |

## 延伸阅读

- [Radford 等（2018）。通过生成式预训练提升语言理解](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford 等（2019）。语言模型是无监督多任务学习者](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown 等（2020）。语言模型是少样本学习者](https://arxiv.org/abs/2005.14165) — GPT-3 和上下文学习。
- [Leviathan、Kalman、Matias（2023）。通过投机解码实现 Transformer 快速推理](https://arxiv.org/abs/2211.17192) — 投机解码论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 标准的因果 LM 参考代码。
