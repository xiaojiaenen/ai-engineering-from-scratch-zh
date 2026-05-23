# GPT — 因果语言建模

> BERT 能看到双向语境。GPT 只能看到过去。那个三角形掩码是现代人工智能中最具影响力的一行代码。

**类型：** 构建  
**语言：** Python  
**先修课程：** Phase 7 · 02 (自注意力机制)，Phase 7 · 05 (完整 Transformer)，Phase 7 · 06 (BERT)  
**时间：** ~75 分钟

## 问题

一个语言模型回答一个问题：给定前 `t-1` 个 token，下一个 token `t` 的概率分布是什么？基于这个信号 —— 即下一个 token 预测 —— 进行训练，你就能得到一个可以逐 token 生成任意文本的模型。

要在整条序列上并行地端到端训练模型，你需要每个位置的预测仅依赖于前面的位置。否则，模型会通过直接查看答案而轻易作弊。

因果掩码实现了这一点。它是一个由 `-inf` 值组成的上三角矩阵，在 softmax 之前被加到注意力分数上。经过 softmax 后，被掩码的位置贡献为零权重。每个位置只能关注自身及其之前的位置。并且因为你在整条序列上一次性应用它，所以在一次前向传播中就能并行得到 N 个下一个 token 预测。

GPT-1 (2018)、GPT-2 (2019)、GPT-3 (2020)、GPT-4 (2023)、GPT-5 (2024)、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi —— 它们都是仅解码器的因果 Transformer，拥有相同的核心循环。只是规模更大、数据更好、RLHF 更精良。

## 核心概念

![因果掩码创建三角形注意力矩阵](../assets/causal-attention.svg)

### 掩码

给定一个长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 之前，将 `M` 加到原始注意力分数上。`exp(-inf) = 0`，因此被掩码的位置贡献的权重为零。注意力矩阵的每一行都是仅对先前位置的概率分布。

实现成本：一次 `torch.tril()` 调用。计算时间：纳秒级。对该领域的影响：根本性。

### 并行训练，串行推理

训练：对整条 `(N, d_model)` 序列进行一次前向传播，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。沿序列维度并行。这就是 GPT 训练能够扩展的原因 —— 你可以在一次 GPU 处理中，在一个批次中处理 100 万个 token。

推理：你逐 token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV 缓存（第 12 课）保存了 `t1…tn` 的隐藏状态，这样你就不需要在每一步重新计算它们。但推理时的串行深度 = 输出长度。这就是自回归的代价，也是为什么解码是所有 LLM 的延迟瓶颈。

### 损失函数 —— 移位一位

给定 token `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对于每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这就是整条序列的交叉熵损失。

你听说过的每个 Transformer 语言模型都基于这个损失进行训练。预训练、微调、SFT —— 损失相同，数据不同。

### 解码策略

训练之后，采样选择比人们想象的更重要。

| 方法        | 作用                                             | 何时使用                                 |
|-------------|------------------------------------------------|------------------------------------------|
| 贪心解码    | 每一步都取 argmax                                | 确定性任务，代码补全                     |
| 温度        | 将 logits 除以 T，然后采样                       | 创造性任务，T 越高 = 多样性越高          |
| Top-k       | 仅从 top-k 个 token 中采样                       | 消除低概率长尾                           |
| Top-p（核采样） | 从累积概率 ≥ p 的最小集合中采样                | 2020 年后的默认选择；能适应分布形状       |
| Min-p       | 保留 `p > min_p * max_p` 的 token                        | 2024 年后；比 top-p 更好地拒绝长尾分布   |
| 贪婪解码    | 草稿模型提出 N 个 token，大模型验证              | 在相同质量下延迟降低 2–3 倍              |

在 2026 年，对于开放权重模型，min-p + 温度 0.7 是一个合理的默认值。对于任何生产环境推理堆栈，贪婪解码都是基本要求。

### "GPT 配方" 成功的原因

1.  **仅解码器。** 没有编码器开销。每层仅一次注意力 + FFN 前向传播。
2.  **规模化。** 1.24 亿 → 15 亿 → 1750 亿 → 万亿。Chinchilla 扩展定律（第 13 课）告诉你如何分配算力。
3.  **上下文学习。** 大约在 60 亿至 130 亿参数规模出现。模型可以遵循少量示例而无需微调。
4.  **RLHF。** 基于人类偏好的后训练，将原始的预训练文本转化成了聊天助手。
5.  **Pre-norm + RoPE + SwiGLU。** 实现大规模稳定训练。

自 GPT-2 以来，核心架构变化不大。所有有趣的事情都发生在数据、规模和后训练上。

## 动手构建

### 第 1 步：因果掩码

参见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前将其加到注意力分数上。这就是全部机制。

### 第 2 步：一个两层类 GPT 模型

堆叠两个解码器块（带掩码的自注意力 + FFN，没有交叉注意力）。添加 token 嵌入、位置编码和一个解嵌入（与 token 嵌入矩阵绑定 —— 这是自 GPT-2 以来的标准技巧）。

### 第 3 步：端到端的下一个 token 预测

在一个包含 20 个 token 的玩具词汇表上，在每个位置生成 logits。针对移位一位的目标计算交叉熵损失。不计算梯度 —— 这是一个前向传播的完整性检查。

### 第 4 步：采样

实现贪心解码、温度采样、Top-k、Top-p、Min-p。在固定提示词上运行每种方法并比较输出。一个采样函数大约 10 行代码。

## 使用它

PyTorch, 2026 年写法：

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

在底层，`generate()` 运行前向传播，提取最后一个位置的 logits，采样下一个 token，将其添加到序列中，并重复。每一个生产级 LLM 推理堆栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都实现了相同的循环，并进行了大量优化 —— 批量预填充、连续批处理、KV 缓存分页、贪婪解码。

**GPT 与 BERT，一句话概括：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。损失函数决定了模型是否能够生成文本。

## 部署它

参见 `outputs/skill-sampling-tuner.md`。该技能为新的生成任务选择采样参数，并在需要确定性解码时进行提示。

## 练习

1.  **简单。** 运行 `code/main.py` 并验证因果注意力矩阵在 softmax 之后是下三角矩阵。抽查：第 3 行应该仅在列 0–3 有权重。
2.  **中等。** 实现宽度为 4 的束搜索。在 10 个短提示上比较束搜索-4 和贪心解码的困惑度。束搜索总是更好吗？（提示：通常对翻译任务更好，对开放式聊天则不然。）
3.  **困难。** 实现贪婪解码：使用一个小型两层模型作为草稿模型，一个六层模型作为验证器。在 100 个长度为 64 的补全任务上测量实际加速比。确认输出与验证器的贪心解码输出匹配。

## 关键术语

| 术语             | 人们怎么说                   | 它的实际含义                                                                 |
|------------------|-----------------------------|-----------------------------------------------------------------------------|
| 因果掩码         | "那个三角形"                 | 加到注意力分数上的上三角 `-inf` 矩阵，使得位置 `i` 只能看到位置 `≤ i`。 |
| 下一个 token 预测 | "那个损失函数"               | 模型分布与每个位置真实下一个 token 的交叉熵。                               |
| 自回归           | "一个一个地生成"             | 将输出反馈作为输入；仅在训练期间并行，生成期间则串行。                     |
| Logits           | "softmax 前的分数"           | LM 头在 softmax 之前的原始输出；基于此进行采样。                            |
| 温度             | "创意旋钮"                   | 将 logits 除以 T；T→0 = 贪心，T→∞ = 均匀分布。                             |
| Top-p            | "核采样"                     | 将分布截断到和为 ≥ p 的最小集合；从剩余部分中采样。                         |
| Min-p            | "比 top-p 更好"              | 保留 `p ≥ min_p × max_p` 的 token；截止阈值适应分布的尖锐程度。                     |
| 贪婪解码         | "草稿 + 验证"                | 廉价模型提出 N 个 token；大模型并行验证。                                   |
| 教师强制         | "训练技巧"                   | 训练期间，喂入真实的前一个 token，而不是模型的预测。每个序列到序列语言模型的标准做法。 |

## 延伸阅读

- [Radford 等人 (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford 等人 (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown 等人 (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 和上下文学习。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 贪婪解码论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 标准的因果语言模型参考代码。