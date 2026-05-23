# 从零开始实现自注意力机制

> 注意力机制就像一个查找表：每个词都会问“谁对我重要？” ——并学习这个答案。

**类型：** 构建
**语言：** Python
**先修知识：** 阶段3（深度学习核心），阶段5第10课（序列到序列模型）
**时间：** 约90分钟

## 学习目标

- 仅使用NumPy从零实现带缩放点积的自注意力机制，包括查询/键/值投影与softmax加权求和
- 构建一个包含多头拆分、并行注意力计算与结果拼接的多头注意力层
- 追踪注意力矩阵如何捕捉token关系，并解释为何通过√(d_k)缩放能防止softmax饱和
- 应用因果掩码将双向注意力转换为自回归（解码器风格）注意力

## 问题背景

循环神经网络逐token处理序列。当你处理到第50个token时，第1个token的信息已经过50次压缩步骤。长程依赖被挤压到固定大小的隐藏状态中——这是LSTM门控机制也无法完全解决的瓶颈。

2014年Bahdanau注意力论文提出了解决方案：让解码器回溯每个编码器位置，并判断哪些对当前步骤重要。但这仍依附于循环神经网络。2017年《Attention Is All You Need》论文提出了更尖锐的问题：如果注意力机制是*唯一*机制会怎样？没有循环。没有卷积。只有注意力。

自注意力让序列中的每个位置在单次并行步骤中都能关注其他所有位置。这正是Transformer快速、可扩展且占据主导地位的原因。

## 核心概念

### 数据库查找类比

可以将注意力机制看作一次软性数据库查找：

```
Traditional database:
  Query: "capital of France"  -->  exact match  -->  "Paris"

Attention:
  Query: "capital of France"  -->  similarity to ALL keys  -->  weighted blend of ALL values
```

每个token会生成三个向量：
- **查询（Q）**：“我在寻找什么？”
- **键（K）**：“我包含什么信息？”
- **值（V）**：“如果被选中，我提供什么信息？”

查询与所有键的点积产生注意力分数。高分意味着“这个键与我的查询匹配”。这些分数对值进行加权。输出是值的加权总和。

### Q、K、V 的计算

每个token嵌入会通过三个可学习的权重矩阵进行投影：

```
Input embeddings (sequence of n tokens, each d-dimensional):

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

Three weight matrices:

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

Projections:

  Q = X @ Wq    shape: (n, dk)      each token's query
  K = X @ Wk    shape: (n, dk)      each token's key
  V = X @ Wv    shape: (n, dv)      each token's value
```

直观表示，对于单个token：

```
             Wq
  x_i ------[*]------> q_i    "What am I looking for?"
       |
       |     Wk
       +----[*]------> k_i    "What do I contain?"
       |
       |     Wv
       +----[*]------> v_i    "What do I offer?"
```

### 注意力矩阵

一旦获得所有token的Q、K、V，注意力分数将构成一个矩阵：

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- how much q1 attends to each key
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

Each row: one token's attention over the entire sequence
```

### 为何需要缩放？

点积的值随维度d_k增长。若d_k=64，点积可能达到数十，导致softmax进入梯度消失的区域。解决方法：除以√(d_k)。

```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

这使数值保持在softmax能产生有效梯度的范围内。

### Softmax 将分数转化为权重

Softmax将原始分数转换为每行的概率分布：

```
Raw scores for q1:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
Attention weights:   [0.52, 0.09, 0.07, 0.14, 0.08]   (sums to ~1.0)
```

现在每个token都有一组权重，表示它应对其他每个token给予多少关注。

### 值的加权求和

每个token的最终输出是所有值向量的加权求和：

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

For token 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### 完整流程

```
                    +-------+
  X (input)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (input)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (input)  ----->|  @ Wv  |-----> V ---------->| weighted |----> output
                    +-------+          ^          |   sum    |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

一行公式表达：

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## 动手实现

### 步骤1：从零实现Softmax

Softmax将原始logits转换为概率。减去最大值以保证数值稳定性。

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### 步骤2：带缩放的点积注意力

核心函数。接收Q、K、V矩阵，返回注意力输出及权重矩阵。

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### 步骤3：带可学习投影的自注意力类

一个完整的自注意力模块，包含使用类Xavier缩放初始化的Wq、Wk、Wv权重矩阵。

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### 步骤4：在句子上运行

为句子创建虚拟嵌入并观察注意力权重。

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### 步骤5：用ASCII热力图可视化注意力

将注意力权重映射到字符以实现快速可视化。

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## 实际应用

PyTorch的`nn.MultiheadAttention`实现了我们构建的功能，并增加了多头拆分与输出投影：

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

关键区别：多头注意力并行运行多个注意力函数，每个函数使用独立的Q、K、V投影（维度为d_k = d_model / n_heads），然后拼接结果。这让模型能同时关注不同类型的关系。

## 成果产出

本课将产出：
- `outputs/prompt-attention-explainer.md` - 一个通过数据库查找类比解释注意力机制的提示词

## 练习

1. 修改`scaled_dot_product_attention`以接收可选的掩码矩阵，在softmax前将特定位置设为负无穷（这就是因果/解码器掩码的工作原理）
2. 从零实现多头注意力：将Q、K、V拆分为`n_heads`个块，分别对每个块运行注意力，拼接结果，并通过最终权重矩阵Wo进行投影
3. 取两个长度相同的句子，通过同一个SelfAttention实例，比较它们的注意力模式。什么改变了？什么保持不变？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 查询（Q） | “问题向量” | 输入的可学习投影，表示该token正在寻找什么信息 |
| 键（K） | “标签向量” | 可学习投影，表示该token包含什么信息，与查询进行匹配 |
| 值（V） | “内容向量” | 携带实际信息的可学习投影，根据注意力分数进行聚合 |
| 带缩放的点积注意力 | “注意力公式” | softmax(QK^T / √(d_k)) @ V —— 缩放可防止高维下的softmax饱和 |
| 自注意力 | “token查看自身与其他” | Q、K、V都来自同一序列的注意力机制，使每个位置都能关注其他所有位置 |
| 注意力权重 | “聚焦程度” | 通过缩放点积的softmax产生的位置概率分布 |
| 多头注意力 | “并行注意力” | 使用不同投影运行多个注意力函数，然后拼接结果以获得更丰富的表示 |

## 延伸阅读

- [《Attention Is All You Need》(Vaswani等, 2017)](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - 最佳全架构可视化解析
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) - 逐行PyTorch实现与注释