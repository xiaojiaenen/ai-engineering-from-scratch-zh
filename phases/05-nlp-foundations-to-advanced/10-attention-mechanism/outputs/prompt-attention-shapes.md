---
name: attention-shapes
description: Debug shape bugs in attention implementations.
description-zh: # Debugging Shape Bugs in Attention Implementations

## Common Shape Issues

### 1. Missing/Extra Dimensions in Reshaping

```python
# WRONG: Forgetting to split heads
# q, k, v shape: (batch, seq_len, d_model)
attn = torch.matmul(q, k.transpose(-2, -1))  # WRONG if d_model != d_k

# CORRECT: Split into heads first
# (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
q = q.view(batch, seq_len, n_heads, d_k).transpose(1, 2)
k = k.view(batch, seq_len, n_heads, d_k).transpose(1, 2)
v = v.view(batch, seq_len, n_heads, d_v).transpose(1, 2)
```

### 2. Transpose Axes Mismatch

```python
# WRONG: Transposing wrong dimensions
# After view+transpose: (batch,
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from `(d_s, d_h, d_attn, T_enc, T_dec, batch_size)`.
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1)` is close to 1.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). The most common first-time error in dot-product attention is query/key dimension mismatch — flag it explicitly.
