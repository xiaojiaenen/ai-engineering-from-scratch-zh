# 数值稳定性

> 浮点数是一种有漏洞的抽象。它会在训练过程中咬你一口，而你事先毫无察觉。

**类型：** 实践
**语言：** Python
**先决条件：** 第一阶段，第01-04课
**时间：** 约120分钟

## 学习目标

- 使用最大值减法技巧实现数值稳定的softmax和log-sum-exp
- 识别浮点数计算中的上溢、下溢和灾难性抵消
- 使用中心有限差分法验证解析梯度与数值梯度的一致性
- 解释为什么bfloat16在训练中优于float16，以及损失缩放如何防止梯度下溢

## 问题所在

你的模型训练了三个小时，然后损失变成了NaN。你加了一条打印语句。在第9000步，logits还是正常的。到第9001步，它们变成了 `inf`。到第9002步，每个梯度都是 `nan`，训练彻底死了。

或者：你的模型训练完成了，但准确率比论文声称的低2%。你检查了所有东西。架构匹配。超参数匹配。数据匹配。问题出在论文用了float32，而你用了float16却没有正确的缩放。32位的累积舍入误差悄无声息地吞噬了你的准确率。

或者：你从头实现了交叉熵损失。在较小的logits上它能正常工作。当logits超过100时，它返回 `inf`。softmax上溢了，因为 `exp(100)` 超出了float32能表示的范围。每个ML框架都用两行代码的技巧处理这个问题。你当时不知道有这个技巧。

数值稳定性不是理论问题。它是训练成功与悄然失败之间的区别。你将来要调试的每一个严肃的ML bug，最终都归结为浮点数问题。

## 核心概念

### IEEE 754：计算机如何存储实数

计算机遵循IEEE 754标准，将实数存储为浮点值。一个浮点数由三部分组成：符号位、指数和尾数（有效数字）。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度（多少位有效数字）。指数决定范围（数字能有多大或多小）。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32给你大约7位十进制精度。这意味着它能区分1.0000001和1.0000002，但不能区分1.00000001和1.00000002。超过7位后，一切都是舍入噪声。

float16给你大约3位精度。它能表示的最大数字是65,504。这对于ML来说小得令人不安，因为logits、梯度和激活值经常超过这个数。

bfloat16是谷歌对float16范围问题的回答。它拥有与float32相同的8位指数（相同的范围，高达3.4e38），但只有7位尾数（精度低于float16）。对于训练神经网络，范围比精度更重要，所以bfloat16通常更优。

### 为什么 0.1 + 0.2 != 0.3

数字0.1无法在二进制浮点数中精确表示。在二进制下，它是一个循环小数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32将其截断为23位尾数。存储的值大约是0.100000001490116。类似地，0.2存储为大约0.200000002980232。它们的和是0.300000004470348，而不是0.3。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这在ML中很重要，因为：
1. 像 `if loss < threshold` 这样的损失比较可能给出错误答案
2. 累积许多小的值（经过数千步的梯度更新）会偏离真实总和
3. 如果你用 `==` 比较浮点数，校验和和可重复性测试会失败

解决方案：永远不要用 `==` 比较浮点数。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消

当你减去两个几乎相等的浮点数时，有效数字相互抵消，剩下的只是舍入噪声被提升为主要数字。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

这是一次减法造成的19%的相对误差。在ML中，这发生在你：
- 计算具有大均值的数据的方差时：`E[x^2] - E[x]^2` 当E[x]很大时
- 减去几乎相等的对数概率时
- 用过小的epsilon计算有限差分梯度时

解决方案：重新排列公式以避免减去大的、几乎相等的数字。对于方差，使用Welford算法或先对数据进行中心化。对于对数概率，全程在对数空间操作。

### 上溢与下溢

当结果太大无法表示时发生上溢。当结果太小（比最小的可表示正数更接近零）时发生下溢。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

在ML中，`exp()` 函数是上溢的主要来源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 函数则走向另一个方向：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在ML中，`exp()` 出现在softmax、sigmoid和概率计算中。`log()` 出现在交叉熵、对数似然和KL散度中。组合 `log(exp(x))` 没有正确的技巧就是一个雷区。

### Log-Sum-Exp技巧

直接计算 `log(sum(exp(x_i)))` 在数值上是危险的。如果任何一个 `x_i` 很大，`exp(x_i)` 就会上溢。如果所有 `x_i` 都是非常负的数，每个 `exp(x_i)` 都会下溢为零，而 `log(0)` 就是 `-inf`。

技巧：在取指数之前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么这样做有效：减去 `max(x)` 后，最大的指数是 `exp(0) = 1`。不可能发生上溢。求和中至少有一项是1，所以总和至少是1，且 `log(1) = 0`。不可能下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

令 `c = max(x)` 则上溢被消除。

这个技巧在ML中无处不在：
- Softmax归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么Softmax需要最大值减法技巧

Softmax将logits转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没有这个技巧，logits为 [100, 101, 102] 会导致上溢：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

使用该技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率是相同的。计算是安全的。这不是优化。这是正确性的要求。

### NaN和Inf：检测与预防

`nan` (Not a Number) 和 `inf` (infinity) 会在计算中像病毒一样传播。梯度更新中出现一个 `nan` 会使权重变成 `nan`，这又会使后续每个输出变成 `nan`。训练在一步之内就死了。

`inf` 出现的情况：
- 一个大正数的 `exp()`
- 除以零：`1.0 / 0.0`
- 累加中的 `float32` 上溢

`nan` 出现的情况：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 负数的 `sqrt()`
- 负数的 `log()`
- 任何涉及已有 `nan` 的算术运算

检测：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

预防策略：
1. 将输入钳位到 `exp()`：`exp(clamp(x, -80, 80))`
2. 分母加上epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内部加上epsilon：`log(x + 1e-8)`
4. 使用稳定的实现（log-sum-exp, stable softmax）
5. 梯度裁剪以防止权重爆炸
6. 在调试期间，每次前向传播后检查 `nan`/`inf`

### 数值梯度检查

解析梯度（来自反向传播）可能有bug。数值梯度检查通过有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是O(h^2)精度，比只有O(h)精度的前向差分 `(f(x+h) - f(x)) / h` 好得多。

选择h：太大会导致近似错误。太小则灾难性抵消会破坏结果。`h = 1e-5` 到 `1e-7` 是典型范围。

检查：计算解析梯度和数值梯度之间的相对误差。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，可能正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

在实现新的层或损失函数时，务必检查梯度。PyTorch为此提供了 `torch.autograd.gradcheck()`。

### 混合精度训练

现代GPU拥有专门的硬件（张量核心），其float16矩阵乘法速度比float32快2-8倍。混合精度训练利用了这一点：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯float16训练的问题：梯度通常非常小（1e-8或更小）。Float16会将低于约6e-8的任何数下溢为零。你的模型停止学习，因为所有的梯度更新都是零。

解决方案是损失缩放：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

动态损失缩放会自动调整缩放因子。从一个较大的值（65536）开始。如果梯度上溢为 `inf`，则将其减半。如果N步没有发生上溢，则将其加倍。

### bfloat16 vs float16：为什么bfloat16在训练中胜出

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16有更高的精度（10位尾数 vs 7位），但范围有限（最大约65,504）。bfloat16精度较低，但拥有与float32相同的范围（最大约3.4e38）。

对于训练神经网络：
- 在训练尖峰期间，激活值和logits经常超过65,504。float16会溢出；bfloat16能处理。
- float16需要损失缩放，但bfloat16通常不需要，因为其范围覆盖了梯度幅度谱。
- bfloat16是float32的简单截断：丢弃尾数的低16位。转换是微不足道的，且在指数上是无损的。

float16更适用于推理，因为其值有界且精度更重要。bfloat16更适用于训练，因为范围更重要。这就是为什么TPU和现代NVIDIA GPU（A100, H100）原生支持bfloat16。

### 梯度裁剪

梯度爆炸发生在梯度通过许多层呈指数增长时（常见于RNN、深度网络和Transformer）。一个大的梯度就能在一步内破坏所有权重。

两种裁剪类型：

**按值裁剪：** 独立地钳制每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单但可能改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留梯度的方向。这就是 `torch.nn.utils.clip_grad_norm_()` 所做的。这是标准选择。

典型值：Transformer用 `max_norm=1.0`，强化学习用 `max_norm=0.5`，更简单的网络用 `max_norm=5.0`。

梯度裁剪不是hack。它是一种安全机制。没有它，一个离群批次就可能产生一个足以毁掉数周训练的梯度。

### 归一化层作为数值稳定器

批归一化、层归一化和RMS归一化通常被呈现为帮助训练收敛的正则化器。它们也是数值稳定器。

没有归一化时，激活值可能在层间呈指数增长或衰减：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

归一化在每一层重新中心化并缩放激活值：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常为1e-5）防止当所有激活值相同时除以零。学习到的参数 `gamma` 和 `beta` 让网络可以恢复它需要的任何尺度。

这使整个网络中的数值保持在数值安全的范围内，既防止了前向传播中的上溢，也防止了反向传播中的梯度爆炸。

### 常见的ML数值Bug

**Bug：几个epoch后损失变成NaN。**
原因：logits增长过大，softmax上溢。或者学习率过高导致权重发散。
解决方案：使用稳定的softmax（最大值减法），降低学习率，添加梯度裁剪。

**Bug：损失卡在 log(num_classes)。**
原因：模型输出接近均匀概率。通常意味着梯度消失或模型根本没有学习。
解决方案：检查数据标签是否正确，验证损失函数，检查是否有死ReLU。

**Bug：验证准确率比预期低1-3%。**
原因：混合精度没有正确的损失缩放。梯度下溢悄悄地将小的更新归零。
解决方案：启用动态损失缩放，或切换到bfloat16。

**Bug：某些层的梯度范数为0.0。**
原因：死ReLU神经元（所有输入为负），或float16下溢。
解决方案：使用LeakyReLU或GELU，使用梯度缩放，检查权重初始化。

**Bug：模型在一个GPU上工作，但在另一个GPU上给出不同的结果。**
原因：非确定性的浮点累加顺序。GPU并行归约在不同的硬件上以不同的顺序求和，而浮点加法不满足结合律。
解决方案：接受微小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度损失。

**Bug：`exp()` 在损失计算中返回 `inf`。**
原因：原始logits未经最大值减法技巧就传递给了 `exp()`。
解决方案：使用 `torch.nn.functional.log_softmax()`，它内部实现了log-sum-exp。

**Bug：从float32切换到float16后训练发散。**
原因：float16无法表示低于6e-8的梯度幅度或高于65,504的激活值。
解决方案：使用带损失缩放的混合精度（AMP），或改用bfloat16。

## 动手实践

### 步骤1：演示浮点数精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤2：实现朴素vs稳定softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 步骤3：实现稳定的log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 步骤4：实现稳定的交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤5：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 应用场景

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现及所有边界情况演示，请参见 `code/numerical.py`。

## 交付成果

本课程产出：
- 包含稳定softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟的 `code/numerical.py`
- 用于诊断训练中NaN/Inf和数值问题的 `outputs/prompt-numerical-debugger.md`

这些稳定的实现将在第三阶段构建训练循环和第四阶段实现注意力机制时再次出现。

## 练习

1. **灾难性抵消。** 使用朴素公式 `E[x^2] - E[x]^2` 在float32下计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后使用Welford在线算法计算。将误差与真实方差（0.6667）进行比较。

2. **精度探索。** 在Python中找到最小的正float32值 `x`，使得 `1.0 + x == 1.0`。这就是机器epsilon。验证它与 `numpy.finfo(numpy.float32).eps` 匹配。

3. **Log-sum-exp边界情况。** 用以下情况测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其他，(c) 所有值非常负（-1000）。验证它在朴素版本失败的地方给出正确结果。

4. **神经网络层的梯度检查。** 实现一个单线性层 `y = Wx + b` 及其解析反向传播。使用 `numerical_gradient` 验证3x2权重矩阵的正确性。

5. **损失缩放实验。** 模拟使用float16训练：在 [1e-9, 1e-3] 范围内创建随机梯度，转换为float16，测量有多少比例变为零。然后应用损失缩放（乘以1024），转换为float16，再缩放回来，再次测量零比例。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|------------|--------------|
| IEEE 754 | “浮点标准” | 定义二进制浮点格式、舍入规则和特殊值（inf, nan）的国际标准。每个现代CPU和GPU都实现它。 |
| 机器epsilon | “精度极限” | 使得 1.0 + e != 1.0 的最小值e（在给定的浮点格式下）。对于float32，约为1.19e-7。 |
| 灾难性抵消 | “减法造成的精度损失” | 当减去几乎相等的浮点数时，有效数字抵消，舍入噪声主导了结果。 |
| 上溢 | “数字太大” | 结果超过最大可表示值并变成inf。exp(89) 会使float32上溢。 |
| 下溢 | “数字太小” | 结果比最小的可表示正数更接近零，并变成0.0。exp(-104) 会使float32下溢。 |
| Log-sum-exp技巧 | “先减去最大值” | 通过提取 exp(max(x)) 来计算 log(sum(exp(x))) 以防止上溢和下溢。用于softmax、交叉熵和对数概率计算。 |
| 稳定softmax | “不会爆炸的softmax” | 在取指数前减去最大logits。数值上结果相同，但不会上溢。 |
| 梯度检查 | “验证你的反向传播” | 比较反向传播得到的解析梯度与有限差分得到的数值梯度，以发现实现错误。 |
| 混合精度 | “float16前向，float32向后” | 对速度关键的操作使用较低精度的浮点数，对数值敏感的操作使用较高精度的浮点数。通常加速2-3倍。 |
| 损失缩放 | “防止梯度下溢” | 在反向传播前将损失乘以一个大常数，使梯度保持在float16的可表示范围内，然后在权重更新前除以相同的常数。 |
| bfloat16 | “脑浮点数” | 谷歌的16位格式，拥有8位指数（与float32范围相同）和7位尾数（精度低于float16）。更适合训练。 |
| 梯度裁剪 | “限制梯度范数” | 缩放梯度向量使其范数不超过阈值。防止梯度爆炸破坏权重。 |
| NaN | “非数字” | 来自未定义运算（0/0, inf-inf, sqrt(-1)）的特殊浮点值。会在所有后续算术运算中传播。 |
| Inf | “无穷大” | 来自上溢或除以零的特殊浮点值。可以组合产生NaN（inf - inf, inf * 0）。 |
| 数值梯度 | “暴力求导” | 通过计算f(x+h)和f(x-h)并除以2h来近似导数。慢但可靠，用于验证。 |

## 扩展阅读

- [每位计算机科学家都应了解的浮点算术 (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考，详尽但完整
- [混合精度训练 (Micikevicius等人, 2018)](https://arxiv.org/abs/1710.03740) -- NVIDIA提出的用于float16训练的损失缩放论文
- [AMP：自动混合精度 (PyTorch文档)](https://pytorch.org/docs/stable/amp.html) -- PyTorch混合精度实践指南
- [bfloat16格式 (Google Cloud TPU文档)](https://cloud.google.com/tpu/docs/bfloat16) -- 谷歌为TPU选择此格式的原因
- [Kahan求和法 (维基百科)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 减少浮点数求和舍入误差的算法