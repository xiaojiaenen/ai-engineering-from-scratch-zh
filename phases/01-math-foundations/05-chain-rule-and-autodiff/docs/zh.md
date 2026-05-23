# 链式法则与自动微分

> 链式法则驱动着每个学习中的神经网络。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段1，课程04（导数与梯度）
**时间：** ~90分钟

## 学习目标

- 构建一个最小的自动微分引擎（Value类），该引擎记录操作并通过反向模式自动微分计算梯度
- 使用拓扑排序实现计算图的前向传播和反向传播
- 仅使用从头构建的自动微分引擎，在异或问题上构建并训练一个MLP
- 使用梯度检验（对比数值有限差分）验证自动微分的正确性

## 问题所在

你可以计算简单函数的导数。但神经网络不是一个简单的函数。它是数百个函数的组合：矩阵乘法、添加偏置、应用激活函数、再次矩阵乘法、softmax、交叉熵损失。输出是一个函数的函数的函数。

要训练网络，你需要损失相对于每个权重的梯度。对于数百万参数，手动计算是不可能的。数值计算（有限差分）太慢。

链式法则提供了数学基础。自动微分提供了算法。它们结合在一起，让你可以在与单次前向传播成比例的时间内，精确计算任意函数组合的梯度。

这就是PyTorch、TensorFlow和JAX的工作原理。你将从头构建一个微型版本。

## 概念

### 链式法则

如果 `y = f(g(x))`，则 `y` 对 `x` 的导数为：

```
dy/dx = dy/dg * dg/dx = f'(g(x)) * g'(x)
```

沿着链相乘导数。每个环节贡献其局部导数。

示例：`y = sin(x^2)`

```
g(x) = x^2       g'(x) = 2x
f(g) = sin(g)     f'(g) = cos(g)

dy/dx = cos(x^2) * 2x
```

对于更深层次的组合，链会延伸：

```
y = f(g(h(x)))

dy/dx = f'(g(h(x))) * g'(h(x)) * h'(x)
```

神经网络中的每一层都是这个链中的一个环节。

### 计算图

计算图使链式法则可视化。每个操作成为一个节点。数据在图中向前流动。梯度向后流动。

**前向传播（计算值）：**

```mermaid
graph TD
    x1["x1 = 2"] --> mul["* (multiply)"]
    x2["x2 = 3"] --> mul
    mul -->|"a = 6"| add["+ (add)"]
    b["b = 1"] --> add
    add -->|"c = 7"| relu["relu"]
    relu -->|"y = 7"| y["output y"]
```

**反向传播（计算梯度）：**

```mermaid
graph TD
    dy["dy/dy = 1"] -->|"relu'(c)=1 since c>0"| dc["dy/dc = 1"]
    dc -->|"dc/da = 1"| da["dy/da = 1"]
    dc -->|"dc/db = 1"| db["dy/db = 1"]
    da -->|"da/dx1 = x2 = 3"| dx1["dy/dx1 = 3"]
    da -->|"da/dx2 = x1 = 2"| dx2["dy/dx2 = 2"]
```

反向传播在每个节点应用链式法则，将梯度从输出传播到输入。

### 前向模式 vs 反向模式

有两种通过图应用链式法则的方式。

**前向模式**从输入开始，向前推送导数。它计算 `dx/dx = 1` 并通过每个操作传播。当输入少、输出多时适用。

```
Forward mode: seed dx/dx = 1, propagate forward

  x = 2       (dx/dx = 1)
  a = x^2     (da/dx = 2x = 4)
  y = sin(a)  (dy/dx = cos(a) * da/dx = cos(4) * 4 = -2.615)
```

**反向模式**从输出开始，向后拉取梯度。它计算 `dy/dy = 1` 并逆向通过每个操作传播。当输入多、输出少时适用。

```
Reverse mode: seed dy/dy = 1, propagate backward

  y = sin(a)  (dy/dy = 1)
  a = x^2     (dy/da = cos(a) = cos(4) = -0.654)
  x = 2       (dy/dx = dy/da * da/dx = -0.654 * 4 = -2.615)
```

神经网络有数百万个输入（权重）和一个输出（损失）。反向模式在一次反向传播中计算所有梯度。这就是反向传播使用反向模式的原因。

| 模式 | 种子 | 方向 | 最佳适用场景 |
|------|------|------|-------------|
| 前向 | `dx_i/dx_i = 1` | 从输入到输出 | 输入少，输出多 |
| 反向 | `dy/dy = 1` | 从输出到输入 | 输入多，输出少（神经网络） |

### 前向模式的对偶数

前向模式可以用对偶数优雅地实现。一个对偶数形式为 `a + b*epsilon`，其中 `epsilon^2 = 0`。

```
Dual number: (value, derivative)

(2, 1) means: value is 2, derivative w.r.t. x is 1

Arithmetic rules:
  (a, a') + (b, b') = (a+b, a'+b')
  (a, a') * (b, b') = (a*b, a'*b + a*b')
  sin(a, a')         = (sin(a), cos(a)*a')
```

为输入变量赋予导数种子1。导数会自动通过每个操作传播。

### 构建自动微分引擎

一个自动微分引擎需要三样东西：

1.  **值包装。** 将每个数字包装在一个存储其值和梯度的对象中。
2.  **图记录。** 每个操作记录其输入和局部梯度函数。
3.  **反向传播。** 对图进行拓扑排序，然后反向遍历，在每个节点应用链式法则。

这正是PyTorch的 `autograd` 所做的事情。`torch.Tensor` 类包装值，在 `requires_grad=True` 时记录操作，当你调用 `.backward()` 时计算梯度。

### PyTorch自动微分的内部工作原理

当你编写PyTorch代码时：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # 7.0 = 2*x + 3 = 2*2 + 3
```

PyTorch内部：

1.  为 `x` 创建一个 `Tensor` 节点，使用 `requires_grad=True`
2.  每个操作 (`**`, `*`, `+`) 创建一个新节点并记录反向函数
3.  `y.backward()` 通过记录的图触发反向模式自动微分
4.  每个节点的 `grad_fn` 计算局部梯度并将其传递给父节点
5.  梯度通过加法（而非替换）累积在 `.grad` 属性中

图是动态的（运行时定义）。每次前向传播都会构建一个新图。这就是为什么PyTorch支持模型内部的控制流（if/else，循环）。

## 构建它

### 步骤1：Value 类

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

每个 `Value` 存储其数值数据、其梯度（初始为零）、一个反向函数，以及指向产生它的子节点的指针。

### 步骤2：带梯度追踪的算术运算

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
```

每个操作创建一个闭包，该闭包知道如何计算局部梯度并乘以梯度 (`out.grad`)。`+=` 处理一个值在多个操作中使用的情况。

### 步骤3：反向传播

```python
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

拓扑排序确保每个节点的梯度在传播到其子节点之前被完全计算。种子梯度是1.0 (dy/dy = 1)。

### 步骤4：更多操作以实现完整引擎

基本的Value类处理加法、乘法和relu。一个真正的自动微分引擎需要更多。以下是构建神经网络所需的操作：

```python
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
```

**为什么每个操作都重要：**

| 操作 | 反向规则 | 使用场景 |
|------|---------|---------|
| `__sub__` | 复用加法 + 取负 | 损失计算 (pred - target) |
| `__pow__` | n * x^(n-1) | 多项式激活函数，MSE (error^2) |
| `__truediv__` | 复用乘法 + 幂次方(-1) | 归一化，学习率缩放 |
| `exp` | exp(x) * 梯度 | Softmax，对数似然 |
| `log` | (1/x) * 梯度 | 交叉熵损失，对数概率 |
| `tanh` | (1 - tanh^2) * 梯度 | 经典激活函数 |

巧妙之处在于：`__sub__` 和 `__truediv__` 是根据现有操作定义的。它们自动获得正确的梯度，因为链式法则通过底层的加法/乘法/幂次方操作组合起来。

### 步骤5：从头构建一个迷你MLP

有了完整的Value类，你就可以构建一个神经网络了。没有PyTorch。没有NumPy。只有Value和链式法则。

```python
import random

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

一个 `Neuron` 计算 `tanh(w1*x1 + w2*x2 + ... + b)`。一个 `Layer` 是一个神经元列表。一个 `MLP` 堆叠层。每个权重都是一个 `Value`，因此调用 `loss.backward()` 会将梯度传播到每个参数。

**在异或问题上训练：**

```python
random.seed(42)
model = MLP([2, 4, 1])  # 2 inputs, 4 hidden neurons, 1 output

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR pattern (using -1/1 for tanh)

for step in range(100):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 20 == 0:
        print(f"step {step:3d}  loss = {loss.data:.4f}")

print("\nPredictions after training:")
for x, y in zip(xs, ys):
    print(f"  input={x}  target={y:2d}  pred={model(x).data:6.3f}")
```

这就是micrograd。一个使用自动微分的纯Python完整神经网络训练循环。每个商业深度学习框架都在大规模地做同样的事情。

### 步骤6：梯度检验

如何知道你的自动微分是否正确？与数值导数进行比较。这就是梯度检验。

```python
def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff
```

在一个复杂表达式上测试它：

```python
def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_check(expr, 0.5)
print(f"Autodiff:  {ad:.8f}")
print(f"Numerical: {num:.8f}")
print(f"Difference: {diff:.2e}")
# Difference should be < 1e-5
```

在实现新操作时，梯度检验至关重要。如果你的反向传播有错误，数值检验可以捕获它。每个严肃的深度学习实现在开发过程中都会运行梯度检验。

**何时使用梯度检验：**

| 情况 | 进行梯度检验？ |
|------|---------------|
| 为你的自动微分添加新操作 | 是，总是 |
| 调试不收敛的训练循环 | 是，首先检查梯度 |
| 生产环境训练 | 否，太慢（每个参数2倍前向传播） |
| 自动微分代码的单元测试 | 是，自动化它 |

### 步骤7：与手动计算验证

```python
x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2          # a = 6.0
b = a + Value(1.0)    # b = 7.0
y = b.relu()          # y = 7.0

y.backward()

print(f"y = {y.data}")          # 7.0
print(f"dy/dx1 = {x1.grad}")   # 3.0 (= x2)
print(f"dy/dx2 = {x2.grad}")   # 2.0 (= x1)
```

手动检查：`y = relu(x1*x2 + 1)`。由于 `x1*x2 + 1 = 7 > 0`，relu 是恒等映射。
`dy/dx1 = x2 = 3`。 `dy/dx2 = x1 = 2`。引擎匹配。

## 使用它

### 与PyTorch验证

```python
import torch

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()

print(f"PyTorch dy/dx1 = {x1.grad.item()}")  # 3.0
print(f"PyTorch dy/dx2 = {x2.grad.item()}")  # 2.0
```

梯度相同。你的引擎计算出与PyTorch相同的结果，因为数学是相同的：通过链式法则的反向模式自动微分。

### 一个更复杂的表达式

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = (a * b + c).relu()  # relu(2*(-3) + 10) = relu(4) = 4

f.backward()
print(f"df/da = {a.grad}")  # -3.0 (= b)
print(f"df/db = {b.grad}")  #  2.0 (= a)
print(f"df/dc = {c.grad}")  #  1.0
```

## 交付成果

本课程产出：
- `outputs/skill-autodiff.md` -- 一项构建和调试自动微分系统的技能
- `code/autodiff.py` -- 一个你可以扩展的最小自动微分引擎

这里构建的Value类是阶段3中神经网络训练循环的基础。

## 练习

1.  给Value类添加 `__pow__`，以便你可以计算 `x ** n`。验证在 `x=2` 处 `d/dx(x^3)` 等于 `12.0`。
2.  添加 `tanh` 作为激活函数。验证 `tanh'(0) = 1` 且 `tanh'(2) = 0.0707` (近似)。
3.  为一个神经元构建计算图：`y = relu(w1*x1 + w2*x2 + b)`。计算所有五个梯度并与PyTorch验证。
4.  使用对偶数实现前向模式自动微分。创建一个 `Dual` 类，并验证它给出的导数与你的反向模式引擎相同。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 链式法则 | “乘以导数” | 组合函数的导数等于每个函数局部导数的乘积，在正确点处计算 |
| 计算图 | “网络图” | 一个有向无环图，节点是操作，边承载值（前向）或梯度（反向） |
| 前向模式 | “向前推送导数” | 从输入到输出传播导数的自动微分。每个输入变量一次传播。 |
| 反向模式 | “反向传播” | 从输出到输入传播梯度的自动微分。每个输出变量一次传播。 |
| 自动微分 | “自动计算梯度” | 一个记录对值的操作、构建图并通过链式法则计算精确梯度的系统 |
| 对偶数 | “值加导数” | 形式为 a + b*epsilon (epsilon^2 = 0) 的数，通过算术运算携带导数信息 |
| 拓扑排序 | “依赖顺序” | 图节点排序，使每个节点在其所有依赖之后出现。正确梯度传播所必需。 |
| 梯度累积 | “相加，不替换” | 当一个值输入到多个操作时，其梯度是所有传入梯度贡献的总和 |
| 动态图 | “运行时定义” | 在每次前向传播时重新构建的计算图，允许模型内部有Python控制流（PyTorch风格） |
| 梯度检验 | “数值验证” | 将自动微分梯度与数值有限差分梯度进行比较以验证正确性。调试所必需。 |
| MLP | “多层感知机” | 具有一个或多个隐藏神经元层的神经网络。每个神经元计算加权和加上偏置，然后应用激活函数。 |
| 神经元 | “加权和 + 激活” | 基本单元：输出 = 激活(w1*x1 + w2*x2 + ... + b)。权重和偏置是可学习的参数。 |

## 延伸阅读

- [3Blue1Brown：反向传播微积分](https://www.youtube.com/watch?v=tIeHLnjs5U8) -- 神经网络中链式法则的可视化解释
- [PyTorch自动微分机制](https://pytorch.org/docs/stable/notes/autograd.html) -- 真实系统的工作原理
- [Baydin等人，机器学习中的自动微分：综述](https://arxiv.org/abs/1502.05767) -- 全面的参考文献