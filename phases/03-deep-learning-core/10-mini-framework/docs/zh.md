# 构建你自己的迷你框架

> 你已经构建了神经元、层、网络、反向传播、激活函数、损失函数、优化器、正则化、初始化和学习率调度。所有这些都是独立的部件。现在将它们连接成一个框架。不是 PyTorch。不是 TensorFlow。是你自己的。

**类型：** 构建
**语言：** Python
**前提：** 阶段03全部课程（第01-09课）
**时间：** 约120分钟

## 学习目标

- 构建一个完整的深度学习框架（约500行代码），包含 Module、Linear、ReLU、Sigmoid、Dropout、BatchNorm、Sequential、损失函数、优化器和 DataLoader
- 解释 Module 抽象（forward、backward、parameters）以及为什么需要训练/评估模式切换
- 将所有组件连接到一个可工作的训练循环中，训练一个4层网络进行圆形分类
- 将你框架中的每个组件映射到 PyTorch 的等价物（nn.Module、nn.Sequential、optim.Adam、DataLoader）

## 问题

你有十节课的构建块分散在不同的文件里。一个 `Value` 类在这里，一个训练循环在那里，权重初始化在另一个文件里，学习率调度又在另一个文件里。要训练一个网络，你需要从五个不同的课程中复制粘贴，并手动将它们连接起来。

这就是框架要解决的问题。PyTorch 给你 `nn.Module`、`nn.Sequential`、`optim.Adam`、`DataLoader` 和一个将它们连接在一起的训练循环模式。TensorFlow 给你 `keras.Layer`、`keras.Sequential`、`keras.optimizers.Adam`。这些都不是魔法。它们是组织模式，使得定义、训练和评估网络成为可能，而无需每次都重新发明轮子。

你将用大约500行 Python 代码构建同样的东西。不需要 numpy。不需要外部依赖。一个框架，可以定义任何前馈网络，用 SGD 或 Adam 训练它，批量处理数据，应用 dropout 和批量归一化，使用任何激活函数，并调度学习率。

完成后，你将确切理解当你在 PyTorch 中编写 `model = nn.Sequential(...)` 时会发生什么。你将理解为什么 `model.train()` 和 `model.eval()` 存在。你将理解为什么 `optimizer.zero_grad()` 是一个单独的调用。你会理解所有这一切，因为你构建了所有这一切。

## 概念

### Module 抽象

PyTorch 中的每一层都继承自 `nn.Module`。一个 Module 有三个职责：

1. **forward()** -- 给定输入计算输出
2. **parameters()** -- 返回所有可训练权重
3. **backward()** -- 计算梯度（在 PyTorch 中由 autograd 处理，在我们这里是显式的）

一个 Linear 层是一个 Module。一个 ReLU 激活是一个 Module。一个 dropout 层是一个 Module。一个批归一化层是一个 Module。它们都有相同的接口。

### Sequential 容器

`nn.Sequential` 将 Module 链接在一起。前向传播：将数据依次通过 Module 1、Module 2、Module 3。反向传播：反向链接。容器本身就是一个 Module -- 它有 forward()、parameters() 和 backward()。这是组合模式：一系列 Module 本身就是一个 Module。

### 训练模式 vs 评估模式

Dropout 在训练期间随机将神经元置零，但在评估期间让所有数据通过。批归一化在训练期间使用批次统计量，但在评估期间使用运行均值。`train()` 和 `eval()` 方法切换此行为。每个 Module 都有一个 `training` 标志。

### 优化器

优化器使用梯度更新参数。SGD: `param -= lr * grad`。Adam: 维护动量和方差估计，然后更新。优化器不知道网络架构 -- 它只看到一个扁平的参数列表及其梯度。

### DataLoader

批处理很重要，有两个原因。首先，对于大型问题，你无法将整个数据集放入内存。其次，小批量梯度下降提供的噪声有助于逃离局部极小值。DataLoader 将数据分成批次，并可选择在每个周期之间打乱顺序。

### 框架架构

```mermaid
graph TD
    subgraph "Modules"
        Linear["Linear<br/>W*x + b"]
        ReLU["ReLU<br/>max(0, x)"]
        Sigmoid["Sigmoid<br/>1/(1+e^-x)"]
        Dropout["Dropout<br/>random zero mask"]
        BatchNorm["BatchNorm<br/>normalize activations"]
    end

    subgraph "Containers"
        Sequential["Sequential<br/>chains modules"]
    end

    subgraph "Loss Functions"
        MSE["MSELoss<br/>(pred - target)^2"]
        BCE["BCELoss<br/>binary cross-entropy"]
    end

    subgraph "Optimizers"
        SGD["SGD<br/>param -= lr * grad"]
        Adam["Adam<br/>adaptive moments"]
    end

    subgraph "Data"
        DataLoader["DataLoader<br/>batching + shuffle"]
    end

    Sequential --> |"contains"| Linear
    Sequential --> |"contains"| ReLU
    Sequential --> |"forward/backward"| MSE
    SGD --> |"updates"| Sequential
    DataLoader --> |"feeds"| Sequential
```

### 训练循环

```mermaid
sequenceDiagram
    participant DL as DataLoader
    participant M as Model
    participant L as Loss
    participant O as Optimizer

    loop Each Epoch
        DL->>M: batch of inputs
        M->>M: forward pass (layer by layer)
        M->>L: predictions
        L->>L: compute loss
        L->>M: backward pass (gradients)
        M->>O: parameters + gradients
        O->>M: updated parameters
        O->>O: zero gradients
    end
```

### Module 层次结构

```mermaid
classDiagram
    class Module {
        +forward(x)
        +backward(grad)
        +parameters()
        +train()
        +eval()
    }

    class Linear {
        -weights
        -biases
        +forward(x)
        +backward(grad)
    }

    class ReLU {
        +forward(x)
        +backward(grad)
    }

    class Sequential {
        -modules[]
        +forward(x)
        +backward(grad)
        +parameters()
    }

    Module <|-- Linear
    Module <|-- ReLU
    Module <|-- Sequential
    Sequential *-- Module
```

## 构建它

### 步骤 1: Module 基类

每个层都实现的抽象接口。

```python
class Module:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
```

### 步骤 2: Linear 层

基本的构建块。存储权重和偏置，计算前向的 Wx + b，以及反向的权重/输入梯度。

```python
import math
import random


class Linear(Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        std = math.sqrt(2.0 / fan_in)
        self.weights = [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
        self.biases = [0.0] * fan_out
        self.weight_grads = [[0.0] * fan_in for _ in range(fan_out)]
        self.bias_grads = [0.0] * fan_out
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.input = None

    def forward(self, x):
        self.input = x
        output = []
        for i in range(self.fan_out):
            val = self.biases[i]
            for j in range(self.fan_in):
                val += self.weights[i][j] * x[j]
            output.append(val)
        return output

    def backward(self, grad):
        input_grad = [0.0] * self.fan_in
        for i in range(self.fan_out):
            self.bias_grads[i] += grad[i]
            for j in range(self.fan_in):
                self.weight_grads[i][j] += grad[i] * self.input[j]
                input_grad[j] += grad[i] * self.weights[i][j]
        return input_grad

    def parameters(self):
        params = []
        for i in range(self.fan_out):
            for j in range(self.fan_in):
                params.append((self.weights, i, j, self.weight_grads))
            params.append((self.biases, i, None, self.bias_grads))
        return params
```

### 步骤 3: 激活函数 Module

ReLU、Sigmoid 和 Tanh 作为 Module。每个都为反向传播缓存所需信息。

```python
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = [1.0 if v > 0 else 0.0 for v in x]
        return [max(0.0, v) for v in x]

    def backward(self, grad):
        return [g * m for g, m in zip(grad, self.mask)]


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = []
        for v in x:
            v = max(-500, min(500, v))
            self.output.append(1.0 / (1.0 + math.exp(-v)))
        return self.output

    def backward(self, grad):
        return [g * o * (1 - o) for g, o in zip(grad, self.output)]


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = [math.tanh(v) for v in x]
        return self.output

    def backward(self, grad):
        return [g * (1 - o * o) for g, o in zip(grad, self.output)]
```

### 步骤 4: Dropout Module

在训练期间随机将元素置零。将剩余元素缩放 1/(1-p) 以保持期望值不变。在评估期间不执行任何操作。

```python
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x
        self.mask = [0.0 if random.random() < self.p else 1.0 / (1 - self.p) for _ in x]
        return [v * m for v, m in zip(x, self.mask)]

    def backward(self, grad):
        if self.mask is None:
            return grad
        return [g * m for g, m in zip(grad, self.mask)]
```

### 步骤 5: BatchNorm Module

将激活值按特征归一化为零均值和单位方差。维护用于评估模式的运行统计量。

```python
class BatchNorm(Module):
    def __init__(self, size, momentum=0.1, eps=1e-5):
        super().__init__()
        self.size = size
        self.gamma = [1.0] * size
        self.beta = [0.0] * size
        self.gamma_grads = [0.0] * size
        self.beta_grads = [0.0] * size
        self.running_mean = [0.0] * size
        self.running_var = [1.0] * size
        self.momentum = momentum
        self.eps = eps
        self.x_norm = None
        self.std_inv = None
        self.batch_input = None

    def forward_batch(self, batch):
        batch_size = len(batch)
        output_batch = []

        if self.training:
            mean = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            self.std_inv = [1.0 / math.sqrt(v + self.eps) for v in var]

            self.x_norm = []
            self.batch_input = batch
            for sample in batch:
                normed = [(sample[j] - mean[j]) * self.std_inv[j] for j in range(self.size)]
                self.x_norm.append(normed)
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

            for j in range(self.size):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            std_inv = [1.0 / math.sqrt(v + self.eps) for v in self.running_var]
            for sample in batch:
                normed = [(sample[j] - self.running_mean[j]) * std_inv[j] for j in range(self.size)]
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

        return output_batch

    def forward(self, x):
        result = self.forward_batch([x])
        return result[0]

    def backward(self, grad):
        if self.x_norm is None:
            return grad
        for j in range(self.size):
            self.gamma_grads[j] += self.x_norm[0][j] * grad[j]
            self.beta_grads[j] += grad[j]
        return [grad[j] * self.gamma[j] * self.std_inv[j] for j in range(self.size)]

    def parameters(self):
        params = []
        for j in range(self.size):
            params.append((self.gamma, j, None, self.gamma_grads))
            params.append((self.beta, j, None, self.beta_grads))
        return params
```

### 步骤 6: Sequential 容器

链接 module。前向从左到右，反向从右到左。

```python
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, grad):
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def eval(self):
        self.training = False
        for module in self.modules:
            module.eval()
```

### 步骤 7: 损失函数

MSE 和二元交叉熵。每个都返回损失值并提供返回梯度的 backward()。

```python
class MSELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        n = len(predicted)
        self.loss = sum((p - t) ** 2 for p, t in zip(predicted, target)) / n
        return self.loss

    def backward(self):
        n = len(self.predicted)
        return [2 * (p - t) / n for p, t in zip(self.predicted, self.target)]


class BCELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        eps = 1e-7
        n = len(predicted)
        self.loss = 0
        for p, t in zip(predicted, target):
            p = max(eps, min(1 - eps, p))
            self.loss += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        self.loss /= n
        return self.loss

    def backward(self):
        eps = 1e-7
        n = len(self.predicted)
        grads = []
        for p, t in zip(self.predicted, self.target):
            p = max(eps, min(1 - eps, p))
            grads.append((-t / p + (1 - t) / (1 - p)) / n)
        return grads
```

### 步骤 8: SGD 和 Adam 优化器

两者都接受一个参数列表并使用梯度更新权重。

```python
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.params = parameters
        self.lr = lr

    def step(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                container[i][j] -= self.lr * grad_container[i][j]
            else:
                container[i] -= self.lr * grad_container[i]

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)

    def step(self):
        self.t += 1
        for idx, (container, i, j, grad_container) in enumerate(self.params):
            if j is not None:
                g = grad_container[i][j]
            else:
                g = grad_container[i]

            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * g * g

            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            update = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

            if j is not None:
                container[i][j] -= update
            else:
                container[i] -= update

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0
```

### 步骤 9: DataLoader

将数据分成批次，可选择在每个周期打乱。

```python
class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.data[i] for i in batch_indices]
            inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            yield inputs, targets

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
```

### 步骤 10: 在圆形分类数据上训练一个4层网络

将所有东西连接在一起。定义一个模型，选择一个损失函数，选择一个优化器，运行训练循环。

```python
def make_circle_data(n=500, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], [label]))
    return data


def train():
    random.seed(42)

    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid(),
    )

    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    data = make_circle_data(500)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    loader = DataLoader(train_data, batch_size=16, shuffle=True)

    model.train()

    for epoch in range(100):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_inputs, batch_targets in loader:
            batch_loss = 0
            for x, t in zip(batch_inputs, batch_targets):
                pred = model.forward(x)
                loss = criterion(pred, t)
                batch_loss += loss

                optimizer.zero_grad()
                grad = criterion.backward()
                model.backward(grad)
                optimizer.step()

                predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
                if predicted_class == t[0]:
                    total_correct += 1
                total_samples += 1

            total_loss += batch_loss

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100

        if epoch % 10 == 0 or epoch == 99:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Train Accuracy: {accuracy:.1f}%")

    model.eval()
    correct = 0
    for x, t in test_data:
        pred = model.forward(x)
        predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
        if predicted_class == t[0]:
            correct += 1
    test_accuracy = correct / len(test_data) * 100
    print(f"\nTest Accuracy: {test_accuracy:.1f}% ({correct}/{len(test_data)})")

    return model, test_accuracy
```

## 使用它

这是你刚刚构建的 PyTorch 等价版本：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_predictions = model(test_inputs)
```

结构是相同的。`Sequential`、`Linear`、`ReLU`、`Sigmoid`、`BCELoss`、`Adam`、`zero_grad`、`backward`、`step`、`train`、`eval`。每个概念都一一对应。区别在于 PyTorch 自动处理 autograd（无需在每个 module 中实现 backward()），可以在 GPU 上运行，并且已经优化了多年。但骨架是相同的。

现在当你看到 PyTorch 代码时，你会确切知道每一行正在发生什么。这种理解就是全部意义所在。

## 交付

本课将产生：
- `outputs/prompt-framework-architect.md` -- 一个使用框架抽象设计神经网络架构的提示

## 练习

1. 添加一个 `SoftmaxCrossEntropyLoss` 类用于多类分类。对预测进行 softmax，计算交叉熵损失，并处理组合的反向传播。在一个3类螺旋数据集上进行测试。

2. 在优化器中实现学习率调度：添加一个 `set_lr()` 方法并接入第09课的余弦调度。用预热 + 余弦调度训练圆形分类器，并与常数学习率进行比较。

3. 给 Sequential 添加一个 `save()` 和 `load()` 方法，将所有权重序列化为 JSON 文件并重新加载。验证加载的模型产生的预测与原始模型相同。

4. 在 Adam 优化器中实现权重衰减（L2 正则化）。添加一个 `weight_decay` 参数，在每一步将权重向零收缩。比较 decay=0 和 decay=0.01 的训练情况。

5. 用正确的小批量梯度累积替换逐样本训练循环：累积一个批次中所有样本的梯度，然后除以批次大小并执行一次优化器步骤。测量这是否改变了收敛速度。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Module | "一个层" | 框架中的基本抽象 -- 任何具有 forward()、backward() 和 parameters() 的东西 |
| Sequential | "按顺序堆叠层" | 一个容器，按顺序链接 module，前向按顺序应用，反向反向应用 |
| 前向传播 | "运行网络" | 通过按顺序将输入传递给每个 module 来计算输出 |
| 反向传播 | "计算梯度" | 通过反向传递损失梯度来计算参数梯度 |
| 参数 | "可训练权重" | 网络中优化器可以更新的所有值 -- 权重和偏置 |
| 优化器 | "更新权重的东西" | 使用梯度更新参数的算法，实现 SGD、Adam 或其他规则 |
| DataLoader | "喂数据的东西" | 一个迭代器，将数据集分成批次，可选择在周期之间打乱 |
| 训练模式 | "model.train()" | 一个标志，启用随机行为，如 dropout 和使用批次统计的批归一化 |
| 评估模式 | "model.eval()" | 一个标志，禁用 dropout 并为批归一化使用运行统计量 |
| 梯度清零 | "清除梯度" | 在计算下一批梯度之前将所有参数梯度重置为零 |

## 扩展阅读

- Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019) -- 描述 PyTorch 设计决策的论文
- Chollet, "Deep Learning with Python, Second Edition" (2021) -- 第3章使用相同的 module/层抽象介绍 Keras 内部原理
- Johnson, "Tiny-DNN" (https://github.com/tiny-dnn/tiny-dnn) -- 一个用于理解框架内部原理的、仅头文件的 C++ 深度学习框架