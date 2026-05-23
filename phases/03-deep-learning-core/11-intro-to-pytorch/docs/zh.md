# PyTorch 入门

> 你从活塞和曲轴开始搭建了引擎。现在来学习大家都在用的那个。

**类型：** 构建
**语言：** Python
**先决条件：** 课程 03.10（构建你自己的微型框架）
**时长：** ~75 分钟

## 学习目标

- 使用 PyTorch 的 `nn.Module`、`nn.Sequential` 和 `autograd` 构建和训练神经网络
- 使用 PyTorch 张量、GPU 加速以及标准训练循环（`zero_grad`、`forward`、`loss`、`backward`、`step`）
- 将你的从零开始的微型框架组件转换为 PyTorch 对应版本
- 在同一任务上，对你的纯 Python 框架和 PyTorch 进行性能分析并比较训练速度

## 问题

你已经有了一个可用的微型框架。线性层、ReLU、Dropout、批归一化、Adam、DataLoader、训练循环。它用纯 Python 在一个圆形分类问题上训练一个 4 层网络。

但在同一问题上，它比 PyTorch 慢 500 倍。

你的微型框架使用嵌套的 Python 循环一次处理一个样本。PyTorch 将相同的运算调度到优化的 C++/CUDA 内核上，这些内核在 GPU 上运行。在单块 NVIDIA A100 上，PyTorch 大约用 6 小时就能在 ImageNet（128 万张图像）上训练一个 ResNet-50（2560 万参数）。如果不会先耗尽内存，你的框架在相同任务上大约需要 3000 小时。

速度并不是唯一的差距。你的框架没有 GPU 支持。没有自动微分——你为每个模块手动编写了 `backward()`。没有序列化。没有分布式训练。没有混合精度。除了打印语句外，没有调试梯度流的方法。

PyTorch 填补了所有这些空白。而且它做到了这一点，同时保持了你已经建立的完全相同的思维模型：`Module`、`forward()`、`parameters()`、`backward()`、`optimizer.step()`。概念是一一对应的。语法几乎相同。区别在于，PyTorch 在你从头设计的相同接口背后，封装了十年的系统工程。

## 概念

### 为什么 PyTorch 赢了

2015年，TensorFlow 要求你在运行任何东西之前先定义一个静态计算图。你构建图，编译它，然后将数据输入。调试意味着盯着图的可视化。改变架构意味着从头重建图。

PyTorch 在 2017 年以不同的理念推出：**即时执行**。你写 Python。它立即运行。``y = model(x)`` 实际上就是现在计算 y，而不是“向一个稍后会计算 y 的图添加一个节点”。这意味着标准的 Python 调试工具可以工作。`print()` 有效。`pdb` 有效。前向传播中的 `if/else` 也有效。

到了 2020 年，市场已经做出了选择。PyTorch 在机器学习研究论文中的份额从 2017 年的 7% 增长到 2022 年的超过 75%。Meta、Google DeepMind、OpenAI、Anthropic 和 Hugging Face 都将 PyTorch 作为主要框架。TensorFlow 2.x 因此采用了即时执行——这隐晦地承认了 PyTorch 的设计是正确的。

教训是：开发体验具有复利效应。一个慢 10% 但调试速度快 50% 的框架每次都会胜出。

### 张量

张量是一个具有三个关键属性的多维数组：形状、数据类型和设备。

```python
import torch

x = torch.zeros(3, 4)           # shape: (3, 4), dtype: float32, device: cpu
x = torch.randn(2, 3, 224, 224) # batch of 2 RGB images, 224x224
x = torch.tensor([1, 2, 3])     # from a Python list
```

**形状** 是维度。标量是 `()`，向量是 `(n,)`，矩阵是 `(m, n)`，一批图像是 `(batch, channels, height, width)`。

**数据类型** 控制精度和内存。

| dtype | 位数 | 范围 | 用例 |
|-------|------|-------|----------|
| float32 | 32 | 约7位小数 | 默认训练 |
| float16 | 16 | 约3.3位小数 | 混合精度 |
| bfloat16 | 16 | 范围与 float32 相同，精度更低 | 大语言模型训练 |
| int8 | 8 | -128 到 127 | 量化推理 |

**设备** 决定了计算发生在哪里。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 4, device=device)
x = x.to("cuda")
x = x.cpu()
```

每个操作都要求所有张量在同一个设备上。这是 PyTorch 初学者最常遇到的错误：``RuntimeError: Expected all tensors to be on the same device``。解决方法是将所有内容在计算前移到同一个设备上。

**重塑** 是常数时间操作——它改变的是元数据，而不是数据本身。

```python
x = torch.randn(2, 3, 4)
x.view(2, 12)      # reshape to (2, 12) -- must be contiguous
x.reshape(6, 4)    # reshape to (6, 4) -- works always
x.permute(2, 0, 1) # reorder dimensions
x.unsqueeze(0)     # add dimension: (1, 2, 3, 4)
x.squeeze()        # remove size-1 dimensions
```

### 自动微分 (Autograd)

你的微型框架要求你为每个模块实现 `backward()`。PyTorch 不需要。它记录张量上的每一个操作，形成一个**有向无环图**（计算图），然后反向遍历该图以自动计算梯度。

```mermaid
graph LR
    x["x (leaf)"] --> mul["*"]
    w["w (leaf, requires_grad)"] --> mul
    mul --> add["+"]
    b["b (leaf, requires_grad)"] --> add
    add --> loss["loss"]
    loss --> |".backward()"| add
    add --> |"grad"| b
    add --> |"grad"| mul
    mul --> |"grad"| w
```

与你的框架的关键区别：PyTorch 使用基于**磁带**的自动微分。在前向传播期间，每个操作都被追加到一个“磁带”上。调用 ``.backward()`` 会反向回放这个磁带。

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2 + 3 * x
z = y.sum()
z.backward()
print(x.grad)  # dz/dx = 2x + 3
```

自动微分的三条规则：

1. 只有具有 ``requires_grad=True`` 的**叶子张量**才会累积梯度。
2. 梯度默认是累积的——在每次反向传播前调用 ``optimizer.zero_grad()``。
3. ``torch.no_grad()`` 禁用梯度跟踪（在评估时使用）。

### `nn.Module`

``nn.Module`` 是 PyTorch 中每个神经网络组件的基类。你在第 10 课中已经构建了这个抽象。PyTorch 的版本增加了自动参数注册、递归模块发现、设备管理和状态字典序列化。

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

当你在 ``__init__`` 中将一个 ``nn.Module`` 或 ``nn.Parameter`` 赋值为属性时，PyTorch 会自动注册它。``model.parameters()`` 递归地收集每个已注册的参数。这就是为什么你再也不需要像在微型框架中那样手动收集权重了。

关键构建模块：

| 模块 | 功能 | 参数 |
|--------|-------------|------------|
| `nn.Linear(in, out)` | Wx + b | in*out + out |
| `nn.Conv2d(in_ch, out_ch, k)` | 2D卷积 | in_ch*out_ch*k*k + out_ch |
| `nn.BatchNorm1d(features)` | 激活归一化 | 2 * features |
| `nn.Dropout(p)` | 随机置零 | 0 |
| `nn.ReLU()` | max(0, x) | 0 |
| `nn.GELU()` | 高斯误差线性 | 0 |
| `nn.Embedding(vocab, dim)` | 查找表 | vocab * dim |
| `nn.LayerNorm(dim)` | 逐样本归一化 | 2 * dim |

### 损失函数和优化器

PyTorch 提供了你构建的所有功能的生产就绪版本。

**损失函数** (来自 ``torch.nn``)：

| 损失函数 | 任务 | 输入 |
|------|------|-------|
| `nn.MSELoss()` | 回归 | 任意形状 |
| `nn.CrossEntropyLoss()` | 多类别分类 | Logits（非 softmax） |
| `nn.BCEWithLogitsLoss()` | 二分类 | Logits（非 sigmoid） |
| `nn.L1Loss()` | 回归（更鲁棒） | 任意形状 |
| `nn.CTCLoss()` | 序列对齐 | 对数概率 |

注意：``CrossEntropyLoss`` 内部结合了 ``LogSoftmax`` + ``NLLLoss``。请传递原始 logits，而不是 softmax 输出。这是一个常见的错误，会静默地产生错误的梯度。

**优化器** (来自 ``torch.optim``)：

| 优化器 | 使用场景 | 典型学习率 |
|-----------|-------------|-----------|
| `SGD(params, lr, momentum)` | CNN、调优良好的流程 | 0.01--0.1 |
| `Adam(params, lr)` | 默认起点 | 1e-3 |
| `AdamW(params, lr, weight_decay)` | Transformer、微调 | 1e-4--1e-3 |
| `LBFGS(params)` | 小规模、二阶方法 | 1.0 |

### 训练循环

每个 PyTorch 训练循环都遵循相同的 5 步模式。你在第 10 课已经知道了这个。

```mermaid
sequenceDiagram
    participant D as DataLoader
    participant M as Model
    participant L as Loss fn
    participant O as Optimizer

    loop Each Epoch
        D->>M: batch = next(dataloader)
        M->>L: predictions = model(batch)
        L->>L: loss = criterion(predictions, targets)
        L->>M: loss.backward()
        O->>M: optimizer.step()
        O->>O: optimizer.zero_grad()
    end
```

规范模式：

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

批次循环内的五行代码。训练了 GPT-4、Stable Diffusion 和 LLaMA 的五行代码。架构会变。数据会变。这五行代码不会变。

### Dataset 和 DataLoader

PyTorch 的 ``Dataset`` 是一个抽象类，有两个方法：``__len__`` 和 ``__getitem__``。``DataLoader`` 用批处理、洗牌和多进程数据加载来包装它。

```python
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
```

``num_workers=4`` 生成 4 个进程来并行加载数据，同时 GPU 训练当前批次。在磁盘受限的工作负载（大图像、音频）上，仅此一项就能使训练速度翻倍。

### GPU 训练

将模型移到 GPU：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

这会递归地将每个参数和缓冲区移动到 GPU。然后在训练期间移动每个批次：

```python
inputs, targets = inputs.to(device), targets.to(device)
```

**混合精度** 通过在 float16 中运行前向/反向传播，同时保持主权重为 float32，将内存使用减半，并在现代 GPU（A100、H100、RTX 4090）上将吞吐量翻倍：

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in loader:
    with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 对比：微型框架 vs PyTorch vs JAX

| 特性 | 微型框架 (L10) | PyTorch | JAX |
|---------|---------------------|---------|-----|
| 自动微分 | 手动 `backward()` | 基于磁带的 autograd | 函数式变换 |
| 执行方式 | 即时执行（Python 循环） | 即时执行（C++ 内核） | 跟踪 + JIT 编译 |
| GPU 支持 | 无 | 有 (CUDA, ROCm, MPS) | 有 (CUDA, TPU) |
| 速度 (MNIST MLP) | ~300 秒/轮 | ~0.5 秒/轮 | ~0.3 秒/轮 |
| 模块系统 | 自定义 Module 类 | `nn.Module` | 无状态函数 (Flax/Equinox) |
| 调试 | `print()` | `print()`, `pdb`, `breakpoint()` | 更难（JIT 跟踪破坏 print） |
| 生态系统 | 无 | Hugging Face, Lightning, timm | Flax, Optax, Orbax |
| 学习曲线 | 你构建的 | 中等 | 陡峭（函数式范式） |
| 生产用途 | 玩具问题 | Meta, OpenAI, Anthropic, HF | Google DeepMind, Midjourney |

## 动手构建

使用 PyTorch 原生组件训练一个三层 MLP 在 MNIST 上。不使用高级包装器。不使用 ``torchvision.datasets``。我们自己下载并解析原始数据。

### 步骤 1：从原始文件加载 MNIST

MNIST 以 4 个 gzip 文件的形式发布：训练图像 (60,000 x 28 x 28)、训练标签、测试图像 (10,000 x 28 x 28)、测试标签。我们下载它们并解析二进制格式。

```python
import torch
import torch.nn as nn
import struct
import gzip
import urllib.request
import os

def download_mnist(path="./mnist_data"):
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    os.makedirs(path, exist_ok=True)
    for f in files:
        filepath = os.path.join(path, f)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(base_url + f, filepath)

def load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        images = images.reshape(num, rows * cols).float() / 255.0
    return images

def load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return labels
```

### 步骤 2：定义模型

一个三层 MLP：784 -> 256 -> 128 -> 10。ReLU 激活。Dropout 用于正则化。没有批归一化以保持简单。

```python
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)
```

输出层产生 10 个原始 logits（每个数字一个）。没有 softmax——``CrossEntropyLoss`` 内部会处理。

参数数量：784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 235,146。按现代标准来说很小。GPT-2 small 有 1.24 亿。这个模型几秒钟就能训好。

### 步骤 3：训练循环

规范的前向-损失-反向-步进模式。

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total
```

注意评估时的 ``torch.no_grad()``。这会禁用自动微分，减少内存使用并加速推理。如果没有它，PyTorch 会构建一个你永远不会用到的计算图。

### 步骤 4：连接所有组件

```python
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_mnist()
    train_images = load_images("./mnist_data/train-images-idx3-ubyte.gz")
    train_labels = load_labels("./mnist_data/train-labels-idx1-ubyte.gz")
    test_images = load_images("./mnist_data/t10k-images-idx3-ubyte.gz")
    test_labels = load_labels("./mnist_data/t10k-labels-idx1-ubyte.gz")

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {num_params:,}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print()

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        print(
            f"Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print(f"\nModel saved to mnist_mlp.pt")
    print(f"Final test accuracy: {test_acc:.4f}")
```

10 轮后预期输出：约 97.8% 测试准确率。在 CPU 上训练时间：约 30 秒。在 GPU 上：约 5 秒。在你的微型框架上使用相同架构：约 45 分钟。

## 使用它

### 快速对比：微型框架 vs PyTorch

| 微型框架 (第 10 课) | PyTorch |
|---------------------------|---------|
| ``model = Sequential(Linear(784, 256), ReLU(), ...)`` | ``model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), ...)`` |
| ``pred = model.forward(x)`` | ``pred = model(x)`` |
| ``optimizer.zero_grad()`` | ``optimizer.zero_grad()`` |
| ``grad = criterion.backward()`` 然后 ``model.backward(grad)`` | ``loss.backward()`` |
| ``optimizer.step()`` | ``optimizer.step()`` |
| 无 GPU | ``model.to("cuda")`` |
| 为每个模块手动编写 backward | Autograd 处理一切 |

接口几乎一模一样。区别在于底层的一切。

### 保存和加载模型

```python
torch.save(model.state_dict(), "model.pt")

model = MNISTModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()
```

始终保存 ``state_dict()``（参数字典），而不是模型对象。保存模型对象使用 pickle，当你重构代码时会中断。状态字典是可移植的。

### 学习率调度

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
)
for epoch in range(10):
    train_one_epoch(model, train_loader, criterion, optimizer, device)
    scheduler.step()
```

PyTorch 提供了 15 种以上的调度器：`StepLR`、`ExponentialLR`、`CosineAnnealingLR`、`OneCycleLR`、`ReduceLROnPlateau`。它们都插入相同的优化器接口。

## 发布

本课产生两个成果：

- ``outputs/prompt-pytorch-debugger.md`` —— 一个用于诊断常见 PyTorch 训练失败的提示
- ``outputs/skill-pytorch-patterns.md`` —— PyTorch 训练模式的技能参考

## 练习

1.  **添加批归一化。** 在每个线性层之后（激活函数之前）插入 ``nn.BatchNorm1d``。比较测试准确率和训练速度与仅使用 Dropout 的版本。批归一化应在更少轮数内达到 98%+ 的准确率。

2.  **实现一个学习率查找器。** 用指数增长的学习率（从 1e-7 到 1.0）训练一个轮次。绘制损失 vs 学习率曲线。最佳学习率在损失开始上升之前。用这个为 MNIST 模型选择一个更好的学习率。

3.  **通过混合精度移植到 GPU。** 在训练循环中添加 ``torch.amp.autocast`` 和 ``GradScaler``。测量在 GPU 上使用和不使用混合精度的吞吐量（样本/秒）。在 A100 上，预期有约 2 倍的加速。

4.  **构建一个自定义 Dataset。** 下载 Fashion-MNIST（格式与 MNIST 相同，但内容是服装）。实现一个带有 ``__getitem__`` 和 ``__len__`` 的 ``FashionMNISTDataset(Dataset)`` 类。训练相同的 MLP 并比较准确率。Fashion-MNIST 更难——预期约 88% vs ~98%。

5.  **用 SGD + 动量替换 Adam。** 用 ``SGD(params, lr=0.01, momentum=0.9)`` 训练。比较收敛曲线。然后添加一个 ``CosineAnnealingLR`` 调度器，看看 SGD 在第 10 轮时是否能追上 Adam。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| 张量 (Tensor) | “一个多维数组” | 一个带类型的、设备感知的数组，其每个操作都内置了对自动微分的支持 |
| 自动微分 (Autograd) | “自动反向传播” | 一个基于磁带的系统，在前向传播期间记录操作，然后反向回放它们以计算精确的梯度 |
| `nn.Module` | “一个层” | 任何可微分计算块的基类——注册参数，支持嵌套，处理训练/评估模式 |
| `state_dict` | “模型权重” | 一个有序字典，将参数名称映射到张量——训练模型的可移植、可序列化表示 |
| `.backward()` | “计算梯度” | 反向遍历计算图，为每个 `requires_grad=True` 的叶子张量计算和累积梯度 |
| `.to(device)` | “移到 GPU” | 递归地将所有参数和缓冲区转移到指定设备 (CPU, CUDA, MPS) |
| `DataLoader` | “数据管线” | 一个迭代器，对来自 Dataset 的数据进行批处理、洗牌，并可选地并行加载 |
| 混合精度 (Mixed precision) | “使用 float16” | 用 float16 进行前向/反向传播以提速，同时保持 float32 主权重以确保数值稳定性 |
| 即时执行 (Eager execution) | “现在运行它” | 操作在调用时立即执行，而不是推迟到后续的编译步骤——这是 PyTorch 区别于 TF 1.x 的核心设计选择 |
| `zero_grad` | “重置梯度” | 在下一次反向传播之前将所有参数梯度置零，因为 PyTorch 默认累积梯度 |

## 延伸阅读

- Paszke 等人，"PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019) —— 解释 PyTorch 设计权衡的原论文
- PyTorch 教程："Learning PyTorch with Examples" (https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) —— 从张量到 nn.Module 的官方路径
- PyTorch 性能调优指南 (https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) —— 混合精度、DataLoader workers、固定内存和其他生产优化
- Horace He, "Making Deep Learning Go Brrrr" (https://horace.io/brrr_intro.html) —— 为什么 GPU 训练快，附带 PyTorch 特定的优化策略