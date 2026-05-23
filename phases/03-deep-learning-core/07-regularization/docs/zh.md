# 正则化

> 你的模型在训练数据上达到99%准确率，在测试数据上却只有60%。它只是记住了数据，而不是学会了规律。正则化是为了强制泛化能力而施加在模型复杂性上的税。

**类型：** 构建
**语言：** Python
**先决课程：** 第03.06课（优化器）
**时间：** ~75分钟

## 学习目标

- 从头实现带反转缩放的Dropout、L2权重衰减、批归一化、层归一化和RMSNorm
- 测量训练-测试准确率差距，并通过正则化实验诊断过拟合
- 解释为什么Transformer使用LayerNorm而不是BatchNorm，以及为什么现代LLM偏好RMSNorm
- 根据过拟合的严重程度，应用正确的正则化技术组合

## 问题所在

一个拥有足够参数的神经网络可以记忆任何数据集。这不是假设——Zhang等人（2017）通过在随机标签的ImageNet上训练标准网络证明了这一点。网络在完全随机的标签分配上达到了接近零的训练损失。它们记忆了百万个没有规律可学的随机输入-输出对。训练损失完美。测试准确率为零。

这就是过拟合问题，并且随着模型变大而加剧。GPT-3拥有1750亿参数。训练集大约有5000亿token。拥有如此多的参数，模型有足够的容量逐字记忆训练数据中的大部分内容。没有正则化，它只会重复训练样本，而不是学习可泛化的模式。

训练性能和测试性能之间的差距就是过拟合差距。本课中的每项技术都从不同角度攻击这个差距。Dropout迫使网络不依赖任何单个神经元。权重衰减防止任何单个权重变得过大。批归一化平滑损失函数景观，使优化器能找到更平坦、更易泛化的极小值。层归一化做同样的事，但在批归一化失效的情况下工作（小批量、变长序列）。RMSNorm通过省略均值计算，速度提升了10%。每项技术都很简单。组合起来，它们就是记忆型模型和泛化型模型之间的区别。

## 核心概念

### 过拟合谱系

每个模型都处于从欠拟合（太简单无法捕捉模式）到过拟合（太复杂以至于捕捉了噪声）的谱系上。最佳点介于两者之间，正则化从过拟合一侧将模型推向最佳点。

```mermaid
graph LR
    Under["Underfitting<br/>Train: 60%<br/>Test: 58%<br/>Model too simple"] --> Good["Good Fit<br/>Train: 95%<br/>Test: 92%<br/>Generalizes well"]
    Good --> Over["Overfitting<br/>Train: 99.9%<br/>Test: 65%<br/>Memorized noise"]

    Dropout["Dropout"] -->|"Pushes left"| Over
    WD["Weight Decay"] -->|"Pushes left"| Over
    BN["BatchNorm"] -->|"Pushes left"| Over
    Aug["Data Augmentation"] -->|"Pushes left"| Over
```

### Dropout

最简单、解释最优雅的正则化技术。训练期间，以概率p随机将每个神经元的输出置零。

```
output = activation(z) * mask    where mask[i] ~ Bernoulli(1 - p)
```

当p=0.5时，每次前向传播一半神经元被置零。网络必须学习冗余表示，因为它无法预测哪些神经元可用。这防止了共适应——神经元学会依赖于特定其他神经元的存在。

集成解释：一个有N个神经元和Dropout的网络创建了2^N个可能的子网络（神经元开启或关闭的每种组合）。使用Dropout训练近似于同时训练所有2^N个子网络，每个在不同的小批量上。测试时，你使用所有神经元（无Dropout）并将输出缩放(1 - p)以匹配训练期间的期望值。这相当于对2^N个子网络的预测取平均——从单个模型中获得的庞大集成。

实践中，缩放在训练期间而非测试期间应用（反转Dropout）：

```
During training:  output = activation(z) * mask / (1 - p)
During testing:   output = activation(z)   (no change needed)
```

这更简洁，因为测试代码完全不需要知道Dropout。

默认比率：Transformer用p = 0.1，MLP用p = 0.5，CNN用p = 0.2-0.3。更高的Dropout = 更强的正则化 = 更大的欠拟合风险。

### 权重衰减（L2正则化）

将所有权重的平方和加到损失中：

```
total_loss = task_loss + (lambda / 2) * sum(w_i^2)
```

正则化项的梯度是 lambda * w。这意味着在每一步，每个权重都按与其大小成比例的比例向零缩小。大权重受到更多惩罚。模型被推向没有单个权重占主导的解。

这对泛化为何有帮助：过拟合模型往往有大权重，这些权重放大训练数据中的噪声。权重衰减使权重保持较小，这限制了模型的有效容量，并迫使其依赖稳健、可泛化的特征，而不是记忆的细节。

lambda超参数控制强度。典型值：

- AdamW用于Transformer时：0.01
- SGD用于CNN时：1e-4
- 严重过拟合模型：0.1

如第06课所讨论：权重衰减和L2正则化在SGD中等价，但在Adam中不等价。用Adam训练时，始终使用AdamW（解耦权重衰减）。

### 批归一化

在传递给下一层之前，对每个层的输出在小批量上进行归一化。

对于某一层的小批量激活：

```
mu = (1/B) * sum(x_i)           (batch mean)
sigma^2 = (1/B) * sum((x_i - mu)^2)   (batch variance)
x_hat = (x_i - mu) / sqrt(sigma^2 + eps)   (normalize)
y = gamma * x_hat + beta        (scale and shift)
```

Gamma和beta是可学习参数，允许网络在最优情况下撤销归一化。如果没有它们，你将强制每层的输出都是零均值单位方差，这可能不是网络想要的。

**训练与推理分离：** 训练期间，mu和sigma来自当前小批量。推理期间，你使用训练期间累积的运行平均值（动量=0.1的指数移动平均，即90%旧值 + 10%新值）。

批归一化为何有效仍有争议。原论文声称它减少了"内部协变量偏移"（层输入分布随着前序层更新而变化）。Santurkar等人（2018）表明这个解释是错误的。实际原因：批归一化使损失景观更平滑。梯度更具预测性，Lipschitz常数更小，优化器可以安全地采取更大的步长。这就是为什么批归一化允许你使用更高的学习率并更快收敛。

批归一化有一个根本限制：它依赖于批量统计。批量大小为1时，均值和方差无意义。小批量（< 32）时，统计量是噪声且损害性能。这对于目标检测（内存限制批量大小）和语言建模（序列长度变化）等任务很重要。

### 层归一化

跨特征而非跨批量进行归一化。对于单个样本：

```
mu = (1/D) * sum(x_j)           (feature mean)
sigma^2 = (1/D) * sum((x_j - mu)^2)   (feature variance)
x_hat = (x_j - mu) / sqrt(sigma^2 + eps)
y = gamma * x_hat + beta
```

D是特征维度。每个样本独立归一化——不依赖于批量大小。这就是为什么Transformer使用LayerNorm而不是BatchNorm。序列长度可变，批量大小通常很小（或在生成时为1），并且训练和推理期间的计算相同。

Transformer中的LayerNorm应用于每个自注意力块和每个前馈块之后（后归一化），或之前（前归一化，对训练更稳定）。

### RMSNorm

不减均值的LayerNorm。由Zhang & Sennrich（2019）提出。

```
rms = sqrt((1/D) * sum(x_j^2))
y = gamma * x / rms
```

就是这样。没有均值计算，没有beta参数。观察结果：LayerNorm中的重新中心化（减均值）对模型性能贡献很小，但耗费计算。移除它可以在精度相同的情况下减少约10%的开销。

LLaMA、LLaMA 2、LLaMA 3、Mistral和大多数现代LLM使用RMSNorm而不是LayerNorm。在数十亿参数和数万亿token的规模下，10%的节省是显著的。

### 归一化对比

```mermaid
graph TD
    subgraph "Batch Normalization"
        BN_D["Normalize across BATCH<br/>for each feature"]
        BN_S["Batch: [x1, x2, x3, x4]<br/>Feature 1: normalize [x1f1, x2f1, x3f1, x4f1]"]
        BN_P["Needs batch > 32<br/>Different train vs eval<br/>Used in CNNs"]
    end
    subgraph "Layer Normalization"
        LN_D["Normalize across FEATURES<br/>for each sample"]
        LN_S["Sample x1: normalize [f1, f2, f3, f4]"]
        LN_P["Batch-independent<br/>Same train vs eval<br/>Used in Transformers"]
    end
    subgraph "RMS Normalization"
        RN_D["Like LayerNorm<br/>but skip mean subtraction"]
        RN_S["Just divide by RMS<br/>No centering"]
        RN_P["10% faster than LayerNorm<br/>Same accuracy<br/>Used in LLaMA, Mistral"]
    end
```

### 数据增强作为正则化

不是模型修改，而是数据修改。在保持标签不变的情况下变换训练输入：

- 图像：随机裁剪、翻转、旋转、颜色抖动、遮挡
- 文本：同义词替换、回译、随机删除
- 音频：时间拉伸、音高偏移、噪声添加

效果与正则化相同：它增加了训练集的有效大小，使模型更难记忆特定样本。一个只看到每张原始图像一次的模型可以记忆它。一个看到每张图像50个增强版本的模型被迫学习不变结构。

### 早停

最简单的正则化器：当验证损失开始增加时停止训练。此时模型尚未过拟合。实践中，你每个epoch跟踪验证损失，保存最佳模型，并继续训练一个"耐心"窗口（通常5-20个epoch）。如果验证损失在耐心窗口内没有改善，你停止并加载最佳保存的模型。

### 何时应用什么

```mermaid
flowchart TD
    Gap{"Train-test<br/>accuracy gap?"} -->|"> 10%"| Heavy["Heavy regularization"]
    Gap -->|"5-10%"| Medium["Moderate regularization"]
    Gap -->|"< 5%"| Light["Light regularization"]

    Heavy --> D5["Dropout p=0.3-0.5"]
    Heavy --> WD2["Weight decay 0.01-0.1"]
    Heavy --> Aug["Aggressive data augmentation"]
    Heavy --> ES["Early stopping"]

    Medium --> D3["Dropout p=0.1-0.2"]
    Medium --> WD1["Weight decay 0.001-0.01"]
    Medium --> Norm["BatchNorm or LayerNorm"]

    Light --> D1["Dropout p=0.05-0.1"]
    Light --> WD0["Weight decay 1e-4"]
```

## 构建它

### 步骤1：Dropout（训练和评估模式）

```python
import random
import math


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        self.mask = None

    def forward(self, x):
        if not self.training:
            return list(x)
        self.mask = []
        output = []
        for val in x:
            if random.random() < self.p:
                self.mask.append(0)
                output.append(0.0)
            else:
                self.mask.append(1)
                output.append(val / (1 - self.p))
        return output

    def backward(self, grad_output):
        grads = []
        for g, m in zip(grad_output, self.mask):
            if m == 0:
                grads.append(0.0)
            else:
                grads.append(g / (1 - self.p))
        return grads
```

### 步骤2：L2权重衰减

```python
def l2_regularization(weights, lambda_reg):
    penalty = 0.0
    for w in weights:
        penalty += w * w
    return lambda_reg * 0.5 * penalty

def l2_gradient(weights, lambda_reg):
    return [lambda_reg * w for w in weights]
```

### 步骤3：批归一化

```python
class BatchNorm:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        self.training = True
        self.num_features = num_features

    def forward(self, batch):
        batch_size = len(batch)
        if self.training:
            mean = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            for j in range(self.num_features):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            mean = list(self.running_mean)
            var = list(self.running_var)

        self.x_hat = []
        output = []
        for sample in batch:
            normalized = []
            out_sample = []
            for j in range(self.num_features):
                x_h = (sample[j] - mean[j]) / math.sqrt(var[j] + self.eps)
                normalized.append(x_h)
                out_sample.append(self.gamma[j] * x_h + self.beta[j])
            self.x_hat.append(normalized)
            output.append(out_sample)
        return output
```

### 步骤4：层归一化

```python
class LayerNorm:
    def __init__(self, num_features, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)

        self.x_hat = []
        output = []
        for j in range(self.num_features):
            x_h = (x[j] - mean) / math.sqrt(var + self.eps)
            self.x_hat.append(x_h)
            output.append(self.gamma[j] * x_h + self.beta[j])
        return output
```

### 步骤5：RMSNorm

```python
class RMSNorm:
    def __init__(self, num_features, eps=1e-6):
        self.gamma = [1.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        rms = math.sqrt(sum(xi * xi for xi in x) / len(x) + self.eps)
        output = []
        for j in range(self.num_features):
            output.append(self.gamma[j] * x[j] / rms)
        return output
```

### 步骤6：有正则化和无正则化的训练

```python
def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class RegularizedNetwork:
    def __init__(self, hidden_size=16, lr=0.05, dropout_p=0.0, weight_decay=0.0):
        random.seed(0)
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout_p = dropout_p
        self.weight_decay = weight_decay
        self.dropout = Dropout(p=dropout_p) if dropout_p > 0 else None

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x, training=True):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        if self.dropout and training:
            self.dropout.training = True
            self.h = self.dropout.forward(self.h)
        elif self.dropout:
            self.dropout.training = False
            self.h = self.dropout.forward(self.h)

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            self.w2[i] -= self.lr * (d_out * self.h[i] + self.weight_decay * self.w2[i])
            for j in range(2):
                self.w1[i][j] -= self.lr * (d_h * self.x[j] + self.weight_decay * self.w1[i][j])
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def evaluate(self, data):
        correct = 0
        total_loss = 0.0
        for x, y in data:
            pred = self.forward(x, training=False)
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            if (pred >= 0.5) == (y >= 0.5):
                correct += 1
        return total_loss / len(data), correct / len(data) * 100

    def train_model(self, train_data, test_data, epochs=300):
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in train_data:
                pred = self.forward(x, training=True)
                self.backward(y)
                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            train_loss = total_loss / len(train_data)
            train_acc = correct / len(train_data) * 100
            test_loss, test_acc = self.evaluate(test_data)
            history.append((train_loss, train_acc, test_loss, test_acc))
            if epoch % 75 == 0 or epoch == epochs - 1:
                gap = train_acc - test_acc
                print(f"    Epoch {epoch:3d}: train_acc={train_acc:.1f}%, test_acc={test_acc:.1f}%, gap={gap:.1f}%")
        return history
```

## 使用它

PyTorch将所有归一化和正则化作为模块提供：

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10),
)

model.train()
out_train = model(torch.randn(32, 784))

model.eval()
out_test = model(torch.randn(1, 784))
```

`model.train()` / `model.eval()` 切换至关重要。它开启/关闭Dropout，并告诉BatchNorm使用批量统计还是运行统计。推理前忘记`model.eval()`是深度学习中最常见的错误之一。你的测试准确率会随机波动，因为Dropout仍然活跃，BatchNorm使用小批量统计。

对于Transformer，模式不同：

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.ff(x))
        return x
```

LayerNorm，而不是BatchNorm。Dropout p=0.1，而不是p=0.5。这些是Transformer的默认设置。

## 发布它

本课产出：
- `outputs/prompt-regularization-advisor.md` -- 一个诊断过拟合并推荐正确正则化策略的提示

## 练习

1. 为2D数据实现空间Dropout：不是丢弃单个神经元，而是丢弃整个特征通道。通过将连续的特征组视为通道并丢弃整个组来模拟这一点。在圆圈数据集（hidden_size=32）上，将训练-测试差距与标准Dropout进行比较。

2. 结合第05课的标签平滑和本课的Dropout实现。用四种配置训练：两者都没有，只有Dropout，只有标签平滑，两者都有。测量每种情况的最终训练-测试准确率差距。哪种组合差距最小？

3. 在你的圆圈数据集网络的隐藏层和激活之间添加一个BatchNorm层。在学习率0.01、0.05和0.1下，分别在有和没有BatchNorm的情况下训练。BatchNorm应能在原始网络发散的更高学习率下实现稳定训练。

4. 实现早停：每个epoch跟踪测试损失，保存最佳权重，如果测试损失20个epoch没有改善则停止。运行正则化网络1000个epoch。报告最佳测试准确率出现在哪个epoch，以及你节省了多少计算周期。

5. 在4层网络（而不仅仅是2层）上比较LayerNorm与RMSNorm。用相同的权重初始化两者。训练200个epoch，比较最终准确率、训练速度（每epoch时间）和第一层的梯度幅度。验证RMSNorm在准确率相同的情况下更快。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|------------|--------------|
| 过拟合 | "模型记住了数据" | 当模型的训练性能显著超过测试性能时，表明它学习了噪声而非信号 |
| 正则化 | "防止过拟合" | 任何约束模型复杂性以提高泛化能力的技术：Dropout、权重衰减、归一化、增强 |
| Dropout | "随机神经元删除" | 训练期间以概率p将随机神经元置零，迫使学习冗余表示；等价于训练一个集成 |
| 权重衰减 | "L2惩罚" | 通过每步减去lambda * w来将所有权重向零收缩；通过权重大小惩罚复杂性 |
| 批归一化 | "每批归一化" | 使用批量统计在训练期间和运行平均在推理期间，跨批量维度归一化层输出 |
| 层归一化 | "每样本归一化" | 跨每个样本内的特征进行归一化；不依赖批量，用于批量大小变化的Transformer中 |
| RMSNorm | "不带均值的LayerNorm" | 均方根归一化；从LayerNorm中去掉均值减法，以10%的速度提升获得相同精度 |
| 早停 | "在过拟合前停止" | 当验证损失停止改善时停止训练；最简单的正则化器，常与其他技术一起使用 |
| 数据增强 | "从少量数据生成更多数据" | 变换训练输入（翻转、裁剪、噪声）以增加有效数据集大小并强制不变性学习 |
| 泛化差距 | "训练-测试分割" | 训练和测试性能之间的差异；正则化旨在最小化这个差距 |

## 扩展阅读

- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014) -- 原始的Dropout论文，包含集成解释和大量实验
- Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (2015) -- 引入BatchNorm及其训练过程，深度学习论文中被引用最多的之一
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019) -- 表明RMSNorm以更少的计算匹配LayerNorm精度；被LLaMA和Mistral采用
- Zhang et al., "Understanding Deep Learning Requires Rethinking Generalization" (2017) -- 展示神经网络可以记忆随机标签的里程碑论文，挑战了传统的泛化观点