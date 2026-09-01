```markdown
# 图像分类

> 分类器是从像素到类别概率分布的映射函数。其余都是辅助设施。

**类型：** 构建
**语言：** Python
**前置要求：** 第二阶段第09课（模型评估）、第三阶段第10课（迷你框架）、第四阶段第03课（CNN）
**时长：** 约75分钟

## 学习目标

- 在 CIFAR-10 上构建端到端的图像分类流程：数据集、数据增强、模型、训练循环、评估
- 解释每个组件（dataloader、loss、optimizer、scheduler、augmentation）的作用，并预测破坏其中任何一个时 loss 曲线的表现
- 从零实现 mixup、cutout 和 label smoothing，并说明每种方法在何时值得使用
- 阅读混淆矩阵和每类精确率/召回率表，以诊断数据集合模型中超出聚合准确率的故障

## 问题

每一个落地视觉任务在某个层面都归结为图像分类。检测对区域进行分类。分割对像素进行分类。检索按与类别质心的相似度排序。掌握分类的正确方式——数据集循环、增强策略、loss、评估——是将技能迁移到阶段中其他所有任务的关键。

大多数分类 bug 不在模型中。它们存在于流水线里：损坏的归一化、未打乱的训练集、扭曲标签的增强、被训练数据污染的验证集、在第30个 epoch 后无声发散的学习率。一个在正确设置下能达到 93% 的 CNN，在使用有缺陷的设置时通常只能得 70-75%，而 loss 曲线看起来始终合理。

本课手工连接整个流水线，使每个部分都可检查。你不会使用任何来自 `torchvision.datasets` 可能隐藏 bug 的东西。

## 概念

### 分类流水线

```mermaid
flowchart LR
    A["Dataset<br/>(images + labels)"] --> B["Augment<br/>(random transforms)"]
    B --> C["Normalise<br/>(mean/std)"]
    C --> D["DataLoader<br/>(batch + shuffle)"]
    D --> E["Model<br/>(CNN)"]
    E --> F["Logits<br/>(N, C)"]
    F --> G["Cross-entropy loss"]
    F --> H["Argmax<br/>at eval"]
    G --> I["Backward"]
    I --> J["Optimizer step"]
    J --> K["Scheduler step"]
    K --> E

    style A fill:#dbeafe,stroke:#2563eb
    style E fill:#fef3c7,stroke:#d97706
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a
```

这个循环中的每一行都是可能藏 bug 的地方。交叉熵接受原始 logits，而不是 softmax 输出，所以在 loss 之前对 `model(x).softmax()` 会悄悄计算错误的梯度。增强仅应用于输入，不适用于标签——除了 mixup，它同时混合两者。`optimizer.zero_grad()` 必须在每个 step 执行一次；跳过它会导致梯度累积，看起来像学习率极度不稳定。这些 bug 中的每一个都会使学习曲线变平，而不会抛出错误。

### 交叉熵、logits 和 softmax

分类器对每张图像输出称为 logits 的 `C` 个数值。应用 softmax 将它们转换为概率分布：

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

交叉熵衡量正确类别的负对数概率：

```
CE(z, y) = -log( softmax(z)_y )
        = -z_y + log( sum_j exp(z_j) )
```

右侧形式是数值稳定的（log-sum-exp）。PyTorch 的 `nn.CrossEntropyLoss` 将 softmax + NLL 融合为一个算子，直接接受原始 logits。自己先应用 softmax 几乎总是 bug——你计算的是 log(softmax(softmax(z)))，这是一个无意义的量。

### 为什么增强有效

CNN 具有平移的归纳偏置（来自权重共享），但对裁剪、翻转、颜色抖动或遮挡没有内置不变性。教会它这些不变性的唯一方式是展示具有这些特征的像素。训练期间的每个随机变换都是在说："这两张图像有相同的标签；学习忽略差异的特征。"

```
原始裁剪:  "dog facing left"
翻转:           "dog facing right"       <- 相同标签，不同像素
旋转(+15):    "dog, slight tilt"
颜色抖动:  "dog in warmer light"
RandomErasing:  "dog with patch missing"
```

规则：增强必须保留标签。对数字的 Cutout 和旋转可能将 "6" 变成 "9"；对于该数据集，你使用较小的旋转范围，并选择尊重数字特定不变性的增强。

### Mixup 和 Cutmix

普通增强变换像素但保持标签为 one-hot。**Mixup** 和 **cutmix** 通过插值打破这一点。

```
Mixup:
  lambda ~ Beta(a, a)
  x = lambda * x_i + (1 - lambda) * x_j
  y = lambda * y_i + (1 - lambda) * y_j

Cutmix:
  将 x_j 的随机矩形粘贴到 x_i 中
  y = y_i 和 y_j 的面积加权混合
```

为什么它有帮助：模型停止记忆尖锐的 one-hot 目标，学会在类别之间插值。训练 loss 上升，测试准确率上升。它是任何分类器最廉价的鲁棒性升级。

### 标签平滑

Mixup 的亲属。不是针对 `[0, 0, 1, 0, 0]` 训练，而是针对 `[eps/C, eps/C, 1-eps, eps/C, eps/C]` 训练，其中 `eps` 是一个很小的值如 0.1。防止模型产生任意尖锐的 logits，几乎不花费任何代价即可改善校准。自 PyTorch 1.10 以来内置于 `nn.CrossEntropyLoss(label_smoothing=0.1)`。

### 超越准确率的评估

聚合准确率掩盖了不平衡。一个始终预测多数类的 90-10 二分类器得分 90%。真正告诉你发生了什么的是这些工具：

- **每类准确率** — 每个类别一个数值；立即显现表现不佳的类别。
- **混淆矩阵** — C x C 网格，行 i 列 j = 真实类别 i 被预测为类别 j 的数量；对角线是正确的，非对角线是你的模型出问题的地方。
- **Top-1 / Top-5** — 正确类别是否在前 1 或前 5 个预测中；Top-5 在 ImageNet 上很重要，因为"诺维奇梗犬"vs"诺福克梗犬"这样的类别确实模糊。
- **校准 (ECE)** — 0.8 置信度的预测是否 80% 正确？现代网络系统地过度自信；用温度缩放或标签平滑修复。

```figure
receptive-field
```

## 构建

### 第1步：确定性合成数据集

CIFAR-10 存储在磁盘上。为使本课可重现且快速，我们构建一个看起来像 CIFAR 的合成数据集——32x32 RGB 图像，带有模型必须学习的类别特定结构。完全相同的流水线无需更改即可在真实 CIFAR-10 上使用。

```python
import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    Y = []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1)
            X.append(img.astype(np.float32))
            Y.append(c)
    X = np.stack(X)
    Y = np.array(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(self.Y[i])
```

每个类别都有自己的调色板和频率模式，加上高斯噪声迫使模型学习信号而非记忆像素。十个类别，每类一千张图像，已打乱。

### 第2步：归一化和增强

每个视觉流水线必备的两种变换。

```python
def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn
```

裁剪前使用反射填充而非零填充，因为黑色边框是一个模型会以非有用方式学习的信号。

### 第3步：Mixup

在训练步骤内混合两张图像和两个标签。作为批量变换实现，使其位于前向传播附近而非数据集内部。

```python
def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

`soft_cross_entropy` 是针对软标签分布的交叉熵。当目标是恰好 one-hot 时，它退化为通常的 one-hot 情况。

### 第4步：训练循环

完整配方：一遍数据、每个批次一次梯度、每个 epoch 一次 scheduler 步进。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x_m, y_soft = mixup_batch(x, y, num_classes)
            logits = model(x_m)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        # 当 mixup 开启时，训练准确率 vs 未混合标签 `y` 只是近似值
        # （模型看到的是软目标，而非 y）。将其视为粗略进度信号；依靠验证准确率获取真实性能。
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total, cm
```

编写训练循环时每次检查的五条不变式：

1. 训练前 `model.train()`，评估前 `model.eval()` — 切换 dropout 和 batchnorm 行为。
2. `.zero_grad()` 在 `.backward()` 之前。
3. 累积指标时使用 `.item()` 以免让计算图持续存活。
4. 评估时使用 `@torch.no_grad()` — 节省内存和时间，防止微妙事故。
5. 对原始 logits 进行 argmax，而非 softmax — 结果相同，少一个算子。

### 第5步：组装起来

使用上一课的 `TinyResNet`，训练几个 epoch，评估。

```python
from main import synthetic_cifar, ArrayDataset
from main import standardize, random_hflip, random_crop, compose
from main import mixup_batch, soft_cross_entropy
from main import train_one_epoch, evaluate
# TinyResNet 来自上一课 (03-cnns-lenet-to-resnet)。
# 调整导入路径到你存储上一课代码的位置。
from cnns_lenet_to_resnet import TinyResNet  # 示例占位符

X, Y = synthetic_cifar(num_per_class=500)
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
train_tf = compose(random_hflip(), random_crop(pad=4), standardize(mean, std))
eval_tf = standardize(mean, std)

train_ds = ArrayDataset(X_train, Y_train, transform=train_tf)
val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, 10, use_mixup=True)
    va_loss, va_acc, _ = evaluate(model, val_loader, device, 10)
    scheduler.step()
    print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
```

在合成数据集上，此流水线在五轮内达到近乎完美的验证准确率，这就是目标：流水线正确，模型可以学习可学的内容。将数据集替换为真实 CIFAR-10，相同循环无需更改即可训练到 ~90%。

### 第6步：读取混淆矩阵

仅靠准确率永远无法告诉你模型在哪里失败。混淆矩阵可以。

```python
def print_confusion(cm, labels=None):
    c = cm.shape[0]
    labels = labels or [str(i) for i in range(c)]
    print(f"{'':>6}" + "".join(f"{l:>5}" for l in labels))
    for i in range(c):
        row = cm[i].tolist()
        print(f"{labels[i]:>6}" + "".join(f"{v:>5}" for v in row))
    print()
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    for i in range(c):
        print(f"{labels[i]:>6}  prec {prec[i]:.3f}  rec {rec[i]:.3f}  f1 {f1[i]:.3f}")

_, _, cm = evaluate(model, val_loader, device, 10)
print_confusion(cm)
```

行是真实类别，列是预测。类别 3 和 5 之间非对角线上的聚类计数意味着模型混淆了这两个类别，为你提供了针对性数据收集或类别特定增强的起点。

## 使用

`torchvision` 将上述所有内容包装成语义化的组件。对于真实 CIFAR-10，完整流水线是四行代码加上训练循环。

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
```

注意两点：mean/std 是**数据集特定的**——在 CIFAR-10 训练集上计算，而非 ImageNet——以及反射填充是社区默认的裁剪策略。在这里复制粘贴 ImageNet 统计值会造成约 1% 的准确率泄漏，直到有人分析模型时才被察觉。

## 交付

本课产出：

- `outputs/prompt-classifier-pipeline-auditor.md` — 一个提示，用于审计训练脚本是否符合上述五条不变式并揭示第一个违规项。
- `outputs/skill-classification-diagnostics.md` — 一个技能，给定混淆矩阵和类别名称列表，总结每类故障并提出最具影响力的单条修复建议。

## 练习

1. **(简单)** 在合成数据集上用和不用 mixup 训练同一个模型五个 epoch。绘制两者的训练和验证 loss。解释为什么带 mixup 的训练 loss 更高，但验证准确率相似或更好。
2. **(中等)** 实现 Cutout——在每张训练图像中置零一个随机 8x8 方块——并与无增强、hflip+crop、hflip+crop+cutout、hflip+crop+mixup 进行消融实验。报告每个配置的验证准确率。
3. **(困难)** 构建 CIFAR-100 流水线（100 个类别，相同输入大小），并复现 ResNet-34 训练到与已发表准确率相差 1% 以内。加分项：扫描三个学习率和两个权重衰减，记录到本地 CSV，生成最终混淆矩阵-最高混淆表。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------------|----------------------------------------------------------|
| Logits | "原始输出" | 每张图片预 softmax 的 C 个数值向量；交叉熵期望这些，而非 softmax 后的值 |
| 交叉熵 | "损失函数" | 正确类别的负对数概率；在一个稳定算子内结合 log-softmax 和 NLL |
| DataLoader | "批处理器" | 包装数据集，提供打乱、分批和（可选的）多工作进程加载；被归咎于一半的训练 bug |
| 增强 | "随机变换" | 训练期间任何保留标签的像素级变换；教授 CNN 不具备的原生不变性 |
| Mixup / Cutmix | "混合两张图像" | 混合输入和标签，使分类器学习平滑插值而非硬边界 |
| 标签平滑 | "更软的 targets" | 用 (1-eps, eps/(C-1), ...) 替代 one-hot；改善校准并略微提升准确率 |
| Top-k 准确率 | "Top-5" | 正确类别在前 k 个最高概率预测中；用于具有真正模糊类别的数据集 |
| 混淆矩阵 | "错误所在之处" | C x C 表格，条目 (i, j) 计数真实类别 i 被预测为 j 的图像数；对角线是正确的，非对角线告诉你该修复什么 |

## 延伸阅读

- [CS231n: Training Neural Networks](https://cs231n.github.io/neural-networks-3/) — 仍是单页上最清晰的训练流水线导览
- [Bag of Tricks for Image Classification (He et al., 2019)](https://arxiv.org/abs/1812.01187) — 每个小技巧的组合，在 ImageNet 上为 ResNet 准确率增加 3-4%
- [mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412) — 原始 mixup 论文；三页理论加上令人信服的实验
- [Why temperature scaling matters (Guo et al., 2017)](https://arxiv.org/abs/1706.04599) — 证明现代网络校准错误的论文，并用一个标量参数修复它
```
