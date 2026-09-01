# 从零实现卷积

> 卷积是一个在图像上滑动的小型全连接层，在所有位置共享相同的权重。

**类型:** 构建练习
**语言:** Python
**前置知识:** 第 3 阶段（深度学习核心），第 4 阶段第 01 课（图像基础）
**时间:** 约 75 分钟

## 学习目标

- 仅使用 NumPy 从零实现 2D 卷积，包括嵌套循环版本和向量化 `im2col` 版本
- 计算任意输入尺寸、核大小、填充和步长组合下的输出空间尺寸，并解释 `(H - K + 2P) / S + 1` 公式的由来
- 手工设计核（边缘检测、模糊、锐化、Sobel），并解释每种核为何产生对应的激活模式
- 将卷积堆叠成特征提取器，并说明堆叠深度与感受野大小之间的关系

## 问题陈述

对一个 224x224 RGB 图像的全连接层，每个神经元需要 224 × 224 × 3 = 150,528 个输入权重。一个包含 1,000 个神经元的隐藏层就已经有 1.5 亿参数——此时你还没学到任何有用的东西。更糟的是，该层没有概念：左上角的狗和右下角的狗是同一个模式。它将每个像素位置视为独立的，这恰好是错误的：将猫平移三个像素不应迫使网络重新学习这个概念。

图像模型需要的两个属性是**平移等变**（输入平移时输出也平移）和**参数共享**（同一个特征探测器在所有位置运行）。全连接层两者都不具备。而卷积免费提供了这两个特性。

卷积并非为深度学习而发明。它与 JPEG 压缩、Photoshop 的高斯模糊、工业视觉中的边缘检测以及所有已发布的音频滤波器使用的是同一操作。CNN 在 2012 年至 2020 年主导 ImageNet 的原因在于，对于邻近值相关且同一模式可出现在任意位置的**数据**而言，卷积是正确的先验。

## 概念

### 一个核，滑动执行

2D 卷积取一个小权重矩阵（称为**核**或**滤波器**），将其在输入上滑动，并在每个位置计算元素级乘积之和。该和成为输出中的一个像素。

```mermaid
flowchart LR
    subgraph IN["输入 (H x W)"]
        direction LR
        I1["5 x 5 图像"]
    end
    subgraph K["核 (3 x 3)"]
        K1["学习到的<br/>权重"]
    end
    subgraph OUT["输出 (H-2 x W-2)"]
        O1["3 x 3 特征图"]
    end
    I1 --> |"滑动核<br/>在每个位置计算点积"| O1
    K1 --> O1

    style IN fill:#dbeafe,stroke:#2563eb
    style K fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

在 5x5 输入上的具体 3x3 示例（无填充，步长为 1）：

```
输入 X (5 x 5):                核 W (3 x 3):

  1  2  0  1  2                   1  0 -1
  0  1  3  1  0                   2  0 -2
  2  1  0  2  1                   1  0 -1
  1  0  2  1  3
  2  1  1  0  1

核在所有合法的 3 x 3 窗口上滑动。输出 Y 为 3 x 3:

 Y[0,0] = sum( W * X[0:3, 0:3] )
 Y[0,1] = sum( W * X[0:3, 1:4] )
 Y[0,2] = sum( W * X[0:3, 2:5] )
 Y[1,0] = sum( W * X[1:4, 0:3] )
 ... 以此类推
```

这一个公式——**共享权重、局部性、滑动窗口**——就是全部思想。其余只是例行计算。

### 输出尺寸公式

给定输入空间尺寸 `H`、核大小 `K`、填充 `P`、步长 `S`：

```
H_out = floor( (H - K + 2P) / S ) + 1
```

记住这个公式。你在设计每个架构时都会计算几十次。

| 场景 | H | K | P | S | H_out |
|----------|---|---|---|---|-------|
| 有效卷积，无填充 | 32 | 3 | 0 | 1 | 30 |
| Same 卷积（保持尺寸） | 32 | 3 | 1 | 1 | 32 |
| 下采样 2 倍 | 32 | 3 | 1 | 2 | 16 |
| 2x2 池化 | 32 | 2 | 0 | 2 | 16 |
| 大感受野 | 32 | 7 | 3 | 2 | 16 |

"Same 填充"意味着选择 P 使得当 S == 1 时 H_out == H。对于奇数 K，P = (K - 1) / 2。这就是为什么 3x3 核占据主导地位——它们是最小的仍具有中心的奇数核。

### 填充

若无填充，每次卷积都会缩小特征图。堆叠 20 个这样的卷积后，你的 224x224 图像会变成 184x184，这在边界上浪费算力并增加需要匹配形状的残差连接的复杂度。

```
在 5 x 5 输入上的零填充 (P = 1)：

  0  0  0  0  0  0  0
  0  1  2  0  1  2  0
  0  0  1  3  1  0  0
  0  2  1  0  2  1  0       现在核可以以像素
  0  1  0  2  1  3  0       (0, 0) 为中心，仍有三行和三列的值相乘。
  0  2  1  1  0  1  0
  0  0  0  0  0  0  0
```

实践中遇到的模式：`zero`（最常见）、`reflect`（镜像边缘，避免生成模型中的硬边界）、`replicate`（复制边缘）、`circular`（环绕，用于环形问题）。

### 步长

步长是滑动的步幅。`stride=1` 是默认值。`stride=2` 将空间尺寸减半，是 CNN 内部下采样的经典方式，无需单独的池化层——每个现代架构（ResNet、ConvNeXt、MobileNet）都在某处用步长卷积替代最大池化。

```
在 5 x 5 输入上步长为 1，3 x 3 核：

  起点: (0,0) (0,1) (0,2)        -> 输出第 0 行
          (1,0) (1,1) (1,2)        -> 输出第 1 行
          (2,0) (2,1) (2,2)        -> 输出第 2 行

  输出: 3 x 3

相同输入上步长为 2：

  起点: (0,0) (0,2)              -> 输出第 0 行
          (2,0) (2,2)              -> 输出第 1 行

  输出: 2 x 2
```

### 多个输入通道

真实图像有 3 个通道。在 RGB 输入上的 3x3 卷积实际上是一个 3x3x3 的张量：每个输入通道一个 3x3 切片。在每个空间位置，你对所有三个切片执行乘加并加上偏置。

```
输入:   (C_in,  H,  W)        3 x 5 x 5
核:     (C_in,  K,  K)        3 x 3 x 3（一个核）
输出:   (1,     H', W')       2D 特征图

对于一个产生 C_out 输出通道的层，堆叠 C_out 个核：

权重:   (C_out, C_in, K, K)   例如 64 x 3 x 3 x 3
输出:   (C_out, H', W')       64 x 3 x 3

参数量: C_out * C_in * K * K + C_out   （+ C_out 是偏置）
```

最后一行是你设计模型时需要计算的。在 3 通道输入上的 64 通道 3x3 卷积有 `64 * 3 * 3 * 3 + 64 = 1,792` 个参数。很便宜。

### im2col 技巧

嵌套循环易读但慢。GPU 需要大型矩阵乘法。技巧是：将输入的每个感受野窗口展平成一个大矩阵的一列，将核展平为一行，整个卷积就变成一个矩阵乘法。

```mermaid
flowchart LR
    X["输入<br/>(C_in, H, W)"] --> IM2COL["im2col<br/>(提取补丁)"]
    IM2COL --> COLS["列矩阵<br/>(C_in * K * K, H_out * W_out)"]
    W["权重<br/>(C_out, C_in, K, K)"] --> FLAT["展平<br/>(C_out, C_in * K * K)"]
    FLAT --> MM["矩阵乘法"]
    COLS --> MM
    MM --> OUT["输出<br/>(C_out, H_out * W_out)<br/>重塑为 (C_out, H_out, W_out)"]

    style X fill:#dbeafe,stroke:#2563eb
    style W fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

每个生产级卷积实现都是此方法的变体加上缓存分块技巧（direct conv、Winograd、大核的 FFT 卷积）。理解 im2col 就理解了核心。

### 感受野

单个 3x3 卷积查看 9 个输入像素。堆叠两个 3x3 卷积后，第二层的神经元查看 5x5 输入像素。三个 3x3 卷积给出 7x7。一般地：

```
L 个堆叠的 K x K 卷积后的感受野 (步长 1) = 1 + L * (K - 1)

带步长的情况：感受野沿每层随步长相乘增长。
```

"3x3 一路到底"可行（VGG、ResNet、ConvNeXt）的根本原因在于：两个 3x3 卷积看到的输入区域与一个 5x5 卷积相同，但参数更少且中间多了一个非线性层。

```figure
convolution-kernel
```

## 构建

### 第 1 步：填充数组

从最小的原语开始：一个在 H x W 数组周围填充零的函数。

```python
import numpy as np

def pad2d(x, p):
    if p == 0:
        return x
    h, w = x.shape[-2:]
    out = np.zeros(x.shape[:-2] + (h + 2 * p, w + 2 * p), dtype=x.dtype)
    out[..., p:p + h, p:p + w] = x
    return out

x = np.arange(9).reshape(3, 3)
print(x)
print()
print(pad2d(x, 1))
```

尾轴技巧 `x.shape[:-2]` 意味着同一个函数对 `(H, W)`、`(C, H, W)` 或 `(N, C, H, W)` 均适用，无需修改。

### 第 2 步：嵌套循环的 2D 卷积

参考实现——慢，但清晰。这是 `torch.nn.functional.conv2d` 原理上做的。

```python
def conv2d_naive(x, w, b=None, stride=1, padding=0):
    c_in, h, w_in = x.shape
    c_out, c_in_w, kh, kw = w.shape
    assert c_in == c_in_w

    x_pad = pad2d(x, padding)
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w_in + 2 * padding - kw) // stride + 1

    out = np.zeros((c_out, h_out, w_out), dtype=np.float32)
    for oc in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                hs = i * stride
                ws = j * stride
                patch = x_pad[:, hs:hs + kh, ws:ws + kw]
                out[oc, i, j] = np.sum(patch * w[oc])
        if b is not None:
            out[oc] += b[oc]
    return out
```

四个嵌套循环（输出通道、行、列，以及隐式的 C_in、kh、kw 上的求和）。这是你验证每个更快实现的基准。

### 第 3 步：用手设计的核验证

构建一个垂直 Sobel 核，应用到合成阶梯图像上，观察垂直边缘是否点亮。

```python
def synthetic_step_image():
    img = np.zeros((1, 16, 16), dtype=np.float32)
    img[:, :, 8:] = 1.0
    return img

sobel_x = np.array([
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]]
], dtype=np.float32)[None]

x = synthetic_step_image()
y = conv2d_naive(x, sobel_x, padding=1)
print(y[0].round(1))
```

期望在第 7 列出现大的正值（从左到右亮度增加），其余位置为零。这个单次打印就是你检验数学正确性的依据。

### 第 4 步：im2col

将输入中每个核大小的窗口转换为矩阵的一列。对于 `C_in=3, K=3`，每列有 27 个数。

```python
def im2col(x, kh, kw, stride=1, padding=0):
    c_in, h, w = x.shape
    x_pad = pad2d(x, padding)
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1

    cols = np.zeros((c_in * kh * kw, h_out * w_out), dtype=x.dtype)
    col = 0
    for i in range(h_out):
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            patch = x_pad[:, hs:hs + kh, ws:ws + kw]
            cols[:, col] = patch.reshape(-1)
            col += 1
    return cols, h_out, w_out
```

它仍然是 Python 循环，但现在繁重的计算将由单个向量化矩阵乘法完成。

### 第 5 步：通过 im2col + 矩阵乘法实现快速卷积

用一次矩阵乘法替代四重循环。

```python
def conv2d_im2col(x, w, b=None, stride=1, padding=0):
    c_out, c_in, kh, kw = w.shape
    cols, h_out, w_out = im2col(x, kh, kw, stride, padding)
    w_flat = w.reshape(c_out, -1)
    out = w_flat @ cols
    if b is not None:
        out += b[:, None]
    return out.reshape(c_out, h_out, w_out)
```

正确性检查：运行两种实现并比较。

```python
rng = np.random.default_rng(0)
x = rng.normal(0, 1, (3, 16, 16)).astype(np.float32)
w = rng.normal(0, 1, (8, 3, 3, 3)).astype(np.float32)
b = rng.normal(0, 1, (8,)).astype(np.float32)

y_naive = conv2d_naive(x, w, b, padding=1)
y_im2col = conv2d_im2col(x, w, b, padding=1)

print(f"最大绝对差: {np.max(np.abs(y_naive - y_im2col)):.2e}")
```

`max abs diff` 应约为 `1e-5`——差异来自浮点累加顺序，而非 bug。

### 第 6 步：一手设计的核库

五个滤波器，展示单个卷积层在任何训练之前能表达什么。

```python
KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    "blur_3x3": np.ones((3, 3), dtype=np.float32) / 9.0,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
}

def apply_kernel(img2d, kernel):
    x = img2d[None].astype(np.float32)
    w = kernel[None, None]
    return conv2d_im2col(x, w, padding=1)[0]
```

应用于任何灰度图像时，blur 柔和化，sharpen 锐化边缘，Sobel-x 点亮垂直边缘，Sobel-y 点亮水平边缘。这些正是 AlexNet 和 VGG 中*第一个*训练的卷积层最终学到的模式——因为一个好的图像模型无论后续任务是什么，都需要边缘和斑点检测器。

## 使用

PyTorch 的 `nn.Conv2d` 用 autograd、CUDA 内核和 cuDNN 优化包装了相同的操作。形状语义完全相同。

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
print(conv)
print(f"权重形状: {tuple(conv.weight.shape)}   # (C_out, C_in, K, K)")
print(f"偏置形状:   {tuple(conv.bias.shape)}")
print(f"参数量:  {sum(p.numel() for p in conv.parameters())}")

x = torch.randn(8, 3, 224, 224)
y = conv(x)
print(f"\n输入形状: {tuple(x.shape)}")
print(f"输出形状: {tuple(y.shape)}")
```

将 `padding=1` 改为 `padding=0`，输出降至 222x222。将 `stride=1` 改为 `stride=2`，输出降至 112x112。与你刚才记住的公式相同。

## 交付

本课产出：

- `outputs/prompt-cnn-architect.md` —— 一个 prompt，给定输入尺寸、参数预算和目标感受野，在每个步骤设计具有正确 K/S/P 的 `Conv2d` 层堆叠。
- `outputs/skill-conv-shape-calculator.md` —— 一个技能，逐层遍历网络规格，返回每个块的输出形状、感受野和参数量。

## 练习

1. **(简单)** 给定 128x128 灰度输入和一个 `[Conv3x3(s=1,p=1), Conv3x3(s=2,p=1), Conv3x3(s=1,p=1), Conv3x3(s=2,p=1)]` 堆叠，手工计算每层的输出空间尺寸和感受野。用 PyTorch `nn.Sequential` 的虚拟卷积验证。
2. **(中等)** 扩展 `conv2d_naive` 和 `conv2d_im2col` 以接受 `groups` 参数。证明 `groups=C_in=C_out` 复现了深度可分离卷积，且其参数量为 `C * K * K` 而非 `C * C * K * K`。
3. **(困难)** 手工实现 `conv2d_im2col` 的反向传播：给定输出的梯度，计算 `x` 和 `w` 的梯度。用相同输入和权重与 `torch.autograd.grad` 对比验证。技巧：im2col 的梯度是 `col2im`，且需要累加重叠窗口。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 卷积 | "滑动滤波器" | 在每个空间位置以共享权重应用的可学习点积；数学上是互相关，但大家都叫它卷积 |
| 核 / 滤波器 | "特征探测器" | 形状为 (C_in, K, K) 的小权重张量，其与输入窗口的点积产生一个输出像素 |
| 步长 | "跳多远" | 连续核放置之间的步幅；步长 2 将每个空间维度减半 |
| 填充 | "边缘补零" | 在输入周围添加额外值，使核能以边界像素为中心；`same` 填充保持输出尺寸等于输入尺寸 |
| 感受野 | "神经元能看到多少" | 给定输出激活所依赖的原始输入区域，随深度和步长增长 |
| im2col | "GEMM 技巧" | 将每个感受野窗口重排为列，使卷积变成一次大型矩阵乘法——每个快速卷积内核的核心 |
| 深度卷积 | "每通道一个核" | 一个 `groups == C_in` 的卷积，每个输出通道仅从其匹配的输入通道计算；MobileNet 和 ConvNeXt 的骨干 |
| 平移等变 | "平移输入，平移输出" | 将输入平移 k 像素则输出平移 k 像素的属性；由共享权重免费获得 |

## 延伸阅读

- [A guide to convolution arithmetic for deep learning (Dumoulin & Visin, 2016)](https://arxiv.org/abs/1603.07285) —— 关于填充/步长/空洞的权威图解，每门课程都在默默参考
- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/) —— 经典讲义，包括原始的 im2col 解释
- [The Annotated ConvNet (fast.ai)](https://nbviewer.org/github/fastai/fastbook/blob/master/13_convolutions.ipynb) —— 一个笔记本，从手动卷积引导至训练好的数字分类器
- [Receptive Field Arithmetic for CNNs (Dang Ha The Hien)](https://distill.pub/2019/computing-receptive-fields/) —— 感受野计算的论文级交互式解释
