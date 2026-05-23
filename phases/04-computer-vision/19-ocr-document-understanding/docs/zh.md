# OCR 与文档理解

> OCR 是一个三阶段流程——检测文本框、识别字符、然后重新排布。每个现代 OCR 系统都会重排这些阶段或将其合并。

**类型：** 学习 + 使用
**语言：** Python
**先决条件：** 第 4 阶段第 06 课（检测），第 7 阶段第 02 课（自注意力）
**时间：** ~45 分钟

## 学习目标

- 追溯经典 OCR 流程（检测 -> 识别 -> 排版）和现代端到端替代方案（Donut、Qwen-VL-OCR）
- 为序列到序列 OCR 训练实现 CTC（连接时序分类）损失函数
- 使用 PaddleOCR 或 EasyOCR 进行生产环境文档解析而无需训练
- 区分 OCR、版面解析和文档理解——并为每个任务选择合适的工具

## 问题所在

充满文本的图像无处不在：收据、发票、身份证件、扫描书籍、表格、白板、标志、截图。从中提取结构化数据——不仅仅是字符，而是“这是总金额”——这是最高价值的应用视觉问题之一。

该领域分为三个技能层级：

1. **OCR 本身**：将像素转换为文本。
2. **版面解析**：将 OCR 输出分组到不同区域（标题、正文、表格、页眉）。
3. **文档理解**：从版面中提取结构化字段（`invoice_total = $42.50`）。

每个层级都有经典和现代的方法，而“我想要从图像中提取文本”和“我需要从这张收据中获取总金额”之间的差距，比大多数团队意识到的要大。

## 核心概念

### 经典流程

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测** 生成每行或每个词的四边形区域。
- **识别** 将每个区域裁剪为固定高度，运行 CNN + BiLSTM + CTC 以生成字符序列。
- **排版** 重建阅读顺序（拉丁文从上到下、从左到右；阿拉伯文、日文则不同）。

### CTC 概述

OCR 识别从固定长度的特征图生成可变长度序列。CTC（Graves 等人，2006）允许你在没有字符级对齐的情况下进行训练。模型在每个时间步输出一个（词汇表 + 空白）上的概率分布；CTC 损失函数对所有在合并重复项并移除空白后能还原为目标文本的对齐方式进行边际化计算。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC 是 2015 年 CRNN 能够工作，并且在 2026 年仍训练大多数生产 OCR 模型的原因。

### 现代端到端模型

- **Donut**（Kim 等人，2022）——一个 ViT 编码器 + 一个文本解码器；读取图像并直接输出 JSON。无需文本检测器，无需版面模块。
- **TrOCR**——用于行级 OCR 的 ViT + Transformer 解码器。
- **Qwen-VL-OCR / InternVL**——为 OCR 任务微调的完整视觉语言模型；在 2026 年处理复杂文档的准确率最高。
- **PaddleOCR**——成熟的生产级包装中的经典 DB + CRNN 流程；仍然是开源主力。

端到端模型需要更多数据和计算，但避免了多阶段流程的误差累积。

### 版面解析

对于结构化文档，运行一个版面检测器（LayoutLMv3、DocLayNet）来标记每个区域：标题、段落、图表、表格、脚注。然后阅读顺序变为“按版面顺序遍历各区域并拼接”。

对于表单，使用**键值对提取**模型（针对视觉丰富文档使用 Donut，针对普通扫描件使用 LayoutLMv3）。它们接收图像 + 检测到的文本 + 位置信息，并预测结构化的键值对。

### 评估指标

- **字符错误率（CER）**——Levenshtein 距离 / 参考长度。越低越好。生产目标：在清晰扫描件上 < 2%。
- **单词错误率（WER）**——在单词级别上相同。
- **结构化字段的 F1 分数**——用于键值对任务；衡量 `key: value` 是否正确出现。
- **JSON 上的编辑距离**——用于端到端文档解析；Donut 论文引入了归一化树编辑距离。

## 动手构建

### 步骤 1：CTC 损失 + 贪心解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

``F.ctc_loss`` 在可用时使用高效的 CuDNN 实现。贪心解码器比波束搜索更简单，并且在 CER 上通常与之相差不到 1%。

### 步骤 2：微型 CRNN 识别器

用于行级 OCR 的最简 CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高度的输入（CNN 最大池化将高度降为 1）。宽度是 CTC 的时间维度。

### 步骤 3：合成 OCR 数据

生成黑底白字的数字字符串，用于端到端冒烟测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实的 OCR 数据集会增加字体、噪声、旋转、模糊和颜色。上述流程完全一致。

### 步骤 4：训练草图

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这个简单的合成数据上，损失函数应在 200 步内从 ~3 下降到 ~0.2。

## 投入使用

三条生产路径：

- **PaddleOCR**——成熟、快速、多语言。一行代码用法：``paddleocr.PaddleOCR(lang="en").ocr(image_path)``。
- **EasyOCR**——原生 Python、多语言、基于 PyTorch。
- **Tesseract**——经典；当模型效果不佳时，对于旧扫描文档仍然有用。

对于端到端文档解析，使用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于具有可重复结构的收据、发票和表单，微调 Donut。对于任意文档或需要推理的 OCR，像 Qwen-VL-OCR 这样的 VLM 是目前的默认选择。

## 交付成果

本课产出：

- ``outputs/prompt-ocr-stack-picker.md``——一个提示，根据文档类型、语言和结构选择 Tesseract / PaddleOCR / Donut / VLM-OCR。
- ``outputs/skill-ctc-decoder.md``——一项技能，能够从头编写贪心和波束搜索 CTC 解码器，包括长度归一化。

## 练习

1. **（简单）** 在 5 位随机数字字符串上训练 TinyCRNN 500 步。报告在留出集上的 CER。
2. **（中等）** 用波束搜索（beam_width=5）替换贪心解码。报告 CER 变化量。波束搜索在哪些输入上表现更好？
3. **（困难）** 在一组 20 张收据上使用 PaddleOCR，提取行项目，并根据人工标注的基准真值计算 {item_name, price} 对的 F1 分数。

## 关键术语

| 术语 | 人们怎么说 | 其实际含义 |
|------|----------------|----------------------|
| OCR | "从像素提取文本" | 将图像区域转换为字符序列 |
| CTC | "无对齐损失" | 一种损失函数，无需每个时间步的标签即可训练序列模型；对所有对齐方式进行边际化计算 |
| CRNN | "经典 OCR 模型" | 卷积特征提取器 + BiLSTM + CTC；2015 年的基线模型，至今仍在生产中使用 |
| Donut | "端到端 OCR" | ViT 编码器 + 文本解码器；直接从图像生成 JSON |
| 版面解析 | "查找区域" | 在文档中检测并标记标题/表格/图表/段落等区域 |
| 阅读顺序 | "文本序列" | 将识别的区域按顺序排列成句子；对于拉丁文简单，对于混合版面则不简单 |
| CER / WER | "错误率" | 在字符或单词粒度上计算的 Levenshtein 距离 / 参考长度 |
| VLM-OCR | "能读的 LLM" | 经过训练或提示用于 OCR 任务的视觉语言模型；目前处理复杂文档的最先进模型 |

## 延伸阅读

- [CRNN (Shi 等人, 2015)](https://arxiv.org/abs/1507.05717) ——原始的 CNN+RNN+CTC 架构
- [CTC (Graves 等人, 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) ——原始 CTC 论文；包含密集的算法思想
- [Donut (Kim 等人, 2022)](https://arxiv.org/abs/2111.15664) ——无需 OCR 的文档理解 Transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) ——开源生产级 OCR 技术栈