# 音频分类 —— 从 MFCC 上的 k-NN 到 AST 与 BEATs

> 从“狗叫与警笛”到“这是什么语言”，都属于音频分类。特征使用的是 mel 频谱，架构每十年迭代一次，评估指标则始终围绕 AUC、F1 和各类别召回率。

**类型：** 实践
**语言：** Python
**前置知识：** Phase 6 · 02（频谱图与 Mel），Phase 3 · 06（卷积神经网络），Phase 5 · 08（用于文本的 CNN 与 RNN）
**预计时间：** 约 75 分钟

## 问题描述

你拿到一段 10 秒的音频片段，想知道：“这是什么？” 城市声音（警笛、电钻、狗叫）、语音指令（是/否/停止）、语言识别（英语/西班牙语/阿拉伯语）、说话人情绪（愤怒/中性）还是环境音（室内/室外、嘈杂人声）。所有这些都属于*音频分类*，而在 2026 年，基线架构已经非常成熟：log-mel → CNN 或 Transformer → softmax。

核心难点不在于网络结构，而在于数据。音频数据集存在严重的类别不平衡、强烈的领域偏移（干净 vs 噪声）以及标签噪声（谁来决定“城市嘈杂声”和“餐厅噪音”的区别？）。问题的 80% 在于数据筛选、增强与评估，而不是把 CNN 换成 Transformer。

## 核心概念

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**基于 MFCC 的 k-NN（1990 年代的基线）。** 将每段音频的 MFCC 展平，计算与标签库的余弦相似度，返回 Top K 的多数投票结果。在干净的小数据集（Speech Commands、ESC-50）上表现出乎意料地好。无需 GPU 即可运行。

**基于 log-mel 的 2D CNN（2015-2019）。** 将 `(T, n_mels)` 的 log-mel 视为图像。使用 ResNet-18 或 VGG 风格架构。在时间轴上做全局均值池化。对类别进行 softmax。在 2026 年的大多数 Kaggle 比赛中仍是基线方案。

**音频谱图 Transformer，AST（2021-2024）。** 将 log-mel 分块（例如 16×16 的 patch），添加位置编码，输入 ViT。在 AudioSet 的监督学习中达到 SOTA（mAP 0.485）。

**BEATs 与 WavLM-base（2024-2026）。** 在数百万小时的音频上进行自监督预训练。仅需 1-10% 原本需要的有监督数据，即可在你的任务上微调。在 2026 年，这是非语音音频任务的默认起点。BEATs-iter3 在 AudioSet 上比 AST 高出 1-2 的 mAP，同时计算量仅为 1/4。

**将 Whisper encoder 作为冻结骨干网络（2024）。** 取 Whisper 的编码器，去掉解码器，挂载线性分类器。在零音频增强的情况下，于语言 ID 和简单事件分类上接近 SOTA。堪称“免费午餐”基线。

### 类别不平衡才是真正的挑战

ESC-50：50 个类别，每类 40 段 —— 均衡，简单。UrbanSound8K：10 个类别，不平衡比例达 10:1。AudioSet：632 个类别，长尾分布高达 100,000:1。有效的技巧：

- 训练时采用平衡采样（而非评估时）。
- Mixup：将两段音频（及其标签）线性插值作为增强。
- SpecAugment：掩码随机时间与频率带。简单；至关重要。

### 评估指标

- 多分类互斥（Speech Commands）：top-1 准确率、top-5 准确率。
- 多分类多标签（AudioSet、UrbanSound 风格）：平均精度均值（mAP）。
- 高度不平衡：各类别召回率 + 宏平均 F1。

2026 年你应该知道的基准数据：

| 基准测试 | 基线 | 2026 SOTA | 来源 |
|----------|------|-----------|------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

```figure
mfcc-pipeline
```

## 动手实现

### 步骤 1：特征提取

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 步骤 2：固定长度汇总

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但有效：对时间维度取均值与方差，可为 13 系数的 MFCC 生成 26 维固定嵌入。运行瞬间完成。直到 2017 年，该方法仍在 ESC-50 上击败了当时的神经网络基线。

### 步骤 3：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 步骤 4：升级为基于 log-mel 的 CNN

使用 PyTorch：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (批次, 1, 时间步, mel数)
        return self.head(self.body(x).flatten(1))
```

约 300 万参数。在单张 RTX 4090 上训练 ESC-50 仅需约 10 分钟。准确率可达 80% 以上。

### 步骤 5：2026 默认方案 —— 微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

对于 BEATs，通过 `beats` 库使用 `microsoft/BEATs-base`；transformers API 的调用方式完全相同。

## 应用场景

2026 年技术栈推荐：

| 场景 | 首选方案 |
|------|----------|
| 极小数据集 (<1000 段) | MFCC 均值上的 k-NN（你的基线）+ 音频增强 |
| 中小数据集 (1K–100K) | BEATs 或 AST 微调 |
| 大数据集 (>100K) | 从头训练或微调 Whisper-encoder |
| 实时/边缘设备 | 40-MFCC CNN，量化为 int8（类 KWS 风格） |
| 多标签（AudioSet） | BEATs-iter3 + BCE 损失 + mixup + SpecAugment |
| 语言识别 | MMS-LID、SpeechBrain VoxLingua107 基线 |

决策原则：**先从冻结骨干网络开始，而非从零训练模型**。微调 BEATs 的分类头只需数小时即可获得 SOTA 95% 的性能，而非数周。

## 交付成果

保存为 `outputs/skill-classifier-designer.md`。针对给定的音频分类任务，选定架构、数据增强策略、类别平衡方案与评估指标。

## 练习

1. **简单。** 运行 `code/main.py`。该脚本在 4 类合成数据集（不同音高的纯音）上训练 k-NN MFCC 基线。报告混淆矩阵。
2. **中等。** 将 `summarize` 替换为 [均值、方差、偏度、峰度]。在同一个合成数据集上，4 阶矩池化能否超越均值+方差？
3. **困难。** 使用 `torchaudio`，在 ESC-50 的第 1 个折叠上训练 2D CNN。报告 5 折交叉验证准确率。添加 SpecAugment（时间掩码 = 20，频率掩码 = 10），并报告提升幅度。

## 核心术语

| 术语 | 通常的说法 | 实际含义 |
|------|------------|----------|
| AudioSet | 音频界的 ImageNet | Google 的 200 万段、632 类弱标注 YouTube 数据集。 |
| ESC-50 | 小型分类基准 | 50 类 × 40 段环境声音数据集。 |
| AST | Audio Spectrogram Transformer | 基于 log-mel 分块的 ViT；2021 年 SOTA。 |
| BEATs | 自监督音频模型 | Microsoft 模型，至 2026 年 iter3 领跑 AudioSet。 |
| Mixup | 成对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于掩码的增强 | 将谱图的随机时间与频率带置零。 |
| mAP | 主要多标签指标 | 跨类别与阈值的平均精度均值。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) —— 2021–2024 年的标准架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) —— 2024 年以后的默认选择。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) —— 占主导地位的音频数据增强方法。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) —— 经久不衰的 50 类基准数据集。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) —— 632 类的 YouTube 音频分类体系；仍是黄金标准。
