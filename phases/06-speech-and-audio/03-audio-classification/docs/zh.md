# 音频分类 — 从基于MFCC的k-NN到AST与BEATs

> 从“狗吠声对比警笛声”到“这是哪种语言”，都属于音频分类。特征是梅尔频谱。架构每十年更新一代。评估指标始终是AUC、F1和每类召回率。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段6 · 02（频谱图与梅尔）、阶段3 · 06（CNN）、阶段5 · 08（用于文本的CNN与RNN）
**时间：** 约75分钟

## 问题描述

你获得一段10秒的音频片段。你想知道：“这是什么？”城市声音（警笛、电钻、狗叫）、语音指令（是/否/停止）、语言识别（英语/西班牙语/阿拉伯语）、说话者情绪（愤怒/平静），或环境声音（室内/室外、嘈杂人声）。所有这些都是*音频分类*，在2026年，其基线架构已经成熟：对数梅尔频谱 → CNN或Transformer → softmax。

核心难点不在于网络，而在于数据。音频数据集存在严重的类别不平衡、强烈的领域偏移（干净vs嘈杂）以及标签噪声（谁来区分“城市嘈杂声”和“餐厅噪音”？）。80%的问题在于数据整理、增强和评估，而非用Transformer替换CNN。

## 概念解析

![音频分类阶梯：从基于MFCC的k-NN到AST再到BEATs](../assets/audio-classification.svg)

**基于MFCC的k-NN（1990年代基线）。** 将每段音频的MFCC展平，与标记样本库计算余弦相似度，返回前K个样本的多数投票结果。在干净的小型数据集（如Speech Commands, ESC-50）上表现意外地好。无需GPU即可运行。

**基于对数梅尔频谱的2D CNN（2015-2019年）。** 将`(T, n_mels)`对数梅尔频谱视为图像。应用ResNet-18或VGG风格的模型。在时间轴上进行全局平均池化。对类别进行softmax分类。在2026年的大多数Kaggle竞赛中仍是基线方法。

**音频频谱图Transformer, AST (2021-2024年)。** 将对数梅尔频谱划分为图块（例如16×16的块），添加位置嵌入，输入ViT。在AudioSet上的监督学习达到当时最佳水平（mAP 0.485）。

**BEATs与WavLM-base (2024-2026年)。** 在数百万小时的数据上进行自监督预训练。在你的任务上微调时，仅需原本所需监督数据的1-10%。到2026年，这是非语音音频任务的默认起点。BEATs-iter3在AudioSet上比AST高出1-2个mAP，同时计算量仅为四分之一。

**Whisper编码器作为冻结骨干网络 (2024年)。** 取用Whisper的编码器，去掉解码器，接上一个线性分类器。在语言识别和简单事件分类上达到接近最佳的性能，且无需任何音频增强。堪称“免费午餐”基线。

### 类别不平衡才是真正的挑战

ESC-50：50个类别，每类40个片段 — 平衡，简单。UrbanSound8K：10个类别，不平衡比例10:1。AudioSet：632个类别，存在100,000:1的长尾分布。有效应对技巧包括：
- 训练时采用平衡采样（评估时不使用）。
- 混合增强：线性插值两个音频片段（及其标签）作为数据增强。
- SpecAugment：遮蔽随机的时频区间。简单且关键。

### 评估指标

- 互斥多分类（如Speech Commands）：Top-1准确率、Top-5准确率。
- 多标签多分类（如AudioSet, UrbanSound风格）：平均精度均值。
- 严重不平衡情况：每类召回率 + 宏平均F1分数。

你需要知道的2026年基准数据：

| 基准数据集 | 基线模型 | 2026年最佳性能 | 来源 |
|------------|----------|----------------|------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs论文 (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR排行榜 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 结果 |

## 动手构建

### 第一步：特征提取

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 第二步：生成定长摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单而强大：对时间轴计算均值和方差，为13个系数的MFCC生成26维的固定嵌入。几乎瞬时完成。直到2017年，它仍在ESC-50上击败当时的最佳神经网络基线。

### 第三步：k-NN

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

### 第四步：升级到基于对数梅尔频谱的CNN

在PyTorch中：

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

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300万参数。在单块RTX 4090上，ESC-50数据集训练约需10分钟。准确率可达80%以上。

### 第五步：2026年默认选择 — 微调BEATs

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

对于BEATs，使用`microsoft/BEATs-base`通过`beats`库；transformers API的接口形状相同。

## 实际应用

2026年的技术栈：

| 场景 | 推荐起步方案 |
|------|-------------|
| 极小数据集 (<1000个片段) | 基于MFCC均值的k-NN（你的基线） + 音频增强 |
| 中等数据集 (1K–100K) | BEATs 或 AST 微调 |
| 大型数据集 (>100K) | 从头训练或微调Whisper编码器 |
| 实时、边缘部署 | 40个MFCC系数的CNN，量化为int8（关键词检测风格） |
| 多标签分类 (AudioSet) | BEATs-iter3 + BCE损失 + 混合增强 + SpecAugment |
| 语言识别 | MMS-LID，SpeechBrain VoxLingua107基线 |

决策规则：**从冻结骨干网络开始，而非从头构建新模型**。微调一个BEATs分类头，几小时内就能达到95%的最佳性能，而非耗时数周。

## 部署上线

保存为`outputs/skill-classifier-designer.md`。为特定的音频分类任务选定架构、增强策略、类别平衡方法和评估指标。

## 练习题

1. **简单。** 运行`code/main.py`。它在一个4类合成数据集（不同音高的纯音）上训练基于MFCC的k-NN基线。报告混淆矩阵。
2. **中等。** 将`summarize`替换为[均值, 方差, 偏度, 峰度]。四矩池化在同一合成数据集上是否优于均值+方差？
3. **困难。** 使用`torchaudio`，在ESC-50的fold 1上训练一个2D CNN。报告5折交叉验证的准确率。添加SpecAugment（时间遮蔽=20，频率遮蔽=10）并报告性能变化。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| AudioSet | 音频领域的ImageNet | 谷歌的200万片段、632类弱标签YouTube数据集。 |
| ESC-50 | 小型分类基准 | 50类环境声音，每类40个片段。 |
| AST | 音频频谱图Transformer | 基于对数梅尔频谱图块的ViT；2021年达到最佳性能。 |
| BEATs | 自监督音频模型 | 微软模型，其迭代版本3在2026年引领AudioSet。 |
| Mixup | 配对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于遮蔽的增强 | 将频谱图上随机的时频区间置零。 |
| mAP | 多标签任务主要指标 | 跨类别和阈值的平均精度均值。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021至2024年的代表性架构。
- [Chen et al. (2022, 修订版 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024年后的默认选择。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主流的音频增强方法。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 历久弥新的50类基准数据集。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632类YouTube声音分类法；仍是黄金标准。