# 说话人识别与验证

> 自动语音识别问“他们说了什么？”，而说话人识别问“是谁说的？”。数学原理看起来相同——嵌入加上余弦相似度——但每个实际生产决策都取决于单一的等错误率（EER）数值。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段6 · 02（频谱图与梅尔系数），阶段5 · 22（嵌入模型）
**时间：** 约45分钟

## 问题描述

用户说一个口令。你想知道：这是他们声称的那个人吗（*验证*，1:1），还是你注册库中的第一个人（*识别*，1:N）？或者都不是——这是一个未知说话人（*开集*）？

2018年以前：GMM-UBM + i-vectors。EER尚可，但对信道变化（电话 vs 笔记本电脑）和情绪敏感。2018–2022年：x-vectors（使用角度间隔训练的TDNN骨干网络）。2022年后：ECAPA-TDNN和WavLM-large嵌入。到2026年，该领域由三个模型和一个度量主导。

这个度量就是**EER**——等错误率。设定你的决策阈值，使得误接受率 = 误拒绝率。这个交点就是EER。在每篇论文、每个排行榜、每次采购要求中都会用到。

## 核心概念

![注册与验证流程，包含嵌入、余弦相似度与EER](../assets/speaker-verification.svg)

**流程说明。** 注册：录制目标说话人5–30秒的音频；计算一个固定维度的嵌入向量（ECAPA-TDNN为192维，WavLM-large为256维）。验证：获取测试语音的嵌入向量；计算余弦相似度；与阈值比较。

**ECAPA-TDNN（2020年，至2026年仍占主导）。** 强调通道注意力、传播与聚合的时间延迟神经网络。由包含压缩激励机制的1D卷积块、多头注意力池化层组成，后接一个线性层映射到192维。在VoxCeleb 1+2数据集（2,700名说话人，110万条语音）上使用加性角度间隔损失（AAM-softmax）训练。

**WavLM-SV（2022年后）。** 使用AAM损失微调预训练的WavLM-large自监督学习骨干网络。质量更高但速度较慢——模型大小300+ MB vs 15 MB。

**x-vector（基线）。** TDNN + 统计池化。经典方法；在CPU / 边缘设备上仍然有用。

**AAM-softmax。** 在角度空间中添加间隔`m`的标准softmax：`cos(θ + m)`用于正确类别。强制类间角度分离。典型值`m=0.2`，缩放因子`s=30`。

### 评分方法

- **余弦相似度**：在注册嵌入和测试嵌入之间计算。基于阈值的决策。
- **PLDA（概率线性判别分析）。** 将嵌入投影到一个潜在空间，使得同一说话人与不同说话人的似然比有闭合形式解。在余弦相似度基础上使用，可将EER降低10-20%。2020年前的标准方法；现在仅用于闭集设定。
- **分数归一化。** `S-norm` 或 `AS-norm`：使用一个冒充者均值和标准差队列对每个分数进行归一化。对于跨域评估至关重要。

### 你应该知道的数据（2026年）

| 模型 | VoxCeleb1-O EER | 参数量 | 吞吐量（A100） |
|-------|-----------------|--------|----------------|
| x-vector（经典） | 3.10% | 500万 | 400倍实时 |
| ECAPA-TDNN | 0.87% | 1500万 | 200倍实时 |
| WavLM-SV large | 0.42% | 3.16亿 | 20倍实时 |
| Pyannote 3.1 分割 + 嵌入 | 0.65% | 600万 | 100倍实时 |
| ReDimNet（2024） | 0.39% | 2400万 | 100倍实时 |

### 说话人日志化

“在多人对话片段中，谁在何时说了话”。流程：语音活动检测 → 分割 → 对每个片段计算嵌入 → 聚类（凝聚聚类或谱聚类） → 边界平滑。现代技术栈：`pyannote.audio` 3.1，它将说话人分割 + 嵌入 + 聚类打包在一次调用中。2026年在AMI数据集上的最佳日志化错误率约为15%（2022年为23%）。

## 动手构建

### 步骤1：基于MFCC统计量的玩具嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

这远非最新技术——仅用于教学目的。`code/main.py` 将其用作在合成说话人数据上的概念验证。

### 步骤2：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 步骤3：从相似度对计算EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回（eer, threshold_at_eer）。两者都需要报告。

### 步骤4：使用SpeechBrain进行生产

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### 步骤5：使用pyannote进行日志化

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 应用场景

2026年技术栈：

| 场景 | 选择 |
|------|------|
| 闭集1:1验证，边缘设备 | ECAPA-TDNN + 余弦阈值 |
| 开集验证，云端 | WavLM-SV + AS-norm |
| 日志化（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺骗（重放/深度伪造检测） | AASIST 或 RawNet2 |
| 极小嵌入式（关键词唤醒 + 注册） | Titanet-Small (NeMo) |

## 常见陷阱

- **信道不匹配。** 在VoxCeleb（网络视频）上训练的模型 ≠ 电话音频。务必在目标信道上进行评估。
- **短语音。** 测试音频低于3秒时，EER会急剧下降。
- **带噪声的注册。** 一次带噪声的注册会污染锚点。使用≥3个干净样本并取平均。
- **跨条件使用固定阈值。** 务必在目标域的保留开发集上调整阈值。
- **在未归一化嵌入上使用余弦相似度。** 先进行L2归一化；否则向量模长会主导结果。

## 部署上线

保存为`outputs/skill-speaker-verifier.md`。选择模型、注册方案、阈值调整计划和防欺诈措施。

## 练习

1. **简单。** 运行`code/main.py`。构建合成“说话人”（不同的音调特征），进行注册，在一个100对的测试列表上计算EER。
2. **中等。** 在30条VoxCeleb1语音（5名说话人 × 6条）上使用SpeechBrain的ECAPA模型。分别使用余弦相似度和PLDA计算EER。
3. **困难。** 使用`pyannote.audio`构建完整的注册 → 日志化 → 验证流程。在AMI开发集上评估DER。

## 关键术语

| 术语 | 人们常说 | 其实际含义 |
|------|----------|-----------|
| EER | 核心指标 | 误接受率 = 误拒绝率时的阈值。 |
| 验证 | 1:1 | “这是爱丽丝吗？” |
| 识别 | 1:N | “谁在说话？” |
| 开集 | 可能未知 | 测试集可包含未注册的说话人。 |
| 注册 | 登记 | 计算说话人的参考嵌入向量。 |
| AAM-softmax | 损失函数 | 带有加性角度间隔的softmax；强制聚类分离。 |
| PLDA | 经典评分方法 | 概率线性判别分析；在嵌入基础上进行似然比评分。 |
| DER | 日志化度量 | 日志化错误率——漏检 + 误报 + 混淆。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 经典的深度嵌入论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020–2026年的主导架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — 用于说话人验证和日志化的自监督学习骨干网络。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 生产级的日志化 + 嵌入技术栈。
- [VoxCeleb排行榜（2026年更新）](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 各模型当前的EER排名。