# 说话人识别与验证

> ASR 问"他们说了什么？"说话人识别问"谁说的？"数学看起来一样——嵌入向量加余弦相似度——但每个生产决策都取决于一个 EER 数值。

**类型：** 实践项目
**语言：** Python
**前置知识：** Phase 6 · 02 (频谱图与梅尔谱)、Phase 5 · 22 (嵌入模型)
**时间：** 约 45 分钟

## 问题

用户念一段 passphrase（认证短语）。你想要知道：这是他们声称的那个人吗（*验证*，1:1），还是你注册库中的第一个说话人（*辨认*，1:N）？或者既不是——这是一个未知说话人（*开放集*）？

2018 年以前：GMM-UBM + i-vector。EER 尚可，但对信道偏移（手机 vs 笔记本）和情感变化很敏感。2018–2022 年：x-vector（TDNN 主干 + 角度边距训练）。2022 年起：ECAPA-TDNN 和 WavLM-large 嵌入。到 2026 年，该领域由三种模型和一个指标主导。

指标是 **EER**——等错误率。设置决策阈值使得 FAR（误识率）= FRR（拒识率）。交点即为 EER。每篇论文、每个排行榜、每次采购评估都会用到。

## 概念

![注册与验证流程：嵌入 + 余弦相似度 + EER](../assets/speaker-verification.svg)

**流程。** 注册：录制目标说话人 5–30 秒的语音；计算固定维度的嵌入向量（ECAPA-TDNN 为 192 维，WavLM-large 为 256 维）。验证：获取测试 utterance 的嵌入向量；计算余弦相似度；与阈值比较。

**ECAPA-TDNN（2020，2026 仍占主流）。** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network（强调通道注意力、传播与聚合的时延神经网络）。1D 卷积块带 squeeze-excitation，多注意力头池化，再接线性层输出 192 维。在 VoxCeleb 1+2（2,700 个说话人，110 万 utterance）上用 Additive Angular Margin loss（AAM-softmax）训练。

**WavLM-SV（2022+）。** 用 AAM loss 微调预训练的 WavLM-large SSL 主干。质量更高但更慢——300+ MB vs 15 MB。

**x-vector（基线）。** TDNN + 统计池化。经典方案；在 CPU / 边缘设备上仍有价值。

**AAM-softmax。** 在角度空间中加入边距 `m` 的标准 softmax：对正确类取 `cos(θ + m)`。强制类间角度分离。典型参数 `m=0.2`，尺度 `s=30`。

### 评分方式

- **余弦相似度。** 计算注册嵌入与测试嵌入的余弦相似度，基于阈值决策。
- **PLDA（概率 LDA）。** 将嵌入投影到潜在空间，同说话人与不同说话人之间有闭式似然比。叠加在余弦之上可减少 EER 10–20%。2020 年前为标准做法；现在仅用于封闭集场景。
- **分数归一化。** `S-norm` 或 `AS-norm`：用一组冒充者的均值和标准差对每个分数做归一化。跨域评估必备。

### 你应了解的数字（2026）

| 模型 | VoxCeleb1-O EER | 参数量 | 吞吐量（A100） |
|------|-----------------|--------|----------------|
| x-vector（经典） | 3.10% | 500 万 | 400× RT |
| ECAPA-TDNN | 0.87% | 1,500 万 | 200× RT |
| WavLM-SV large | 0.42% | 3.16 亿 | 20× RT |
| Pyannote 3.1 分割 + 嵌入 | 0.65% | 600 万 | 100× RT |
| ReDimNet（2024） | 0.39% | 2,400 万 | 100× RT |

### 说话人日志（Diarization）

多说话人片段中"谁在什么时候说话"。流程：VAD → 分段 → 对每段计算嵌入 → 聚类（凝聚聚类或谱聚类）→ 平滑边界。现代方案：`pyannote.audio` 3.1，一键完成说话人分割 + 嵌入 + 聚类。2026 年 AMI 数据集 SOTA DER 约 15%（2022 年为 23%）。

```figure
sp-eer-crossover
```

## 动手实践

### 步骤 1：基于 MFCC 统计的玩具嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26 维
```

离 SOTA 还差得远——仅用于教学。`code/main.py` 用它作为合成说话人数据的概念验证。

### 步骤 2：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 步骤 3：从相似度对计算 EER

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

返回 (eer, 阈值_at_eer)。两个值都要报告。

### 步骤 4：用 SpeechBrain 构建生产级系统

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll：对 3-5 个干净样本的嵌入取平均
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA 典型阈值；根据你自己的数据调优
```

### 步骤 5：用 pyannote 做说话人日志

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 应用场景

2026 年技术栈：

| 场景 | 推荐方案 |
|------|----------|
| 封闭集 1:1 验证，边缘部署 | ECAPA-TDNN + 余弦阈值 |
| 开放集验证，云端部署 | WavLM-SV + AS-norm |
| 说话人日志（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 防欺骗（重放/深度伪造检测） | AASIST 或 RawNet2 |
| 超小嵌入式（KWS + 注册） | Titanet-Small（NeMo） |

## 陷阱

- **信道失配。** 模型在 VoxCeleb（网络视频）上训练 ≠ 电话录音。务必在目标信道上评估。
- **短 utterance。** 测试音频低于 3 秒时 EER 急剧恶化。
- **含噪注册。** 一个含噪注册样本就会污染锚点。使用 ≥3 个干净样本并取平均。
- **跨条件固定阈值。** 始终在目标域的保留开发集上调优阈值。
- **未归一化嵌入做余弦。** 先 L2 归一化，否则幅度会主导结果。

## 交付

保存为 `outputs/skill-speaker-verifier.md`。选定模型、注册协议、阈值调优计划和反欺诈保障措施。

## 练习

1. **简单。** 运行 `code/main.py`。构建合成"说话人"（不同音色特征），注册后在 100 对测试列表上计算 EER。
2. **中等。** 用 SpeechBrain ECAPA 处理 30 条 VoxCeleb1 utterance（5 个说话人 × 每人 6 条）。分别用余弦和 PLDA 计算 EER。
3. **困难。** 用 `pyannote.audio` 构建完整的注册 → 说话人日志 → 验证流程。在 AMI dev 集上评估 DER。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| EER |  headline 指标 | 误识率 = 拒识率的阈值点。 |
| 验证 (Verification) | 1:1 | "这是 Alice 吗？" |
| 辨认 (Identification) | 1:N | "谁在说话？" |
| 开放集 (Open-set) | 可能存在未知说话人 | 测试集可包含未注册的说话人。 |
| 注册 (Enrollment) | 登记 | 计算某说话人的参考嵌入向量。 |
| AAM-softmax | 损失函数 | 带加法角度边距的 softmax；强制簇间分离。 |
| PLDA | 经典评分 | 概率 LDA；在嵌入之上做似然比评分。 |
| DER | 日志指标 | Diarization Error Rate——漏检 + 虚警 + 混淆之和。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) —— 经典深度嵌入论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) —— 2020–2026 年的主流架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) —— 用于 SV 和说话人日志的 SSL 主干。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) —— 生产级说话人日志 + 嵌入方案。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) —— 各模型最新 EER 排名。
