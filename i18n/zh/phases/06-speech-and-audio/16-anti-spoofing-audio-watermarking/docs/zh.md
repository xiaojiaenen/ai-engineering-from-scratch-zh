# 语音反伪造与音频水印 — ASVspoof 5、AudioSeal、WaveVerify

> 语音克隆的部署速度远超防御措施。2026 年的生产级语音系统需要两样东西：一个分类真假语音的检测器（AASIST、RawNet2），以及一个能抵御压缩和编辑的水印（AudioSeal）。两者缺一不可。

**类型：** 构建
**语言：** Python
**前置要求：** 第 6 阶段 · 06（说话人识别）、第 6 阶段 · 08（语音克隆）
**时间：** 约 75 分钟

## 问题所在

三项相关防御：

1. **反伪造 / 深度伪造检测。** 给定一段音频，它是合成的还是真实的？ASVspoof 基准（ASVspoof 2019 → 2021 → 5）是行业标准。
2. **音频水印。** 在生成音频中嵌入一个不可感知的信号，供后续检测器提取。AudioSeal（Meta）和 WavMark 是开源方案。
3. **可信来源证明。** 对音频文件及元数据进行加密签名。C2PA / Content Authenticity Initiative。

检测应对不配合的对手，水印应对合规要求——AI 生成的音频应当能被识别为 AI 生成。2026 年两者均需部署。

## 核心概念

![反伪造 vs 水印 vs 来源证明 — 三层防御](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024-2025 年度基准

与前几届相比最大变化：

- **众包数据**（非工作室.clean）— 贴近真实场景。
- **约 2000 名说话人**（此前约 100 人）。
- **32 种攻击算法**，涵盖 TTS + 声音转换 + 对抗扰动。
- **两个赛道**：Countermeasure（CM）独立检测；Spoofing-robust ASV（SASV）面向生物识别系统。

ASVspoof 5 当前最优 EER 约 7.23%。在较老的 ASVspoof 2019 LA 上为 0.42% EER。真实世界部署：野外音频预期 5-10% EER。

### AASIST 和 RawNet2 — 检测模型族

**AASIST**（2021，持续更新至 2026）。基于图注意力机制处理频谱特征。ASVspoof 5 反伪造任务当前 SOTA。

**RawNet2**。原始波形上的卷积前端 + TDNN 主干。更简单的基线方案，微调后仍具竞争力。

**NeXt-TDNN + SSL 特征**。2025 变体：ECAPA 风格 + WavLM 特征 + focal loss。在 ASVspoof 2019 LA 上达到 0.42% EER。

### AudioSeal — 2024 年水印默认方案

Meta 的 **AudioSeal**（2024 年 1 月，v0.2 于 2024 年 12 月）。核心设计：

- **局部化。** 以 16 kHz 采样分辨率逐帧检测水印（每帧 1/16000 秒）。
- **生成器与检测器联合训练。** 生成器学习嵌入不可听信号；检测器学习在增强下提取它。
- **鲁棒性强。** 可抵御 MP3/AAC 压缩、均衡器调整、±10% 变速、+10 dB SNR 噪声混合。
- **速度快。** 检测器运行速度达实时 485 倍；比 WavMark 快 1000 倍。
- **容量大。** 16-bit 载荷（可在每条 utterance 中编码模型 ID、生成时间戳、用户 ID）。

### WavMark

AudioSeal 之前的开源基线方案。可逆神经网络，32 bit/秒。存在问题：

- 同步暴力破解速度慢。
- 可被高斯噪声或 MP3 压缩去除。
- 不适合实时场景。

### WaveVerify（2025 年 7 月）

针对 AudioSeal 的弱点——特别是时间操作（倒放、变速）。使用 FiLM 生成器 + Mixture-of-Experts 检测器。在标准攻击上与 AudioSeal 相当，并额外处理时间编辑。

### 对手利用的漏洞

来自 AudioMarkBench："在音高偏移下，所有水印的比特恢复准确率均低于 0.6，表明几乎被完全移除。"**音高偏移是通用攻击。** 2026 年没有任何水印能完全抵御激进的音高修改。这就是为什么你需要在水印之外同时部署检测器（AASIST）。

### C2PA / Content Authenticity Initiative

不是 ML 技术，而是一种清单格式。音频文件携带加密签名的元数据，记录创建工具、作者、日期。Audobox / Seamless 等已采用。适用于来源追溯，但若恶意行为者重新编码并剥离元数据则毫无用处。

```figure
v4-audio-watermark
```

## 动手构建

### 步骤 1：简单频谱特征检测器（玩具版）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音通常高频能量异常平坦。生产级检测器应使用 AASIST，而非此方案。但直觉上成立。

### 步骤 2：AudioSeal 嵌入与检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: [0, 1] 浮点数——水印存在概率
# decoded_payload: 16 bit；与嵌入载荷匹配
```

### 步骤 3：评估 — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 步骤 4：生产集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成均需输出：（1）水印，（2）签名清单，（3）符合保留策略的审计日志。

## 应用场景

| 场景 | 防御方案 |
|------|---------|
| 交付 TTS / 语音克隆 | 每条输出嵌入 AudioSeal（不可妥协） |
| 生物识别语音解锁 | AASIST + ECAPA 集成；活体检测挑战 |
| 呼叫中心欺诈检测 | 对 20% 抽样来电运行 AASIST |
| 播客真实性验证 | 上传时 C2PA 签名，AI 生成内容加 AudioSeal |
| 研究 / 训练检测器 | ASVspoof 5 train/dev/eval 数据集 |

## 常见陷阱

- **只水印不部署检测器。** 毫无意义。在 CI 中集成检测器。
- **检测器未校准。** AASIST 在 ASVspoof LA 上训练会过拟合；真实环境准确率下降。需在目标领域上校准。
- **音高偏移漏洞。** 激进音高偏移可移除大多数水印。需有检测器作为兜底。
- **元数据剥离重上传。** C2PA 可通过重新编码轻松绕过。必须同时部署加密 + 感知（水印）双重防御。
- **把活体检测当万全之策。** 让用户念随机短语可防止重放攻击，但防不住实时克隆。

## 交付物

保存至 `outputs/skill-spoof-defender.md`。为语音生成部署选定检测模型、水印、来源证明清单及运营手册。

## 练习

1. **简单。** 运行 `code/main.py`。玩具检测器 + 玩具水印嵌入/检测在合成音频上。
2. **中等。** 安装 `audioseal`，在 TTS 输出中嵌入 16-bit 载荷并重新解码。用噪声污染音频，测量比特恢复准确率。
3. **困难。** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST。测量 EER。在保留的 F5-TTS 生成片段集上测试——观察 OOD 检测性能如何退化。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| ASVspoof | 基准测试 | 双年度挑战赛；2024 年为 ASVspoof 5。 |
| CM（反制措施） | 检测器 | 分类器：真实语音 vs 合成/转换语音。 |
| SASV | 说话人验证 + CM | 集成生物识别 + 反伪造检测。 |
| AudioSeal | Meta 水印 | 局部化，16-bit 载荷，比 WavMark 快 485 倍。 |
| 比特恢复准确率 | 水印存活率 | 攻击后恢复的载荷比特占比。 |
| C2PA | 来源证明清单 | 关于创建/署名的加密元数据。 |
| AASIST | 检测模型族 | 基于图注意力的反伪造 SOTA。 |

## 延伸阅读

- [Todisco 等（2024）。ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前基准。
- [Defossez 等（2024）。AudioSeal](https://arxiv.org/abs/2401.17264) — 水印默认方案。
- [Chen 等（2025）。WaveVerify](https://arxiv.org/abs/2507.21150) — 面向时间攻击的 MoE 检测器。
- [Jung 等（2022）。AASIST](https://arxiv.org/abs/2110.01200) — SOTA 检测骨干。
- [AudioMarkBench（2024）](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 鲁棒性评估。
- [C2PA 规范](https://c2pa.org/specifications/specifications/) — 来源证明清单格式。
