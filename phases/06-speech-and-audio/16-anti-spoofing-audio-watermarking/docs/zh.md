# 语音反欺骗与音频水印 — ASVspoof 5、AudioSeal、WaveVerify

> 语音克伦技术的普及速度超过了防御手段。2026年的生产级语音系统需要两个核心组件：一个检测器（AASIST、RawNet2）用于区分真实语音与伪造语音，以及一个水印（AudioSeal）能够抵御压缩和编辑操作。两者缺一不可，否则就不要部署语音克伦。

**类型：** 构建
**语言：** Python
**先修要求：** 阶段6 · 06（说话人识别），阶段6 · 08（语音克隆）
**时间：** ~75 分钟

## 问题

三种相互关联的防御手段：

1.  **反欺骗/深度伪造检测。** 给定一段音频，判断其是合成还是真实。ASVspoof 基准测试（ASVspoof 2019 → 2021 → 5）是该领域的黄金标准。
2.  **音频水印。** 在生成的音频中嵌入一个人类难以察觉的信号，该信号可被检测器后续提取。AudioSeal（Meta）和 WavMark 是目前可用的开源选项。
3.  **可认证来源。** 对音频文件及元数据进行加密签名。C2PA / 内容真实性倡议。

检测应对的是不合作的攻击者。水印则用于满足合规性——AI生成的音频应能被标识。2026年，两者均不可或缺。

## 概念

![反欺骗 vs 水印 vs 来源 — 三层防御](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024-2025 基准测试

与之前版本相比，最大的变化是：

-   **众包数据**（非录音棚纯净数据）——更贴近现实条件。
-   **~2000 位说话人**（之前仅~100位）。
-   **32 种攻击算法。** 包含文本到语音、语音转换和对抗性扰动。
-   **两条赛道。** 反制措施（CM）独立检测；抗欺骗的说话人验证（SASV），面向生物特征识别系统。

ASVspoof 5 上的最新水平：~7.23% EER。在较早的 ASVspoof 2019 LA 上：0.42% EER。实际部署中，对野外采集的片段，预期 EER 为 5-10%。

### AASIST 与 RawNet2 — 检测模型系列

**AASIST**（2021年，更新至2026年）。基于频谱特征的图注意力机制。目前 ASVspoof 5 反制措施任务的 SOTA 模型。

**RawNet2。** 针对原始波形的卷积前端 + TDNN 主干网络。更简单的基线；经过微调后仍具竞争力。

**NeXt-TDNN + SSL 特征。** 2025 年的变体：ECAPA 风格 + WavLM 特征 + 焦点损失。在 ASVspoof 2019 LA 上达到了 0.42% EER。

### AudioSeal — 2024 年的水印默认选择

Meta 的 **AudioSeal**（2024年1月，v0.2 更新于2024年12月）。关键设计：

-   **局部化。** 以 16 kHz 采样率（1/16000 秒）逐帧检测水印。
-   **生成器与检测器联合训练。** 生成器学习嵌入听不见的信号；检测器学习在数据增强下找到它。
-   **鲁棒性强。** 能抵御 MP3 / AAC 压缩、均衡器调整、±10% 速度变化、+10 dB 信噪比下的噪声混合。
-   **速度快。** 检测器运行速率为实时 485 倍；比 WavMark 快 1000 倍。
-   **容量。** 16 位有效载荷（可编码模型ID、生成时间戳、用户ID）可嵌入到每条语句中。

### WavMark

AudioSeal 出现之前的开源基线。使用可逆神经网络，速率 32 位/秒。问题：

-   同步暴力破解速度慢。
-   可通过高斯噪声或 MP3 压缩去除。
-   对实时应用不友好。

### WaveVerify（2025年7月）

旨在解决 AudioSeal 的弱点——特别是时间上的操作（反转、变速）。采用基于 FiLM 的生成器 + 混合专家检测器。在标准攻击下与 AudioSeal 具有竞争力；能够处理时间编辑。

### 攻击者利用的漏洞

来自 AudioMarkBench 的结论：“在音高偏移下，所有水印的比特恢复准确率均低于 0.6，表明几乎被完全移除。” **音高偏移是普遍攻击方式。** 截至2026年，尚无水印技术能完全抵御激进的音高修改。这就是为什么你需要将检测（AASIST）与水印结合使用。

### C2PA / 内容真实性倡议

这不是一种 ML 技术，而是一种清单格式。音频文件携带有关创建工具、作者、日期的加密签名元数据。Audbox / Seamless 使用它。对溯源有益；但如果恶意行为者重新编码并剥离元数据，则此方法无效。

## 动手构建

### 步骤 1：一个简单的频谱特征检测器（示例）

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

合成语音通常具有异常平坦的高频能量分布。生产环境的检测器使用 AASIST，而非此方法。但基本思路是相通的。

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
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
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

### 步骤 4：生产环境集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成都应包含：(1) 水印，(2) 签名的清单，(3) 符合保留策略的审计日志。

## 使用场景

| 使用场景 | 防御手段 |
|----------|----------|
| 部署文本到语音/语音克隆 | 在每个输出上嵌入 AudioSeal（无妥协余地） |
| 生物特征语音解锁 | AASIST + ECAPA 集成；活跃性挑战 |
| 呼叫中心欺诈检测 | 对 20% 的来电样本应用 AASIST |
| 播客真实性验证 | 上传时使用 C2PA 签名，若为AI生成则使用 AudioSeal |
| 研究/训练检测器 | ASVspoof 5 训练/开发/评估集 |

## 常见陷阱

-   **部署了水印却从未运行检测器。** 毫无意义。在你的 CI 流程中加入检测器。
-   **未进行校准的检测。** 在 ASVspoof LA 上训练的 AASIST 会过拟合；真实世界的准确率会下降。在你的领域数据上进行校准。
-   **音高偏移漏洞。** 激进的音高偏移会移除大多数水印。需要有检测作为后备方案。
-   **元数据剥离与重新托管。** C2PA 可以通过重新编码轻易绕过。始终将加密和感知（水印）防御结合起来。
-   **将活跃性检测等同于欺骗检测。** 要求用户说一个随机短语可以防止重放攻击，但无法防止实时克隆。

## 部署清单

保存为 `outputs/skill-spoof-defender.md`。为语音生成部署选择检测模型、水印、来源清单和运维手册。

## 练习

1.  **简单。** 运行 `code/main.py`。使用玩具检测器 + 玩具水印对合成音频进行嵌入/检测。
2.  **中等。** 安装 `audioseal`，在一个 TTS 输出中嵌入 16 位有效载荷，然后重新解码。用噪声损坏音频并测量比特恢复准确率。
3.  **困难。** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST。测量 EER。在一个保留的 F5-TTS 生成片段集上测试——观察分布外检测性能的下降情况。

## 关键术语

| 术语 | 人们常说什么 | 实际含义 |
|------|--------------|----------|
| ASVspoof | 那个基准 | 双年度挑战赛；2024年 = ASVspoof 5。 |
| CM (反制措施) | 检测器 | 分类器：真实语音 vs 合成/转换语音。 |
| SASV | 说话人验证 + CM | 集成了生物特征验证与欺骗检测。 |
| AudioSeal | Meta 的水印 | 局部化、16位有效载荷，比 WavMark 快 485 倍。 |
| 比特恢复准确率 | 水印存活率 | 攻击后恢复的有效载荷位比例。 |
| C2PA | 来源清单 | 关于创建/作者身份的加密元数据。 |
| AASIST | 检测器系列 | 基于图注意力机制的反欺骗 SOTA 模型。 |

## 扩展阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前的基准测试。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 水印默认选择。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 用于时间攻击的混合专家检测器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA 检测骨干网络。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 鲁棒性评估。
- [C2PA 规范](https://c2pa.org/specifications/specifications/) — 来源清单格式。