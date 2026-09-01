# Show-o 与离散扩散统一模型

> Transfusion 混合了连续和离散表示。Show-o（Xie 等人，2024年8月）反其道而行：文本 token 使用因果下一步 token 预测，图像 token 使用受 MaskGIT 启发的掩码离散扩散。两者位于同一个 transformer 中，配合混合注意力掩码。结果在一个骨干网络、每种模态一个分词器、一个扩展了掩码预测的损失公式之上，统一了 VQA、文生图、图像修复和混合模态生成。本课讲解 Show-o 的设计——为何掩码离散扩散是一种并行、少步数的图像生成器——并与 Transfusion 和 Emu3 进行对比。

**类型：** 学习
**语言：** Python（标准库、掩码离散扩散采样器）
**前置知识：** 第 12 · 13 课（Transfusion）
**时间：** 约 120 分钟

## 学习目标

- 解释掩码离散扩散：均匀掩码 token 然后要求 transformer 恢复它们的调度策略。
- 在速度和质量上比较并行图像解码（Show-o、MaskGIT）与自回归图像解码（Chameleon、Emu3）。
- 说出 Show-o 在一个检查点中处理的三个任务：T2I、VQA、图像修复。
- 选择一种掩码调度（余弦、线性、截断）并分析其对样本质量的影响。

## 问题所在

Transfusion 的双损失训练虽然有效，但动态更复杂——连续扩散损失与离散 NTP 损失处于不同的数值尺度上。平衡损失权重是一个超参数搜索过程。架构虽然有效但复杂。

Show-o 的方案是：保持两种模态都是离散的（像 Chameleon），但通过掩码离散扩散并行生成图像，而非顺序生成。训练目标变为一个统一的掩码 token 预测，自然地推广了下一步 token 预测。

## 核心概念

### 掩码离散扩散（MaskGIT）

Chang 等人（2022）的原始 MaskGIT 技巧非常优雅。从全掩码图像开始（每个 token 都是特殊的 `<MASK>` id）。每一步并行预测所有掩码 token，然后保留置信度最高的 top-K 预测，重新掩码其余部分。经过约 8-16 次迭代，所有 token 都被填完。每步解封多少 token 的调度经过调优——余弦调度效果良好。

训练很简单：从 [0, 1] 中均匀采样一个掩码比例，应用到图像的 VQ token 上，训练 transformer 恢复被掩码的部分。与 BERT 对文本做的事情如出一辙，只是扩展到了图像生成。

### Show-o：一个 transformer，混合掩码

Show-o 将 MaskGIT 放入因果语言模型 transformer 中。注意力掩码为：

- 文本 token：因果（标准 LLM）。
- 图像 token：图像块内完全双向（因此掩码 token 在预测时能看到其他所有图像 token）。
- 文本到图像：文本 attend 到先前图像，图像 attend 到先前文本。

训练在以下模式间交替：
1. 文本序列的标准 NTP。
2. T2I 样本：文本 → 图像，图像 token 被掩码，使用掩码 token 预测损失。
3. VQA 样本：图像 → 文本，文本 token 被掩码（实际上只是 NTP）。

统一损失是对 `<MASK>` token 的交叉熵，涵盖了文本 NTP（只有最后一个 token 是"掩码"的）和图像掩码扩散（随机子集被掩码）。

### 并行采样

Show-o 在约 16 步内生成一张图像，而非自回归每 token ~1000 步或扩散 ~20 步。每一步并行预测所有掩码 token；提交 top-K 置信度最高的；重复。

对比：
- Chameleon / Emu3（token 级自回归）：N_tokens 次前向传递，每张图像通常 1024-4096 次。
- Transfusion（连续扩散）：约 20 步，每步一次完整 transformer 传递。
- Show-o（掩码离散扩散）：约 16 步，每步一次完整 transformer 传递。

在同等规模模型下，Show-o 比 Chameleon 更快，步数与 Transfusion 大致相当且每步成本更低（离散词表 logits vs 连续 MSE 损失）。

### 一个检查点中的多个任务

Show-o 支持四种推理任务，通过提示格式选择：

- 文本生成：标准自回归文本输出。
- VQA：图像输入，文本输出。
- T2I：文本输入，通过掩码离散扩散输出图像。
- 图像修复：部分 token 被掩码的图像，填入缺失部分。

图像修复能力来自掩码预测训练，是免费获得的。掩码 VQ-token 网格中的某个区域，输入其余部分加文本提示，预测掩码 token。

### 掩码调度

每步解封多少 token 的调度决定质量。Show-o 推荐使用余弦调度：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

第 0 步时所有 token 被掩码（比例 1.0）。第 T 步时无 token 被掩码。余弦调度将质量集中在预测最有信息的中间范围比例上。线性调度也能工作，但更快进入平台期。

### Show-o2

Show-o2（2025 年后续工作，arXiv 2506.15564）扩展了 Show-o：更大的 LLM 基座、更好的分词器、改进的掩码调度。保持相同的架构模式。

### Show-o 的定位

在 2026 年的分类法中：

- 离散 token + NTP：Chameleon、Emu3。简单但推理慢。
- 离散 token + 掩码扩散：Show-o、MaskGIT、LlamaGen、Muse。并行采样，仍受分词器Lossy影响。
- 连续 + 扩散：Transfusion、MMDiT、DiT。质量最高，训练更复杂。
- 连续 + 流匹配在 VLM 中：JanusFlow、InternVL-U。最新一代。

按任务选择：当你需要在开放模型中同时获得 T2I + 图像修复 + VQA 且速度合理时选 Show-o；当质量 paramount 且你能接受双损失基础设施时选 Transfusion。

```figure
masked-diffusion-unmask
```

## 使用它

`code/main.py` 模拟了 Show-o 采样：

- 一个 16 个 VQ token 的玩具网格。
- 一个基于提示和当前未掩码 token 预测 logits 的模拟"transformer"。
- 8 步余弦调度的并行掩码采样。
- 打印中间状态（掩码模式演变）和最终 token。

运行它，观察掩码一步步溶解。

## 交付物

本课产出 `outputs/skill-unified-gen-model-picker.md`。针对一个既需要理解能力（VQA、图像描述）又需要生成能力（T2I、图像修复）且要求开放权重的产品，在 Show-o 系列、Transfusion/MMDiT 系列和 Emu3/Chameleon 系列之间做出选择，并给出具体权衡。

## 练习

1. 掩码离散扩散在约 16 步内采样。为什么不能是 1 步？如果第 0 步就解封所有 token，会发生什么？

2. 图像修复在掩码扩散中是免费获得的。提出一个产品用例（真实或假设），说明 Show-o 的图像修复能力优于专项模型。

3. 余弦调度 vs 线性调度：追踪 T=8 时每步未掩码 token 的数量。哪种更均衡？

4. 一张 512×512 的 Show-o 图像是 1024 个 token。在词表大小 K=16384 时，模型输出 1024 × log₂(16384) = 14,336 比特（约 1.75 KiB）数据。Stable Diffusion 输出 512×512×24 比特 = 6,291,456 比特（约 768 KiB）原始像素。压缩比是多少？换取了怎样的质量？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的类别条件自回归图像模型与 Show-o 的掩码方法有何不同？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 掩码离散扩散 | "MaskGIT 风格" | 训练预测掩码 token；推理时迭代解封置信度最高的预测 |
| 余弦调度 | "解封调度" | 推理步数间的掩码比例衰减；将置信度增长集中在中间范围 |
| 并行解码 | "一次所有 token" | 每一步在一次前向传递中预测完整的掩码 token 序列，然后提交 top-K |
| 混合注意力 | "因果 + 双向" | 文本 token 上因果、图像块内双向的掩码 |
| 图像修复 | "填充生成" | 以部分 token 被掩码的图像为条件，预测缺失部分；从训练目标中免费获得 |
| 提交率 | "每步 top-K" | 每次迭代声明"完成"的 token 数量；控制推理速度与质量的权衡 |

## 延伸阅读

- [Xie 等人 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang 等人 — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun 等人 — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang 等人 — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
