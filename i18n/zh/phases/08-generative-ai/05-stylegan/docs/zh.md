# StyleGAN

> 大多数生成器将 `z` 同时注入每个图层。StyleGAN 将其拆分：先将 `z` 映射为中间表示 `w`，然后通过 AdaIN 在每个分辨率层级中 *注入* `w`。这一改变解耦了潜在空间，使照片级人脸成为七年未变的基准问题。

**类型：** Build
**语言：** Python
**前置知识：** Phase 8 · 03 (GANs)，Phase 4 · 08 (Normalization)，Phase 3 · 07 (CNNs)
**时间：** 约 45 分钟

## 问题背景

DCGAN 通过一堆转置卷积将 `z` 映射为图像。问题在于：`z` 控制一切——姿态、光照、身份、背景——全都纠缠在一起。沿 `z` 的某个轴移动，四个维度会同时变化。你无法要求模型"同一个人、不同姿态"，因为表征并未按这种方式解耦。

Karras 等人（2019，NVIDIA）提出：停止直接将 `z` 输入卷积层。改为输入一个常数张量 `4×4×512`，并学习一个 8 层 MLP，将 `z ∈ Z` 映射为 `w ∈ W`。通过 *自适应实例归一化*（AdaIN）在每个分辨率处注入 `w`：对每个卷积特征图进行归一化，然后通过 `w` 的仿射投影进行缩放和平移。添加逐层噪声以生成随机细节（皮肤纹理、发丝）。

结果：`W` 中大致正交的轴分别对应"高层风格"（姿态、身份）和"精细风格"（光照、颜色）。你可以将两张图片的风格互换——用图片 A 的 `w` 处理低分辨率层级，用图片 B 的 `w` 处理高分辨率层级。这开启了编辑、跨域风格化和整个"StyleGAN 反演"研究路线。

## 核心概念

![StyleGAN: 映射网络 + AdaIN + 逐层噪声](../assets/stylegan.svg)

**映射网络。** `f: Z → W`，8 层 MLP。`Z = N(0, I)^512`。`W` 不强制为高斯分布——它学习一种适应数据分布的形状。

**合成网络。** 从学习到的常数 `4×4×512` 开始。每个分辨率块的结构为：`上采样 → 卷积 → AdaIN(w_i) → 噪声 → 卷积 → AdaIN(w_i) → 噪声`。分辨率依次翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。按特征图归一化，再进行风格重映射。这里的"风格"指的是特征图的一阶和二阶统计量。

**逐层噪声。** 向每个特征图添加单通道高斯噪声，并由学习到的逐通道因子缩放。控制随机细节而不影响整体结构。

**截断技巧。** 推理时采样 `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是大量样本中 `w` 的均值。`ψ < 1` 在多样性和质量之间做权衡。几乎所有 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 创新点 |
|------|------|--------|
| StyleGAN | 2019 | 映射网络 + AdaIN + 噪声 + 渐进式增长 |
| StyleGAN2 | 2020 | 权重解调替换 AdaIN（修复液滴伪影）；skip/residual 架构；路径长度正则化 |
| StyleGAN3 | 2021 | 无混叠卷积 + 等变核；消除纹理黏连到像素网格的问题 |
| StyleGAN-XL | 2022 | 类别条件，1024²，ImageNet |
| R3GAN | 2024 | 更强的正则化重命名；在 FFHQ-1024 上以少 20 倍参数逼近扩散模型 |

2026 年 StyleGAN3 仍是以下场景的默认选择：(a) 高 FPS 窄领域照片级真实感，(b) 少样本域适应（用 100 张图片训练新数据集，冻结映射网络），(c) 基于反演的编辑（找到能重建真实照片的 `w`，然后编辑该 `w`）。对于开放域文生图，它不是合适的工具——扩散模型才是。

```figure
gx-stylegan-mapping
```

## 动手实现

`code/main.py` 在 1D 下实现了一个玩具版"style-GAN lite"：一个映射 MLP、一个合成函数（接受学习到的常数向量并通过 `w` 派生的缩放/偏置进行调制），以及逐层噪声。它展示了通过仿射调制注入 `w` 比将 `z` 拼接到生成器输入更好或至少持平。

### 步骤 1：映射网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 步骤 2：自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

每个特征图的缩放和偏置通过线性投影从 `w` 得到。

### 步骤 3：逐层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每个通道的 sigma 是可学习的。

## 常见陷阱

- **液滴伪影。** StyleGAN 1 因 AdaIN 将均值置零而在特征图中产生 blobby 状液滴。StyleGAN 2 的权重解调通过缩放卷积权重而非激活值来修复此问题。
- **纹理黏连。** StyleGAN 1 和 2 的纹理跟随像素坐标而非物体坐标（插值时可见）。StyleGAN 3 的无混叠卷积通过窗口化 sinc 滤波器解决此问题。
- **模式覆盖。** 截断 `ψ < 0.7` 看起来干净，但仅采样映射网络输出空间中的一个狭窄锥体；如需多样性，请使用 `ψ = 1.0`。
- **反演是有损的。** 将真实照片反演为 `W` 通常通过优化或编码器（e4e、ReStyle、HyperStyle）完成。多次迭代后结果会漂移。

## 使用指南

| 用例 | 方法 |
|------|------|
| 照片级人脸（动漫、产品、窄领域） | StyleGAN3 FFHQ / 自定义微调 |
| 基于照片的人脸编辑 | e4e 反演 + StyleSpace / InterFaceGAN 方向 |
| 人脸交换 / 重演 | StyleGAN + 编码器 + 混合 |
| 虚拟形象管线 | StyleGAN3 + ADA 少数据微调 |
| 少样本域适应 | 冻结映射网络，微调合成网络 |
| 多模态或文本条件生成 | 不要用——改用扩散模型 |

对于回答是"人物照片"的产品级演示，StyleGAN 在推理成本（单次前向传播，4090 上 <10ms）和同等质量基准下的清晰度方面优于扩散模型。

## 交付物

保存至 `outputs/skill-stylegan-inversion.md`。技能需包含：反演方法（e4e / ReStyle / HyperStyle）、预期潜在损失、编辑预算（在 `W` 中能移动多远才出现伪影），以及已知安全的编辑方向列表（年龄、表情、姿态）。

## 练习题

1. **简单。** 用 `adain_on=True` 和 `adain_on=False` 分别运行 `code/main.py`。比较固定潜在向量和扰动潜在向量下输出的分布差异。
2. **中等。** 实现混合正则化：对训练批次计算 `w_a`、`w_b`，合成前半段使用 `w_a`，后半段使用 `w_b`。解码器是否学到了解耦的风格？
3. **困难。** 取一个预训练的 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。通过在标注样本上训练 SVM 找到控制"微笑"的 `w` 方向；报告在身份漂移之前能推动多远。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 映射网络 | "那个 MLP" | `f: Z → W`，8 层，解耦潜在几何与数据统计 |
| W 空间 | "风格空间" | 映射网络的输出；大致解耦 |
| AdaIN | "自适应实例归一化" | 归一化特征图，再通过 `w` 投影进行缩放+平移 |
| 截断技巧 | "Psi" | `w = mean + ψ·(w - mean)`，ψ<1 在多样性与质量间权衡 |
| 路径长度正则化 | "PL reg" | 惩罚 `w` 单位变化引起的图像大变化；使 `W` 更平滑 |
| 权重解调 | "StyleGAN2 的修复" | 归一化卷积权重而非激活值；消除液滴伪影 |
| 无混叠 | "StyleGAN3 的技巧" | 窗口化 sinc 滤波器；消除纹理黏连到像素网格 |
| 反演 | "为真实图像找 w" | 优化或编码 `x → w` 使得 `G(w) ≈ x` |

## 生产备注：为何 StyleGAN 在 2026 年仍被广泛使用

StyleGAN3 在 4090 上以不到 10ms 的时间生成 1024² 的 FFHQ 人脸——`num_steps = 1`，无需 VAE 解码，无需交叉注意力。在生产环境中，这是任何图像生成器的最低延迟基线。同等分辨率下，50 步 SDXL + VAE 解码管线约需 3 秒。这存在 **300 倍差距**，对于窄领域产品（虚拟形象服务、身份证照片生成、虚拟人脸素材）而言，总拥有成本（TCO）更优。

两个运营层面的推论：

- **无需调度器或批处理。** 静态批次在目标负载下最优。连续批处理（对 LLM 和扩散模型至关重要）对此毫无增益，因为每个请求的计算量相同。
- **截断 `ψ` 是安全旋钮。** `ψ < 0.7` 仅在映射网络输出空间的一个狭窄锥体内采样。这是服务层控制样本方差的唯一杠杆。高峰期降低 `ψ`，为付费用户提高 `ψ`。

## 延伸阅读

- [Karras 等 (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras 等 (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras 等 (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov 等 (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e 反演。
- [Sauer 等 (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang 等 (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 现代极简 GAN 配方。
