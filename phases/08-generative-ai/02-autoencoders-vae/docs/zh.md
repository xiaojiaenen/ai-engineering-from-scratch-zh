# 自编码器与变分自编码器 (VAE)

> 一个普通的自编码器进行压缩然后重建。它是在记忆，而不是在生成。只需添加一个小技巧——迫使编码看起来像高斯分布——你就得到了一个采样器。正是这个技巧，即对 `z = μ + σ·ε` 的重参数化，使得你在2026年使用的每一个潜在扩散和流匹配图像模型的输入端都有一个VAE。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段3 · 02 (反向传播), 阶段3 · 07 (卷积神经网络), 阶段8 · 01 (分类学)
**时间：** 约75分钟

## 问题所在

将784像素的MNIST数字压缩为一个16维的编码，然后重建。一个普通的自编码器能很好地完成重建任务（均方误差低），但其编码空间是一团糟。在编码空间中随机选一个点进行解码，你会得到噪声。它没有采样能力。它只是一个伪装成生成模型的压缩模型。

你真正想要的是：(a) 编码空间是一个干净、平滑的分布，你可以从中采样——比如一个各向同性高斯分布 `N(0, I)`；(b) 解码任何样本都能产生一个合理的数字；(c) 编码器和解码器仍然具有良好的压缩性能。三个目标，一个架构，一个损失函数。

Kingma 于2013年提出的VAE通过以下方式解决此问题：训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)`，通过KL惩罚项将该分布拉向先验 `N(0, I)`，然后在解码前从 `q(z|x)` 中采样 `z`。在推理时，丢弃编码器，直接从 `z ~ N(0, I)` 采样并解码。正是KL惩罚项迫使编码空间变得结构化。

在2026年，VAE很少独立使用——在原始图像质量方面已被扩散模型超越——但它们是每个潜在扩散模型（SD 1/2/XL/3、Flux、AudioCraft）首选的编码器。学习VAE，就是学习你所使用的每一个图像流水线中不可见的第一层。

## 核心概念

![自编码器 vs VAE：重参数化技巧](../assets/vae.svg)

**自编码器。** `z = encoder(x)`, `x̂ = decoder(z)`, 损失 = `||x - x̂||²`。编码空间无结构。

**VAE编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义了 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 从 `q(z|x)` 中采样是不可微分的。将采样重写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 的确定性函数加上一个无参数的噪声——梯度可以流过 `μ` 和 `σ`。

**损失函数。** 证据下界 (ELBO)，包含两项：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建项推动 `x̂` 接近 `x`。KL项推动 `q(z|x)` 接近先验。两者相互权衡。较小的β（<1）= 更清晰的样本，编码空间高斯性较弱。较大的β（>1）= 更规整的编码空间，更模糊的样本。β-VAE (Higgins, 2017) 因这个调节参数而闻名，并推动了解纠缠研究。

**采样。** 在推理时：抽取 `z ~ N(0, I)`，通过解码器前向传播。一次前向传播——没有像扩散模型那样的迭代采样。

## 构建它

`code/main.py` 实现了一个不使用numpy或torch的小型VAE。输入是8维的合成数据，来自8维空间中的2分量高斯混合模型。编码器和解码器都是具有单个隐藏层的MLP。我们实现了tanh激活函数、前向传播、损失函数以及手写的反向传播。这并非生产代码——而是用于教学。

### 第1步：编码器前向传播

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

使用 `log σ²` 而不是 `σ`，这样网络输出是无约束的（σ的softplus是一个陷阱——当σ ≈ 0时梯度会消失）。

### 第2步：重参数化与解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 第3步：ELBO计算

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

精确的闭合形式KL计算，因为两个分布都是高斯分布。不要进行数值积分。在2026年，仍然有人使用蒙特卡洛方法估算KL——这会让速度慢3倍且毫无必要。

### 第4步：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## 陷阱

- **后验坍缩。** KL项过于激进地将 `q(z|x) → N(0, I)` 拉向先验，导致 `z` 不包含任何关于 `x` 的信息。解决方法：β退火（从β=0开始，逐渐增加到1）、自由比特，或在非活跃维度上跳过KL计算。
- **样本模糊。** 高斯解码器似然意味着均方误差重建，这对于L2范数（均值）是贝叶斯最优的——一组合理数字的均值是一个模糊的数字。解决方法：离散解码器（VQ-VAE, NVAE），或者仅将VAE用作编码器，并在潜在空间上堆叠扩散模型（Stable Diffusion就是这么做的）。
- **β太大或太早使用大β。** 参见后验坍缩。从β≈0.01开始逐渐增加。
- **潜在维度过小。** 16维适用于MNIST，256维适用于256²的ImageNet，2048维适用于1024²的ImageNet。Stable Diffusion的VAE将512×512×3压缩为64×64×4（空间面积下采样32倍，通道数增加4倍）。

## 使用它

2026年的VAE技术栈：

| 场景 | 选择 |
|------|------|
| 用于扩散模型的图像-潜在编码器 | Stable Diffusion VAE (`sd-vae-ft-ema`) 或 Flux VAE |
| 音频-潜在编码器 | Encodec (Meta), SoundStream, 或 DAC (Descript) |
| 视频潜在表示 | Sora的时空分块, Latte VAE, WAN VAE |
| 解纠缠表示学习 | β-VAE, FactorVAE, TCVAE |
| 离散潜在表示（用于Transformer建模） | VQ-VAE, RVQ (ResidualVQ) |
| 用于生成的连续潜在表示 | 普通VAE，然后在该潜在空间中条件化流/扩散模型 |

一个潜在扩散模型就是一个VAE，其编码器和解码器之间有一个扩散模型。VAE进行粗略压缩，扩散模型承担繁重的工作。视频（VAE + 视频扩散DiT）和音频（Encodec + MusicGen Transformer）遵循相同的模式。

## 部署它

保存 `outputs/skill-vae-trainer.md`。

该技能输入：数据集概况 + 潜在维度目标 + 下游用途（重建、采样或潜在扩散模型输入），并输出：架构选择（普通/β/VQ/RVQ）、β调度策略、潜在维度、解码器似然（高斯 vs 分类）以及评估计划（重建均方误差、每维KL散度、`q(z|x)` 与 `N(0, I)` 之间的Fréchet距离）。

## 练习

1. **简单。** 在 `code/main.py` 中将 `β` 改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终的重建均方误差和KL散度。对于你的合成数据，哪个β是帕累托最优的？
2. **中等。** 将高斯解码器似然替换为伯努利似然（交叉熵损失）。在二值化版本的相同合成数据上比较样本质量。
3. **困难。** 将 `code/main.py` 扩展为一个迷你VQ-VAE：用一个K=32个条目的码本中的最近邻查找替换连续的 `z`。比较重建均方误差，并报告使用了多少个码本条目（码本坍缩是真实存在的）。

## 关键术语

| 术语 | 人们如何说 | 实际含义 |
|------|------------|----------|
| 自编码器 | 编码-解码网络 | `x → z → x̂`，学习均方误差。非生成模型。 |
| VAE | 带采样器的自编码器 | 编码器输出一个分布，KL惩罚塑造编码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ recon - KL[q(z|x) \|\| p(z)]`；当 `q = p(z|x)` 时，界限是紧的。 |
| 重参数化 | `z = μ + σ·ε` | 将随机节点重写为确定性函数 + 纯噪声。使得通过采样进行反向传播成为可能。 |
| 先验 | `p(z)` | 潜在表示的目标分布，通常是 `N(0, I)`。 |
| 后验坍缩 | “KL项赢了” | 编码器忽略 `x`，输出先验；解码器必须凭空生成。 |
| β-VAE | 可调的KL权重 | `loss = recon + β·KL`。更高的β = 更解纠缠但更模糊。 |
| VQ-VAE | 离散潜在表示 | 用最近的码本向量替换连续的 `z`；使得Transformer建模成为可能。 |

## 生产说明：VAE是扩散推理服务器中最热的路径

在Stable Diffusion / Flux / SD3流水线中，每个请求会调用两次VAE——一次用于编码（如果进行图像转图像/修补），一次用于解码。在1024²分辨率下，解码过程通常是整个流水线中单次最大的激活内存峰值，因为它将 `128×128×16` 的潜在表示上采样回 `1024×1024×3`。这带来两个实际后果：

- **对解码进行切片或分块。** `diffusers` 暴露了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。分块会用一个小的接缝瑕疵，换取使用 `O(tile²)` 的内存而不是 `O(H·W)`。对于消费级GPU上1024²及以上的分辨率，这是必不可少的。
- **解码器使用bf16，最终的尺寸调整使用fp32数值计算。** SD 1.x的VAE是以fp32发布的，在1024²+分辨率下转换为fp16时*会静默地产生NaN*。SDXL提供了 `madebyollin/sdxl-vae-fp16-fix` —— 请始终首选fp16修复版本或使用bf16。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE论文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 解纠缠β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 最先进的图像VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE作为编码器。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音频VAE标准。