# GPU 设置与云端

> 在 CPU 上进行训练足以学习，但真正的训练需要 GPU。

**类型：** 构建
**语言：** Python
**前置条件：** 第 0 阶段，第 01 课
**时间：** 约 45 分钟

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地 GPU 的可用性
- 配置 Google Colab 使用 T4 GPU 进行免费的云端实验
- 对比 CPU 与 GPU 上的矩阵乘法性能并测量加速比
- 使用 fp16 经验法则估算 VRAM 能容纳的最大模型

## 问题所在

阶段 1-3 中的大多数课程在 CPU 上运行良好。但一旦你开始训练 CNN、Transformer 或 LLM（阶段 4 及之后），就需要 GPU 加速。在 CPU 上需要 8 小时的训练运行，在 GPU 上只需 10 分钟。

你有三个选择：本地 GPU、云端 GPU 或 Google Colab（免费）。

## 核心概念

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## 动手搭建

### 选项 1：本地 NVIDIA GPU

检查你是否拥有：

```bash
nvidia-smi
```

安装带 CUDA 的 PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 选项 2：Google Colab

1. 前往 [colab.research.google.com](https://colab.research.google.com)
2. 运行时 > 更改运行时类型 > T4 GPU
3. 运行 `!nvidia-smi` 以验证

可将本课程的 Jupyter 笔记本直接上传至 Colab。

### 选项 3：云端 GPU

适用于 Lambda Labs、RunPod 或 Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？没关系。

大多数课程可在 CPU 上运行。需要 GPU 的课程会明确说明并提供 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 动手搭建：GPU 与 CPU 基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上述基准测试，对比 CPU 与 GPU 的耗时
2. 若没有 GPU，请在 Google Colab 上运行并进行对比
3. 检查你的 GPU 显存大小，并估算能容纳的最大模型（经验法则：fp16 每个参数 2 字节）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| CUDA | “GPU 编程” | NVIDIA 的并行计算平台，允许你在 GPU 上运行代码 |
| VRAM | “GPU 内存” | GPU 上的视频内存，与系统内存分离。限制模型大小。 |
| fp16 | “半精度” | 16 位浮点数，使用 fp32 一半的内存，精度损失极小 |
| Tensor Core | “快速矩阵硬件” | 用于矩阵乘法的专用 GPU 核心，比普通核心快 4-8 倍 |