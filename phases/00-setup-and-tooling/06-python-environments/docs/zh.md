# Python 环境

> 依赖地狱真实存在。虚拟环境是解药。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段 0，第 01 课
**时间：** 约 30 分钟

## 学习目标

- 使用 `uv`、`venv` 或 `conda` 创建隔离的虚拟环境
- 编写一个包含可选依赖组的 `pyproject.toml`，并生成锁文件以确保可复现性
- 诊断并修复常见陷阱：全局安装、混用 pip/conda、CUDA 版本不匹配
- 为具有冲突依赖的项目实施分阶段的环境策略

## 问题所在

你为一个微调项目安装了 PyTorch 2.4。下周，另一个项目需要 PyTorch 2.1，因为它的 CUDA 构建被固定了。你进行了全局升级，第一个项目就挂了。你进行了全局降级，第二个项目也挂了。

这就是依赖地狱。在 AI/ML 工作中这很常见，原因是：

- PyTorch、JAX 和 TensorFlow 各自捆绑了它们自己的 CUDA 绑定
- 模型库固定了特定的框架版本
- 一个全局的 `pip install` 会覆盖之前存在的任何东西
- CUDA 11.8 构建无法与 CUDA 12.x 驱动一起工作（反之亦然）

解决方法：每个项目都应拥有自己的隔离环境和独立的包。

## 核心概念

```mermaid
graph TD
    subgraph without["Without virtual environments"]
        SP[System Python] --> T24["torch 2.4.0 (CUDA 12.4)\nProject A needs this"]
        SP --> T21["torch 2.1.0 (CUDA 11.8)\nProject B needs this"]
        SP --> CONFLICT["CONFLICT: only one\ntorch version can exist"]
    end

    subgraph with["With virtual environments"]
        PA["Project A (.venv/)"] --> PA1["torch 2.4.0 (CUDA 12.4)"]
        PA --> PA2["transformers 4.44"]
        PB["Project B (.venv/)"] --> PB1["torch 2.1.0 (CUDA 11.8)"]
        PB --> PB2["diffusers 0.28"]
    end
```

## 构建环境

### 选项 1：uv venv（推荐）

`uv` 是最快的 Python 包管理器（比 pip 快 10-100 倍）。它在一个工具中处理虚拟环境、Python 版本和依赖解析。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

cd your-project
uv venv
source .venv/bin/activate
```

安装包：

```bash
uv pip install torch numpy
```

一步创建包含 `pyproject.toml` 的项目：

```bash
uv init my-ai-project
cd my-ai-project
uv add torch numpy matplotlib
```

### 选项 2：venv（内置）

如果你无法安装 `uv`，Python 自带 `venv`：

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

pip install torch numpy
```

比 `uv` 慢，但在任何安装了 Python 的地方都能工作。

### 选项 3：conda（需要时使用）

Conda 管理非 Python 的依赖项，如 CUDA 工具包、cuDNN 和 C 库。在以下情况使用它：

- 你需要特定版本的 CUDA 工具包，而不想在系统范围内安装它
- 你在一个无法安装系统包的共享集群上
- 一个库的安装说明说“使用 conda”

```bash
# Install miniconda (not the full Anaconda)
curl -LsSf https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b

conda create -n myproject python=3.12
conda activate myproject

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

一个原则：如果你为一个环境使用了 conda，那么该环境中的所有包都应使用 conda 安装。在一个 conda 环境中混用 `pip install` 会导致依赖冲突，调试起来非常痛苦。

### 本课程策略：分阶段

你可以为整个课程创建一个环境。不要这样做。不同阶段需要不同（有时是冲突的）依赖项。

策略：

```
ai-engineering-from-scratch/
├── .venv/                    <-- shared lightweight env for phases 0-3
├── phases/
│   ├── 04-neural-networks/
│   │   └── .venv/            <-- PyTorch env
│   ├── 05-cnns/
│   │   └── .venv/            <-- same PyTorch env (symlink or shared)
│   ├── 08-transformers/
│   │   └── .venv/            <-- might need different transformer versions
│   └── 11-llm-apis/
│       └── .venv/            <-- API SDKs, no torch needed
```

`code/env_setup.sh` 中的脚本为本课程创建基础环境。

## pyproject.toml 基础

每个 Python 项目都应有一个 `pyproject.toml`。它用一个文件取代了 `setup.py`、`setup.cfg` 和 `requirements.txt`。

```toml
[project]
name = "ai-engineering-from-scratch"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "matplotlib>=3.8",
    "jupyter>=1.0",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
torch = ["torch>=2.3", "torchvision>=0.18"]
llm = ["anthropic>=0.39", "openai>=1.50"]
```

然后安装：

```bash
uv pip install -e ".[torch]"    # base + PyTorch
uv pip install -e ".[llm]"     # base + LLM SDKs
uv pip install -e ".[torch,llm]" # everything
```

## 锁文件

锁文件将每个依赖项（包括传递性依赖）固定到确切的版本。这保证了可复现性：任何从锁文件安装的人都会得到完全相同的包。

```bash
# uv generates uv.lock automatically when using uv add
uv add numpy

# pip-tools approach
uv pip compile pyproject.toml -o requirements.lock
uv pip install -r requirements.lock
```

将你的锁文件提交到 git。当有人克隆仓库时，他们从锁文件安装，就能获得相同的版本。

## 常见错误

### 1. 全局安装

```bash
pip install torch  # BAD: installs to system Python

source .venv/bin/activate
pip install torch  # GOOD: installs to virtual environment
```

检查你的包安装到了哪里：

```bash
which python       # should show .venv/bin/python, not /usr/bin/python
which pip           # should show .venv/bin/pip
```

### 2. 混用 pip 和 conda

```bash
conda create -n myenv python=3.12
conda activate myenv
conda install pytorch -c pytorch
pip install some-other-package   # BAD: can break conda's dependency tracking
conda install some-other-package # GOOD: let conda manage everything
```

如果你必须在 conda 中使用 pip（有些包只有 pip 版本），请先安装所有 conda 包，最后再安装 pip 包。

### 3. 忘记激活

```bash
python train.py           # uses system Python, missing packages
source .venv/bin/activate
python train.py           # uses project Python, packages found
```

你的 Shell 提示符应该会显示环境名称：

```
(.venv) $ python train.py
```

### 4. 将 .venv 提交到 git

```bash
echo ".venv/" >> .gitignore
```

虚拟环境大小在 200MB 到 2GB 之间。它们是本地的，在不同机器间不可移植。改为提交 `pyproject.toml` 和锁文件。

### 5. CUDA 版本不匹配

```bash
nvidia-smi                # shows driver CUDA version (e.g., 12.4)
python -c "import torch; print(torch.version.cuda)"  # shows PyTorch CUDA version

# These must be compatible.
# PyTorch CUDA version must be <= driver CUDA version.
```

## 使用它

运行设置脚本来创建你的课程环境：

```bash
bash phases/00-setup-and-tooling/06-python-environments/code/env_setup.sh
```

这会在仓库根目录创建一个 `.venv`，安装核心依赖项并进行验证。

## 练习

1. 运行 `env_setup.sh` 并验证所有检查通过
2. 创建第二个虚拟环境，在其中安装一个不同版本的 numpy，并确认两个环境是隔离的
3. 为一个同时需要 PyTorch 和 Anthropic SDK 的项目编写一个 `pyproject.toml`
4. 故意全局安装一个包（不激活虚拟环境），注意它安装到了哪里，然后卸载它

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 虚拟环境 | “一个 venv” | 一个包含 Python 解释器和包的隔离目录，与系统 Python 分离 |
| 锁文件 | “固定的依赖” | 一个列出每个包及其确切版本的文件，保证在不同机器上安装结果完全一致 |
| pyproject.toml | “新的 setup.py” | 标准的 Python 项目配置文件，取代了 setup.py/setup.cfg/requirements.txt |
| 传递性依赖 | “依赖的依赖” | 包 B 依赖 C；如果你安装了依赖 B 的 A，那么 C 就是 A 的传递性依赖 |
| CUDA 不匹配 | “我的 GPU 不工作了” | PyTorch 编译时使用的 CUDA 版本与你的 GPU 驱动支持的版本不同 |