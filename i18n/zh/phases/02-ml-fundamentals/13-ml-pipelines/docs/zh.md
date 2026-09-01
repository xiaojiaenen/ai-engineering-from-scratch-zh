# ML 管道

> 模型不是产品，管道才是。管道涵盖从原始数据到部署预测的每一个环节，且每一步都必须可复现。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 2，第 12 课（超参数调优）
**时间：** ~120 分钟

## 学习目标

- 从零构建 ML 管道，将缺失值填充、缩放、编码和模型训练串联为单个可复现的对象
- 识别数据泄露场景，并解释管道如何通过仅对训练数据拟合变换器来防止泄露
- 构建 ColumnTransformer，对数值型和分类型特征应用不同的预处理
- 实现管道序列化，并演示同一已拟合管道在训练和生产中产生相同结果

## 问题所在

你有一个 notebook，它加载数据、用中位数填充缺失值、缩放特征、训练模型并打印准确率。它能工作，你也部署了。

一个月后，有人重新训练模型却得到了不同的结果。中位数是在包含测试数据的完整数据集上计算的（数据泄露）。缩放参数未保存，导致推理时使用了不同的统计量。特征工程代码在训练和服务之间被复制粘贴，两份代码逐渐产生分歧。一个分类型列在生产环境中出现了编码器从未见过的新值。

这些并非假设，它们是 ML 系统在生产环境中最常见的失败原因。管道通过将所有变换步骤打包成单个有序、可复现的对象来解决这些问题。

## 核心概念

### 什么是管道

管道是数据变换的有序序列，后接一个模型。每个步骤将前一步的输出作为输入。整个管道仅在训练数据上拟合一次。在推理时，同一已拟合管道对新数据进行变换并产生预测。

```mermaid
flowchart LR
    A[原始数据] --> B[填充缺失值]
    B --> C[缩放数值特征]
    C --> D[编码分类型特征]
    D --> E[训练模型]
    E --> F[预测]
```

管道保证：
- 变换仅在训练数据上拟合（无泄露）
- 推理时应用相同的变换
- 整个对象可序列化并作为单个工件部署
- 交叉验证按折应用管道，防止细微的泄露

### 数据泄露：无声的杀手

数据泄露发生在测试集或未来数据的信息污染了训练过程时。管道可以防止最常见的泄露形式。

**有泄露的（错误）做法：**
```python
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test = X_scaled[:800], X_scaled[800:]
y_train, y_test = y[:800], y[800:]
```

缩放器看到了测试数据。均值和标准差包含了测试样本。这会虚高准确率估计。

**正确的做法：**
```python
X_train, X_test = X[:800], X[800:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

使用管道时，你无需思考这些细节。管道会自动处理。

### sklearn Pipeline

sklearn 的 `Pipeline` 将变换器和估计器串联起来。它提供 `.fit()`、`.predict()` 和 `.score()` 方法，按顺序应用所有步骤。

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

当你调用 `pipe.fit(X_train, y_train)` 时：
1. 缩放器对 X_train 调用 `fit_transform`
2. 模型对缩放后的 X_train 调用 `fit`

当你调用 `pipe.predict(X_test)` 时：
1. 缩放器对 X_test 调用 `transform`（而非 fit_transform）
2. 模型对缩放后的 X_test 调用 `predict`

缩放器在拟合过程中从未看到测试数据。这正是关键所在。

### ColumnTransformer：针对不同列的不同管道

真实数据集同时包含需要不同预处理的数值型和分类型列。`ColumnTransformer` 可以处理这种情况。

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["age", "income", "score"]),
    ("cat", categorical_pipe, ["city", "gender", "plan"]),
])

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", GradientBoostingClassifier()),
])
```

OneHotEncoder 中的 `handle_unknown="ignore"` 对生产环境至关重要。当出现新类别（模型从未见过的城市）时，它会产生零向量而非崩溃。

### 实验追踪

管道使训练可复现，但你还需要追踪实验间的变化：使用了哪些超参数、哪个数据集版本、指标是什么、运行的是哪段代码。

**MLflow** 是最常见的开源解决方案：

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)

    pipe.fit(X_train, y_train)
    accuracy = pipe.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(pipe, "model")
```

每次运行都记录了参数、指标、工件和完整模型。你可以比较不同运行、复现任何实验并部署任何模型版本。

**Weights & Biases (wandb)** 提供相同功能，带有托管仪表板：

```python
import wandb

wandb.init(project="my-pipeline")
wandb.config.update({"max_depth": 5, "n_estimators": 100})

pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)

wandb.log({"accuracy": accuracy})
```

### 模型版本管理

实验追踪之后，你需要管理模型版本。哪个模型在生产环境？哪个在 staging？哪个是上周的？

MLflow 的 Model Registry 提供：
- **版本追踪：** 每个保存的模型都有版本号
- **阶段转换：** "Staging"、"Production"、"Archived"
- **审批流程：** 模型必须显式晋升到生产环境
- **回滚：** 瞬间切换回之前的版本

### 使用 DVC 进行数据版本控制

代码用 git 进行版本控制，数据也应版本控制，但 git 无法处理大文件。DVC（Data Version Control）解决了这个问题。

```
dvc init
dvc add data/training.csv
git add data/training.csv.dvc data/.gitignore
git commit -m "Track training data"
dvc push
```

DVC 将实际数据存储到远程存储（S3、GCS、Azure），并在 git 中保留一个小 `.dvc` 文件来记录哈希值。当你 checkout git commit 时，`dvc checkout` 会恢复使用的确切数据。

这意味着每个 git commit 都锁定了代码和数据。完全可复现。

### 可复现实验

一个可复现的实验需要四要素：

1. **固定随机种子：** 为 numpy、random 和框架（torch、sklearn）设置种子
2. **锁定依赖：** requirements.txt 或 poetry.lock 包含精确版本
3. **版本化数据：** DVC 或类似工具
4. **配置文件：** 所有超参数在配置文件中，而非硬编码

```python
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
```

### 从 Notebook 到生产管道

```mermaid
flowchart TD
    A[Jupyter Notebook] --> B[提取函数]
    B --> C[构建 Pipeline 对象]
    C --> D[添加超参数配置文件]
    D --> E[添加实验追踪]
    E --> F[添加数据验证]
    F --> G[添加测试]
    G --> H[打包部署]

    style A fill:#fdd,stroke:#333
    style H fill:#dfd,stroke:#333
```

典型演进过程：

1. **Notebook 探索：** 快速实验、可视化、特征构思
2. **提取函数：** 将预处理、特征工程、评估移入模块
3. **构建 Pipeline：** 将变换串联为 sklearn Pipeline 或自定义类
4. **配置管理：** 将所有超参数移入 YAML/JSON 配置
5. **实验追踪：** 添加 MLflow 或 wandb 日志
6. **数据验证：** 训练前检查模式、分布和缺失值模式
7. **测试：** 变换器的单元测试、完整管道的集成测试
8. **部署：** 序列化管道，包装为 API（FastAPI、Flask），容器化

### 常见管道错误

| 错误 | 为什么不好 | 修复 |
|-----|-----------|-----|
| 拆分前在完整数据上拟合 | 数据泄露 | 使用 Pipeline 配合 cross_val_score |
| 在管道外进行特征工程 | 训练与服务时的变换不同 | 将所有变换放入 Pipeline |
| 未处理未知类别 | 生产环境新值导致崩溃 | OneHotEncoder(handle_unknown="ignore") |
| 硬编码列名 | 模式变更时崩溃 | 使用配置文件中的列名列表 |
| 无数据验证 | 脏数据导致静默错误预测 | 预测前添加模式检查 |
| 训练/服务偏差 | 模型在生产中看到不同特征 | 训练和服务使用同一 Pipeline 对象 |

```figure
f3-pipeline-flow
```

## 动手构建

`code/pipeline.py` 中的代码从零构建完整的 ML 管道：

### 步骤 1：自定义变换器

```python
class CustomTransformer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

### 步骤 2：从零构建管道

```python
class PipelineFromScratch:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.fit_transform(X_current)
        name, model = self.steps[-1]
        model.fit(X_current, y)
        return self

    def predict(self, X):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)
        name, model = self.steps[-1]
        return model.predict(X_current)
```

### 步骤 3：配合管道的交叉验证

代码演示了如何使用管道进行交叉验证以防止数据泄露：缩放器在每个折的训练数据上独立拟合。

### 步骤 4：使用 sklearn 的完整生产管道

包含 `ColumnTransformer`、多个预处理路径和模型的完整管道，配合适当的交叉验证和实验日志进行训练。

## 交付成果

本课产出：
- `outputs/prompt-ml-pipeline.md` -- 构建和调试 ML 管道的技能指南
- `code/pipeline.py` -- 从零到 sklearn 的完整管道

## 练习

1. 构建一个处理包含 3 个数值列和 2 个分类型列的数据集的管道。使用 `ColumnTransformer` 对数值列应用中位数填充 + 缩放，对分类型列应用最常见值填充 + one-hot 编码。使用 5 折交叉验证训练。

2. 故意引入数据泄露：在拆分前对整个数据集拟合缩放器。比较交叉验证得分（有泄露）与管道交叉验证得分（干净）。差异有多大？

3. 使用 `joblib.dump` 序列化你的管道。在独立脚本中加载并运行预测。验证预测结果完全相同。

4. 向管道添加一个创建多项式特征（2 阶）的自定义变换器，针对最重要的两个数值列。它应该放在管道的什么位置？

5. 为管道设置 MLflow 追踪。运行 5 次不同超参数的实验。使用 MLflow UI（`mlflow ui`）比较运行结果并选择最佳模型。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|-----|-----------|---------|
| Pipeline | "变换链 + 模型" | 一组有序的已拟合变换器和模型，作为一个整体应用以防止泄露 |
| Data leakage | "测试信息泄露到训练中" | 使用训练集之外的信息来构建模型，虚高性能估计 |
| ColumnTransformer | "按列不同预处理" | 对不同列子集应用不同管道，合并结果 |
| Experiment tracking | "记录你的运行" | 为每次训练运行记录参数、指标、工件和代码版本 |
| MLflow | "追踪和部署模型" | 用于实验追踪、模型注册表和部署的开源平台 |
| DVC | "Git for data" | 用于大型数据文件的版本控制系统，将哈希存储在 git 中，数据存储在远程 |
| Model registry | "模型版本目录" | 使用阶段标签（staging、production、archived）追踪模型版本的系统 |
| Training/serving skew | "在 notebook 里能工作" | 训练和推理时数据处理方式的差异，导致静默错误 |
| Reproducibility | "相同代码，相同结果" | 使用相同代码、数据和配置获得相同结果的能力 |

## 延伸阅读

- [scikit-learn Pipeline 文档](https://scikit-learn.org/stable/modules/compose.html) -- 官方管道参考
- [MLflow 文档](https://mlflow.org/docs/latest/index.html) -- 实验追踪和模型注册表
- [DVC 文档](https://dvc.org/doc) -- 数据版本控制
- [Sculley et al., Hidden Technical Debt in Machine Learning Systems (2015)](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) -- ML 系统复杂性的开创性论文
- [Google ML 最佳实践：ML 规则](https://developers.google.com/machine-learning/guides/rules-of-ml) -- 实际生产 ML 建议
