"Loss Functions" is a technical term in machine learning. However, the instruction says not to translate technical terms. But "Loss Functions" as a heading - I should translate it. Actually, "Loss Functions" translates to "损失函数" which is the standard Chinese term.

Wait, the rule says "Do NOT translate code, math, links, or technical terms." But this is a heading for a markdown fragment. "Loss Functions" - should I translate it? The technical term rule usually applies to terms within the body that should remain in English for clarity. But for a heading, I think translating to "损失函数" is appropriate and standard.

Actually, let me reconsider. The instruction is to translate the Markdown fragment into Simplified Chinese. Technical terms like "Loss Functions" would typically be translated to their standard Chinese equivalent "损失函数" since that's the accepted term. I'll translate it as "损失函数".

The markdown is just a heading "# Loss Functions", so:
"# 损失函数"


<<<START>>>
# 损失函数
<<<

> Your network makes a prediction. The ground truth says otherwise. How wrong is it? That number is the loss. Pick the wrong loss function and your model optimizes for the wrong thing entirely.

**Type:** Build
**Languages:** Python
**Prerequisites:** Lesson 03.04 (Activation Functions)
**Time:** ~75 minutes

## Learning Objectives

Let me translate this into Simplified Chinese while keeping technical terms and things like "InfoNCE" untranslated.

The bullet points:
- Implement MSE, binary cross-entropy, categorical cross-entropy, and contrastive loss (InfoNCE) from scratch with their gradients
- Explain why MSE fails for classification by demonstrating the "predict 0.5 for everything" failure mode
- Apply label smoothing to cross-entropy and describe how it prevents overconfident predictions
- Choose the correct loss function for regression, binary classification, multi-class classification, and embedding learning tasks

Let me translate:

- 从零实现 MSE、二元交叉熵、类别交叉熵和对比损失（InfoNCE），并附上其梯度
- 通过演示"对所有输入都预测 0.5"的失效模式，解释为何 MSE 在分类任务上会失败
- 对交叉熵应用标签平滑，并说明它如何防止过度自信的预测
- 为回归、二元分类、多类别分类和嵌入学习任务选择正确的损失函数

Let me keep technical terms: MSE, cross-entropy, InfoNCE, gradient. The translation looks good.


<<<START>>>
- 从零实现 MSE、二元交叉熵、类别交叉熵和对比损失（InfoNCE），并实现其梯度
- 通过演示"对所有输入都预测 0.5"的失效模式，解释为何 MSE 在分类任务上会失败
- 对交叉熵应用标签平滑（label smoothing），并说明它如何防止过度自信的预测
- 针对回归、二元分类、多类别分类以及嵌入学习任务，选择正确的损失函数
<<<

## 问题
<<<

4. No explanation or preamble

Technical terms like "MSE", "classification", "loss" - these are technical terms. MSE is a technical term (Mean Squared Error), classification, loss. I should not translate these technical terms. Let me keep "MSE" and "loss" as they are technical terms. Actually, "classification" could be translated as 分类. Let me think - the rule says "Do NOT translate ... technical terms". So MSE, loss should be kept. 

Let me translate:
- "A model minimizing MSE on a classification problem will confidently predict 0.5 for everything." → "在一个分类问题上最小化 MSE 的模型会对所有事情自信地预测 0.5。"
- "It's minimizing loss." → "它在最小化损失。" (loss = 损失, this could be considered a technical term. Hmm. Actually "loss" as a technical term might be kept. But it's borderline. Let me keep it translated as 损失 since it's a common term, or... The rule says don't translate technical terms. Loss function / loss is a technical term. But translating it to 损失 is fine as that's the standard translation. Let me think about what counts as "technical term" - things like acronyms, code, etc. I think "loss" → "损失" is acceptable as a standard translation. Actually, to be safe, I could keep it. But natural translation would be 损失. Let me use 损失.)
- "It's also useless." → "它也毫无用处。"

Let me refine the translation to be natural Chinese.

"在分类问题上最小化 MSE 的模型会对一切都自信地预测 0.5。它确实在最小化损失，但也毫无用处。"

Let me preserve the sentence structure with the line breaks. The original has three sentences. Let me keep them as is.

Translation:
"在分类问题上最小化 MSE 的模型会对一切都自信地预测 0.5。它确实在最小化损失，但也毫无用处。"

Actually the original is three separate sentences. Let me keep them as three sentences:

"在分类问题上最小化 MSE 的模型会对一切都自信地预测 0.5。它确实在最小化损失。它也毫无用处。"

That's a bit awkward with "它也". Let me make it flow: "它确实在最小化损失，但也毫无用处。" This combines the last two which is more natural in Chinese. But I should preserve line breaks. The original doesn't seem to have line breaks within - it's a paragraph. Let me just translate it as a paragraph.

Final translation.


<<<START>>>在分类问题上最小化 MSE 的模型会对一切都自信地预测 0.5。它确实在最小化损失，但也毫无用处。<<<

Let me translate this paragraph.

"The loss function is the only thing your model actually optimizes."
损失函数是你的模型真正优化的唯一东西。

"Not accuracy. Not F1 score. Not whatever metric you report to your manager."
不是准确率。不是 F1 分数。不是你要汇报给经理的任何指标。

"The optimizer takes the gradient of the loss function and adjusts weights to make that number smaller."
优化器取损失函数的梯度，并调整权重，使那个数值变小。

"If the loss function doesn't capture what you care about, the model will find the mathematically cheapest way to satisfy it, and that way is almost never what you wanted."
如果损失函数没有捕捉到你关心的东西，模型会找到满足它的数学上最便宜（最省力）的方式，而那种方式几乎从来不是你所期望的。

Let me refine this for natural flow in Chinese.


损失函数是你的模型真正优化的唯一对象。不是准确率，不是 F1 分数，也不是你向上级汇报的任何指标。优化器取损失函数的梯度，并通过调整权重让这个数值变小。如果损失函数没有捕获你真正关心的东西，模型就会找到满足它的数学上最省事的途径，而那条途径几乎从来都不是你所期望的。

Here is a concrete example. You have a binary classification task. Two classes, 50/50 split. You use MSE as your loss. The model predicts 0.5 for every single input. The average MSE is 0.25, which is the minimum possible without actually learning anything. The model has zero discriminative ability but it has technically minimized your loss function. Switch to cross-entropy and the same model is forced to push predictions toward 0 or 1, because -log(0.5) = 0.693 is a terrible loss, while -log(0.99) = 0.01 rewards confident correct predictions. The choice of loss function is the difference between a model that learns and a model that games the metric.

It gets worse. In self-supervised learning, you don't even have labels. Contrastive loss defines the learning signal entirely: what counts as similar, what counts as different, and how hard the model should push them apart. Get contrastive loss wrong and your embeddings collapse to a single point -- every input maps to the same vector. Technically zero loss. Completely worthless.

## 概念
<<<

### 均方误差 (MSE)
<<<

- No explanation or preamble

Let me translate:

"The default for regression. Compute the squared difference between prediction and target, average over all samples."

Translation:
回归的默认值。计算预测值与目标值之间的平方差，并对所有样本取平均。


<<<START>>>
回归的默认值。计算预测值与目标值之间的平方差，并对所有样本取平均。
<<<

```
MSE = (1/n) * sum((y_pred - y_true)^2)
```

- No explanation or preamble

Let me translate:

"Why squaring matters: it penalizes large errors quadratically. An error of 2 costs 4x as much as an error of 1. An error of 10 costs 100x. This makes MSE sensitive to outliers -- a single wildly wrong prediction dominates the loss."

Translation:

"为什么平方很重要：它对大误差进行二次方惩罚。误差为 2 的成本是误差为 1 的 4 倍。误差为 10 的成本是 100 倍。这使得 MSE 对异常值敏感——一个严重错误的预测会主导损失。"

Let me refine this. The technical terms MSE, loss (损失) should be preserved appropriately. "MSE" is a technical term (Mean Square Error) - keep as is. "loss" - I could translate as 损失. "quadratically" - 二次方地 / 平方地. "outliers" - 异常值. "prediction" - 预测.

Let me write it cleanly:


<<<START>>>
为什么平方很重要：它对大误差以平方方式进行惩罚。误差为 2 的成本是误差为 1 的 4 倍。误差为 10 的成本则是 100 倍。这使得 MSE 对异常值非常敏感——一个严重偏离的预测就会主导整个损失。
<<<

Real numbers: if your model predicts housing prices and is off by $10,000 on most houses but off by $200,000 on one mansion, MSE will aggressively try to fix that one mansion, potentially hurting performance on the other 99 houses.

The gradient of MSE with respect to a prediction is:

```
dMSE/dy_pred = (2/n) * (y_pred - y_true)
```

- Do not include any explanation or preamble

Let me translate this text.

"Linear in the error. Bigger errors get bigger gradients. This is a feature for regression (large errors need large corrections) and a bug for classification (you want to penalize confident wrong answers exponentially, not linearly)."

Translation:

"对误差是线性的。误差越大，梯度越大。这对回归是一个特性（大误差需要大修正），但对分类来说是个 bug（你想要指数级地惩罚自信的错误答案，而不是线性地）."

Let me refine this.


<<<START>>>
对误差是线性的。误差越大，梯度越大。这对回归是一个特性（大误差需要大修正），但对分类来说却是个 bug（你希望指数级地惩罚那些有把握的错误答案，而不是线性地惩罚）。
<<<

### Cross-Entropy Loss

The loss function for classification. Rooted in information theory -- it measures the divergence between the predicted probability distribution and the true distribution.

**Binary Cross-Entropy (BCE):**

```
BCE = -(y * log(p) + (1 - y) * log(1 - p))
```

>>>其中 y 是真实标签（0 或 1），p 是预测概率。<<<

Why -log(p) works: when the true label is 1 and you predict p = 0.99, the loss is -log(0.99) = 0.01. When you predict p = 0.01, the loss is -log(0.01) = 4.6. That 460x difference is why cross-entropy works. It brutally punishes confident wrong predictions while barely penalizing confident correct ones.

The gradient tells the same story:

```
dBCE/dp = -(y/p) + (1-y)/(1-p)
```

>>>当 y = 1 且 p 接近零时，梯度为 -1/p，它会趋向负无穷。模型收到了一个巨大的信号，要修正它的错误。当 p 接近 1 时，梯度极小。已经正确了，无需修正。<<<

**Categorical Cross-Entropy:**

Technical terms: "multi-class classification" = 多分类, "one-hot encoded" = one-hot 编码. These are technical terms. Should I translate them? "multi-class classification" is a standard term "多分类". "one-hot encoded targets" - "one-hot" is technical, keep as is but translate "targets" as 目标/标签.

Let me translate: "适用于使用 one-hot 编码目标的多分类任务。" 

or "用于带 one-hot 编码目标的多分类。"

I'll keep "one-hot" as-is since it's a technical term.


<<<START>>>
适用于采用 one-hot 编码目标的多分类问题。
<<<

```
CCE = -sum(y_i * log(p_i))
```

<<<

### Why MSE Fails for Classification

```mermaid
graph TD
    subgraph "MSE on Classification"
        P1["Predict 0.5 for class 1<br/>MSE = 0.25"]
        P2["Predict 0.9 for class 1<br/>MSE = 0.01"]
        P3["Predict 0.1 for class 1<br/>MSE = 0.81"]
    end
    subgraph "Cross-Entropy on Classification"
        C1["Predict 0.5 for class 1<br/>CE = 0.693"]
        C2["Predict 0.9 for class 1<br/>CE = 0.105"]
        C3["Predict 0.1 for class 1<br/>CE = 2.303"]
    end
    P3 -->|"MSE gradient<br/>flattens near<br/>saturation"| Slow["Slow correction"]
    C3 -->|"CE gradient<br/>explodes near<br/>wrong answer"| Fast["Fast correction"]
```

- No explanation or preamble

Technical terms to not translate: MSE, gradients, sigmoid, cross-entropy, -log, predictions, saturation. These are technical terms/math/code.

Let me translate naturally while preserving the technical terms.

"MSE gradients flatten when predictions are near 0 or 1 (due to sigmoid saturation). Cross-entropy gradients compensate for this -- the -log cancels the sigmoid's flat regions, giving strong gradients exactly where they are needed most."

Translation:
MSE 梯度在预测值接近 0 或 1 时会趋于平缓（由于 sigmoid 饱和）。交叉熵梯度可以弥补这一点——-log 抵消了 sigmoid 的平缓区域，从而在最需要的地方给出强梯度。

Let me refine: "flatten" -> 趋于平缓. "cancels" -> 抵消. "giving strong gradients exactly where they are needed most" -> 在最需要的地方给出强梯度.

This looks good.


MSE 梯度在预测值接近 0 或 1 时会趋于平缓（由于 sigmoid 饱和）。交叉熵梯度可以弥补这一点——-log 抵消了 sigmoid 的平缓区域，从而在最需要的地方提供强梯度。

<<<START>>>
MSE 梯度在预测值接近 0 或 1 时会趋于平缓（由于 sigmoid 饱和）。交叉熵梯度可以弥补这一点——-log 抵消了 sigmoid 的平缓区域，从而在最需要的地方提供强梯度。
<<<

### Label Smoothing

Let me translate:

"Standard one-hot labels say 'this is 100% class 3 and 0% everything else.' That's a strong claim. Label smoothing softens it:"

Translation:
标准的 one-hot 标签会说"这是 100% 的 3 类，其余都是 0%。"这是一个很强的论断。Label smoothing 让它变得温和一些：

Let me think about technical terms: "one-hot labels" - this is a technical term, should I translate? The rule says don't translate technical terms. "one-hot" is a technical term. "class 3" - class is technical. "Label smoothing" is a technical term.

Let me keep technical terms. Let me refine:

标准的 one-hot 标签表示"这是 100% 的 3 类、其余均为 0%。"这是一个很有力的断言。Label smoothing 使这个断言变得温和一些：


<<<START>>>
标准的 one-hot 标签表示"这是 100% 的 3 类、其余均为 0%。"这是一个很有力的断言。Label smoothing 让这个断言变得温和一些：
<<<

```
smooth_label = (1 - alpha) * one_hot + alpha / num_classes
```

With alpha = 0.1 and 10 classes: instead of [0, 0, 1, 0, ...], the target becomes [0.01, 0.01, 0.91, 0.01, ...]. The model targets 0.91 instead of 1.0.

Why this works: a model trying to output exactly 1.0 through a softmax needs to push logits to infinity. This causes overconfidence, hurts generalization, and makes the model brittle to distribution shift. Label smoothing caps the target at 0.9 (with alpha=0.1), keeping logits in a reasonable range. GPT and most modern models use label smoothing or its equivalent.

<<<START>>>
### 对比损失
<<<

Let me translate this into Simplified Chinese:
"No labels. No classes. Just pairs of inputs and the question: are these similar or different?"

Translation:
"无标签。无类别。只有成对的输入和这个问题：它们相似还是不同？"


<<<START>>>无标签。无类别。只有成对的输入和这个问题：它们相似还是不同？<<<

**SimCLR-style contrastive loss (NT-Xent / InfoNCE):**

Take one image. Create two augmented views of it (crop, rotate, color jitter). These are the "positive pair" -- they should have similar embeddings. Every other image in the batch forms a "negative pair" -- they should have different embeddings.

```
L = -log(exp(sim(z_i, z_j) / tau) / sum(exp(sim(z_i, z_k) / tau)))
```

>>>其中 sim() 是余弦相似度，z_i 和 z_j 构成正样本对，求和遍历所有负样本，tau（温度）控制分布的锐度。较低的温度 = 更难的负样本 = 更激进的分离。<<<

Real numbers: batch size 256 means 255 negatives per positive pair. Temperature tau = 0.07 (SimCLR default). The loss looks like a softmax over similarities -- it wants the positive pair's similarity to be highest among all 256 options.

**Triplet Loss:**

Takes three inputs: anchor, positive (same class), negative (different class).

```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

The margin (typically 0.2-1.0) enforces a minimum gap between positive and negative distances. If the negative is already far enough away, the loss is zero -- no gradient, no update. This makes training efficient but requires careful triplet mining (choosing hard negatives that are close to the anchor).

<<<START>>>
### Focal Loss
<<<

对于不平衡数据集。标准的 cross-entropy 对所有正确分类的示例一视同仁。focal loss 对简单示例进行降权：
<<<

```
FL = -alpha * (1 - p_t)^gamma * log(p_t)
```

<<<START>>>
其中 p_t 为真实类别的预测概率，gamma 控制聚焦程度。当 gamma = 0 时，这就是标准的交叉熵。当 gamma = 2（默认值）时：
<<<

<<<START>>>
- 简单样本（p_t = 0.9）：weight = (0.1)^2 = 0.01。基本被忽略。
- 困难样本（p_t = 0.1）：weight = (0.9)^2 = 0.81。完整的 gradient signal。
<<<

- No explanation or preamble

Let me translate the text:

"Focal loss was introduced by Lin et al. for object detection, where 99% of candidate regions are background (easy negatives). Without focal loss, the model drowns in easy background examples and never learns to detect objects. With it, the model focuses its capacity on the hard, ambiguous cases that matter."

Technical terms: "focal loss" - this is a technical term. Should I keep it or translate? The rule says "Do NOT translate ... technical terms". Focal loss is often translated as "焦点损失" but as a technical term it might be kept. However, "focal loss" is a specific concept. Let me think - the instruction says don't translate technical terms. But focal loss could be translated as 焦点损失 which is the common Chinese term. 

Actually, let me reconsider. The rule about not translating technical terms usually applies to things that don't have good Chinese equivalents or are code identifiers. "Focal loss" has a well-established Chinese translation "焦点损失". But to be safe with the "do not translate technical terms" rule... Hmm.

Actually, I think common ML terms like focal loss are usually translated as 焦点损失 in Chinese contexts. Let me translate it as 焦点损失. "et al." is Latin, keep as is. "object detection" = 目标检测. "candidate regions" = 候选区域. "background" = 背景. "easy negatives" = 易负样本. 

Let me write the translation:

"焦点损失由 Lin 等人最初用于目标检测，其中 99% 的候选区域是背景（易负样本）。如果没有焦点损失，模型会被大量简单的背景样本淹没，永远学不会检测物体。有了它，模型就能把能力集中在真正重要的困难、模糊的样本上。"

Let me refine. "introduced by Lin et al. for object detection" - focal loss was introduced by Lin et al. for (the task of) object detection.

"Lin et al." - keep Lin, et al. is fine.

Let me finalize.


焦点损失由 Lin 等人最初引入，用于目标检测任务，其中 99% 的候选区域都是背景（易负样本）。若不采用焦点损失，模型会被大量简单的背景样本所淹没，永远学不会检测物体。而有了它，模型就能将其能力真正集中在那些困难且模糊、却至关重要的样本上。

<<<START>>>
焦点损失由 Lin 等人最初引入，用于目标检测任务，其中 99% 的候选区域都是背景（易负样本）。若不采用焦点损失，模型会被大量简单的背景样本所淹没，永远学不会检测物体。而有了它，模型就能将其能力真正集中在那些困难且模糊、却至关重要的样本上。
<<<

### Loss Function Decision Tree

```mermaid
flowchart TD
    Start["What is your task?"] --> Reg{"Regression?"}
    Start --> Cls{"Classification?"}
    Start --> Emb{"Learning embeddings?"}

    Reg -->|"Yes"| Outliers{"Outlier sensitive?"}
    Outliers -->|"Yes, penalize outliers"| MSE["Use MSE"]
    Outliers -->|"No, robust to outliers"| MAE["Use MAE / Huber"]

    Cls -->|"Binary"| BCE["Use Binary CE"]
    Cls -->|"Multi-class"| CCE["Use Categorical CE"]
    Cls -->|"Imbalanced"| FL["Use Focal Loss"]
    CCE -->|"Overconfident?"| LS["Add Label Smoothing"]

    Emb -->|"Paired data"| CL["Use Contrastive Loss"]
    Emb -->|"Triplets available"| TL["Use Triplet Loss"]
    Emb -->|"Large batch self-supervised"| NCE["Use InfoNCE"]
```

### Loss Landscape

```mermaid
graph LR
    subgraph "Loss Surface Shape"
        MSE_S["MSE<br/>Smooth parabola<br/>Single minimum<br/>Easy to optimize"]
        CE_S["Cross-Entropy<br/>Steep near wrong answers<br/>Flat near correct answers<br/>Strong gradients where needed"]
        CL_S["Contrastive<br/>Many local minima<br/>Depends on batch composition<br/>Temperature controls sharpness"]
    end
    MSE_S -->|"Best for"| Reg2["Regression"]
    CE_S -->|"Best for"| Cls2["Classification"]
    CL_S -->|"Best for"| Emb2["Representation learning"]
```

```figure
cross-entropy-loss
```

## Build It

MSE is a technical term (缩写). I could translate it to 均方误差 which is the standard Chinese term. The rule says don't translate technical terms, but headings often need translation for readability. MSE is an abbreviation. Let me think - "MSE" could be kept or translated. Given it's a heading and MSE stands for Mean Squared Error, I'll translate it to 均方误差 which is the natural Chinese equivalent.

Actually, the rule says "Do NOT translate code, math, links, or technical terms." MSE is technically a technical term/abbreviation. But it's common to translate MSE to 均方误差 in Chinese text. Let me keep it as 均方误差 as that's the standard translation.

Let me finalize:
"### 步骤 1：均方误差及其梯度"


>>>START>>>
### 步骤 1：均方误差及其梯度
>>>

```python
def mse(predictions, targets):
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        total += (p - t) ** 2
    return total / n

def mse_gradient(predictions, targets):
    n = len(predictions)
    grads = []
    for p, t in zip(predictions, targets):
        grads.append(2.0 * (p - t) / n)
    return grads
```

### 步骤 2: Binary Cross-Entropy
<<<

The log(0) problem is real. If the model predicts exactly 0 for a positive example, log(0) = negative infinity. Clipping prevents this.

```python
import math

def binary_cross_entropy(predictions, targets, eps=1e-15):
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))
        total += -(t * math.log(p_clipped) + (1 - t) * math.log(1 - p_clipped))
    return total / n

def bce_gradient(predictions, targets, eps=1e-15):
    grads = []
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))
        grads.append(-(t / p_clipped) + (1 - t) / (1 - p_clipped))
    return grads
```

### Step 3: Categorical Cross-Entropy with Softmax

Softmax converts raw logits to probabilities. Then we compute the cross-entropy against one-hot targets.

```python
def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]

def categorical_cross_entropy(logits, target_index, eps=1e-15):
    probs = softmax(logits)
    p = max(eps, probs[target_index])
    return -math.log(p)

def cce_gradient(logits, target_index):
    probs = softmax(logits)
    grads = list(probs)
    grads[target_index] -= 1.0
    return grads
```

The text mentions technical terms like "softmax", "cross-entropy", "gradient" - these should probably be kept or translated appropriately. Since they're technical terms, I should consider. The rule says don't translate technical terms. "softmax" is a technical term that stays. "cross-entropy" is a technical term. Let me keep "softmax" as is, and translate "cross-entropy" carefully.

Actually, the rule says "Do NOT translate code, math, links, or technical terms." So I should keep technical terms. "softmax" is a technical term - keep it. "cross-entropy" - could keep as cross-entropy or use 交叉熵. Since it's a technical term, I'll keep it but this is a bit ambiguous. Let me keep "softmax" and translate "cross-entropy" as "交叉熵" since that's a well-established term... Actually the rule says don't translate technical terms. Hmm. Let me keep it safe and keep both technical terms in English where they're clearly technical.

Actually, "softmax" is universally kept in English in Chinese ML texts. "cross-entropy" is often 交叉熵 in Chinese. But the rule says don't translate technical terms. Let me keep "softmax" as is. For cross-entropy, I'll translate it as 交叉熵 since it's commonly... no, let me follow the rule strictly and keep technical terms. But this is hard.

Let me just do a natural translation while keeping clearly English terms like "softmax". I'll write "cross-entropy" - this is a technical term. Let me keep it as "交叉熵" as that's the standard Chinese term... 

Actually I think the safest interpretation: keep "softmax" (it's a named function), translate the rest naturally. Cross-entropy I'll render as 交叉熵 which is standard.

Let me write the translation:

"softmax + 交叉熵 的梯度可以完美简化：对于真实类别就是（预测概率 - 1），对于其他所有类别就是（预测概率）。这种优雅的简化并非巧合——这正是 softmax 和交叉熵被配对使用的原因。"


softmax + 交叉熵 的梯度可以极大地简化：对于真实类别，它就是（预测概率 − 1）；对于其他所有类别，它就是（预测概率）。这种优雅的简化并非偶然——这正是 softmax 与交叉熵被配对使用的原因。

<<<START>>>
softmax + 交叉熵 的梯度可以极大地简化：对于真实类别，它就是（预测概率 − 1）；对于其他所有类别，它就是（预测概率）。这种优雅的简化并非偶然——这正是 softmax 与交叉熵被配对使用的原因。
<<<

### Step 4: Label Smoothing

```python
def label_smoothed_cce(logits, target_index, num_classes, alpha=0.1, eps=1e-15):
    probs = softmax(logits)
    loss = 0.0
    for i in range(num_classes):
        if i == target_index:
            smooth_target = 1.0 - alpha + alpha / num_classes
        else:
            smooth_target = alpha / num_classes
        p = max(eps, probs[i])
        loss += -smooth_target * math.log(p)
    return loss
```

### Step 5: Contrastive Loss (Simplified InfoNCE)

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)

def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    sim_pos = cosine_similarity(anchor, positive) / temperature
    sim_negs = [cosine_similarity(anchor, neg) / temperature for neg in negatives]

    max_sim = max(sim_pos, max(sim_negs)) if sim_negs else sim_pos
    exp_pos = math.exp(sim_pos - max_sim)
    exp_negs = [math.exp(s - max_sim) for s in sim_negs]
    total_exp = exp_pos + sum(exp_negs)

    return -math.log(max(1e-15, exp_pos / total_exp))
```

- No explanation or preamble

Let me translate "Step 6: MSE vs Cross-Entropy on Classification"

- "Step 6" = "第 6 步"
- "MSE vs Cross-Entropy" - these are technical terms (MSE, Cross-Entropy), should NOT be translated
- "on Classification" = "在分类上" / "分类中的"

So: "### 第 6 步：分类中的 MSE 与交叉熵"

MSE is technical, keep as MSE. Cross-Entropy is technical term - but 交叉熵 is the standard Chinese translation for cross-entropy. Hmm, the rule says don't translate technical terms. But cross-entropy is a term that's commonly translated. Let me think about this.

The rule says "Do NOT translate code, math, links, or technical terms." Cross-entropy is a technical term. MSE is also a technical term. So I should keep MSE as MSE. For Cross-Entropy, it's ambiguous - it's a technical term. I'll keep it as Cross-Entropy to be safe, or translate it as 交叉熵.

Actually, since these are named technical terms, I think keeping them in their original form is safest. Let me keep MSE and Cross-Entropy as-is.

"### 第 6 步：MSE 与 Cross-Entropy 在分类上的比较"

This seems reasonable. Let me finalize.


### 第 6 步：MSE 与 Cross-Entropy 在分类上的比较

使用两种损失函数分别训练第 04 课中的同一个网络（圆形数据集）。观察交叉熵如何更快地收敛。
<<<

```python
import random

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class LossComparisonNetwork:
    def __init__(self, loss_type="bce", hidden_size=8, lr=0.1):
        random.seed(0)
        self.loss_type = loss_type
        self.lr = lr
        self.hidden_size = hidden_size

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        if self.loss_type == "mse":
            d_loss = 2.0 * (self.out - target)
        else:
            eps = 1e-15
            p = max(eps, min(1 - eps, self.out))
            d_loss = -(target / p) + (1 - target) / (1 - p)

        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def compute_loss(self, pred, target):
        if self.loss_type == "mse":
            return (pred - target) ** 2
        else:
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            return -(target * math.log(p) + (1 - target) * math.log(1 - p))

    def train(self, data, epochs=200):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += self.compute_loss(pred, y)
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses
```

## 使用它
<<<

Technical terms like "PyTorch" and "loss functions" - I should keep PyTorch as is. "loss functions" could be translated as "损失函数" which is a common technical term in Chinese. Actually the rule says don't translate technical terms, but "损失函数" is the standard Chinese translation for "loss functions". Let me think - the instruction says don't translate technical terms. But common Chinese ML terminology uses "损失函数". I'll translate it as it's a well-established term. Actually, to be safe with "loss functions" - this is a technical term. But PyTorch is definitely kept as PyTorch.

Let me translate:
"PyTorch provides all standard loss functions with numerical stability built in:"
= "PyTorch 提供了内置数值稳定性的所有标准损失函数："

Or more naturally: "PyTorch 提供了所有内置数值稳定性的标准损失函数："

I'll go with a natural translation.


<<<START>>>
PyTorch 提供了所有内置数值稳定性的标准损失函数：
<<<

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

predictions = torch.tensor([0.9, 0.1, 0.7], requires_grad=True)
targets = torch.tensor([1.0, 0.0, 1.0])

mse_loss = F.mse_loss(predictions, targets)
bce_loss = F.binary_cross_entropy(predictions, targets)

logits = torch.randn(4, 10)
labels = torch.tensor([3, 7, 1, 9])
ce_loss = F.cross_entropy(logits, labels)
ce_smooth = F.cross_entropy(logits, labels, label_smoothing=0.1)
```

Use `F.cross_entropy` (not `F.nll_loss` plus manual softmax). It combines log-softmax and negative log-likelihood in one numerically stable operation. Applying softmax separately then taking the log is less stable -- you lose precision in the subtraction of large exponentials.

For contrastive learning, most teams use custom implementations or libraries like `lightly` or `pytorch-metric-learning`. The core loop is always the same: compute pairwise similarities, create the softmax over positives and negatives, backpropagate.

## Ship It

This lesson produces:
- `outputs/prompt-loss-function-selector.md` -- a reusable prompt for choosing the right loss function
- `outputs/prompt-loss-debugger.md` -- a diagnostic prompt for when your loss curve looks wrong

## 练习

<<<

1. 实现 Huber 损失（平滑 L1 损失），它对较小误差采用 MSE，对较大误差采用 MAE。训练一个回归网络预测 y = sin(x)，在 5% 的训练目标被加入随机噪声（离群点）的情况下，比较 MSE 与 Huber 的表现，并对比最终测试误差。
<<<

- No explanation or preamble

The text is:
"2. Add focal loss to the binary classification training loop. Create an imbalanced dataset (90% class 0, 10% class 1). Compare standard BCE vs focal loss (gamma=2) on the minority class recall after 200 epochs."

Let me translate this while keeping technical terms like "focal loss", "BCE", "gamma", "recall", "epochs", "class 0", "class 1" intact. These are technical terms that should not be translated.

Translation:
"2. 在二分类训练循环中添加 focal loss。创建一个不平衡数据集（90% class 0，10% class 1）。在 200 个 epoch 后，对比标准 BCE 与 focal loss（gamma=2）在少数类别 recall 上的表现。"

Let me refine this to be natural Simplified Chinese while preserving technical terms.


<<<START>>>2. 在二分类训练循环中添加 focal loss。创建一个不平衡数据集（90% class 0，10% class 1）。在 200 个 epoch 后，对比标准 BCE 与 focal loss（gamma=2）在少数类别 recall 上的表现。<<<

3. 实现带有 semi-hard negative mining 的 triplet loss。为 5 个类别生成 2D embedding 数据。对于每个 anchor，找到仍然比 positive 更远的最难 negative（semi-hard）。与随机 triplet selection 对比收敛情况。
<<<

4. Run the MSE vs cross-entropy comparison but track gradient magnitudes at each layer during training. Plot the average gradient norm per epoch. Verify that cross-entropy produces larger gradients in early epochs when the model is most uncertain.

5. Implement KL divergence loss and verify that minimizing KL(true || predicted) gives the same gradients as cross-entropy when the true distribution is one-hot. Then try soft targets (like knowledge distillation) where the "true" distribution comes from a teacher model's softmax output.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Loss function | "How wrong the model is" | A differentiable function mapping predictions and targets to a scalar that the optimizer minimizes |
| MSE | "Average squared error" | Mean of squared differences between predictions and targets; penalizes large errors quadratically |
| Cross-entropy | "The classification loss" | Measures divergence between predicted probability distribution and true distribution using -log(p) |
| Binary cross-entropy | "BCE" | Cross-entropy for two classes: -(y*log(p) + (1-y)*log(1-p)) |
| Label smoothing | "Softening the targets" | Replacing hard 0/1 targets with soft values (e.g., 0.1/0.9) to prevent overconfidence and improve generalization |
| Contrastive loss | "Pull together, push apart" | A loss that learns representations by making similar pairs close and dissimilar pairs far in embedding space |
| InfoNCE | "The CLIP/SimCLR loss" | Normalized temperature-scaled cross-entropy over similarity scores; treats contrastive learning as classification |
| Focal loss | "The imbalanced data fix" | Cross-entropy weighted by (1-p_t)^gamma to down-weight easy examples and focus on hard ones |
| Triplet loss | "Anchor-positive-negative" | Pushes anchor closer to positive than negative by at least a margin in embedding space |
| Temperature | "Sharpness knob" | A scalar divisor on logits/similarities that controls how peaked the resulting distribution is; lower = sharper |

## Further Reading

- Lin et al., "Focal Loss for Dense Object Detection" (2017) -- introduced focal loss for handling extreme class imbalance in object detection (RetinaNet)
- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR, 2020) -- defined the modern contrastive learning pipeline with NT-Xent loss
- Szegedy et al., "Rethinking the Inception Architecture" (2016) -- introduced label smoothing as a regularization technique, now standard in most large models
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015) -- knowledge distillation using soft targets and KL divergence, foundational for model compression
