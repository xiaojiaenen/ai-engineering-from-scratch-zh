# Activation Functions

- No explanation or preamble

Let me translate:

"Without nonlinearity, your 100-layer network is a fancy matrix multiply. Activations are the gates that let neural networks think in curves."

Translation:
"没有非线性，你的100层网络只是一个花哨的矩阵乘法。激活函数是让神经网络能够以曲线方式思考的闸门。"

Technical terms like "nonlinearity", "matrix multiply", "Activations", "neural networks" - the rules say do NOT translate technical terms. Hmm, but these are common terms. Let me think about what to keep.

Actually "matrix multiply" is a technical term - could keep as 矩阵乘法 or keep the concept. "Activations" refers to activation functions - 激活函数. "neural networks" is 神经网络.

Let me translate this properly while keeping technical terms where appropriate. Since it's natural language, I'll translate the whole thing. Technical terms in Chinese are standard.

"Without nonlinearity" = 没有非线性
"your 100-layer network is a fancy matrix multiply" = 你的100层网络只是一个花哨的矩阵乘法
"Activations are the gates that let neural networks think in curves" = 激活函数是让神经网络能够以曲线方式思考的闸门

I'll wrap it.


<<<START>>>
没有非线性，你的100层网络只是一次花哨的矩阵乘法。激活函数是让神经网络能够以曲线方式进行思考的闸门。
<<<

**Type:** Build
**Languages:** Python
**Prerequisites:** Lesson 03.03 (Backpropagation)
**Time:** ~75 minutes

## Learning Objectives

- 从零实现 sigmoid、tanh、ReLU、Leaky ReLU、GELU、Swish 和 softmax，并实现它们的导数
- 通过 10+ 层测量不同激活函数的激活幅度，诊断梯度消失问题
- 检测 ReLU 网络中的死神经元，并解释 GELU 如何避免这种失效模式
- 为给定架构（transformer、CNN、RNN、输出层）选择正确的激活函数
<<<

## 问题
<<<

Let me look at the content. There's some special character between "Stack" and "two" - it looks like there's a zero-width character (⁣) before "Stack". Let me preserve that.

Let me translate:
- "Stack two linear transformations" → 堆叠两个线性变换
- "y = W2(W1x + b1) + b2" → keep as code/math
- "Expand it: y = W2W1x + W2b1 + b2." → 展开它：y = W2W1x + W2b1 + b2.
- "That's just y = Ax + c -- a single linear transformation." → 这只是 y = Ax + c -- 单个线性变换
- "No matter how many linear layers you stack, the result collapses to one matrix multiply." → 无论你堆叠多少层线性层，结果都会坍缩为一次矩阵乘法
- "Your 100-layer network has the same representational power as a single layer." → 你100层的网络与单层具有相同的表达能力

Let me preserve the markdown formatting. The original has code formatting on the equations. Let me keep those.


<<<START>>>
Stack两个线性变换：y = W2(W1x + b1) + b2。展开它：y = W2W1x + W2b1 + b2。这其实就是 y = Ax + c -- 单个线性变换。无论你堆叠多少层线性层，结果都会坍缩为一次矩阵乘法。你100层的网络与单层拥有相同的表达能力。
<<<

<<<START>>>
这不是一个理论上的奇闻异事。它意味着一个深度线性网络根本无法学习 XOR，无法对螺旋数据集进行分类，无法识别一张脸。没有激活函数，深度只是一种错觉。
<<<

Let me translate:

"Activation functions break the linearity. They warp the output of each layer through a nonlinear function, giving the network the ability to bend decision boundaries, approximate arbitrary functions, and actually learn. But pick the wrong activation and your gradients vanish to zero (sigmoid in deep networks), explode to infinity (unbounded activations without careful initialization), or your neurons die permanently (ReLU with large negative biases). The choice of activation function directly determines whether your network learns at all."

Technical terms to preserve: Activation functions, sigmoid, ReLU, gradients, neurons, decision boundaries, etc.

Let me translate:

"激活函数打破线性。它们通过非线性函数扭曲每一层的输出，使网络能够弯曲决策边界、近似任意函数，并真正进行学习。但如果选错了激活函数，你的梯度会消失到零（深度网络中的 sigmoid），爆炸到无穷（没有精心初始化的无界激活），或者你的神经元永久死亡（带有较大负偏差的 ReLU）。激活函数的选择直接决定了你的网络到底能否学习。"


<<<START>>>

激活函数打破线性。它们通过非线性函数扭曲每一层的输出，使网络能够弯曲决策边界、近似任意函数，并真正进行学习。但如果选错了激活函数，你的梯度会消失到零（深度网络中的 sigmoid），爆炸到无穷（没有精心初始化的无界激活），或者你的神经元永久死亡（带有较大负偏差的 ReLU）。激活函数的选择直接决定了你的网络到底能否学习。

<<<

## 概念
<<<

### 为什么非线性是必要的

Let me translate:

"Matrix multiplication is composable. Multiplying a vector by matrix A then matrix B is identical to multiplying by AB. This means stacking ten linear layers is mathematically equivalent to one linear layer with one big matrix. All those parameters, all that depth -- wasted. You need something to break the chain. That's what activation functions do."

Let me translate:

矩阵乘法是可组合的。用矩阵A乘以向量，再用矩阵B乘，等同于直接乘以AB。这意味着堆叠十个线性层在数学上等价于一个拥有一个大矩阵的线性层。所有这些参数，所有这些深度——都浪费了。你需要某些东西来打破这个链。这就是激活函数的作用。

Technical terms: "matrix multiplication", "linear layers", "activation functions" - these are technical terms that I should keep. Actually, the instruction says "Do NOT translate... technical terms." So terms like "矩阵乘法" (matrix multiplication) might be considered technical terms. But typically when translating to Chinese, we do translate these into their Chinese equivalents. Let me re-read the rule.

"Do NOT translate code, math, links, or technical terms."

Hmm, this is tricky. Technical terms like "matrix multiplication" - the English term itself. I think the rule is about not translating things like code snippets, math expressions, links, and proper technical terms that have specific meanings. But actually, standard practice in translation is to render these into Chinese. Let me think about what makes sense here.

Actually, I think the spirit of "don't translate technical terms" is more about things like API names, function names, specific library terms, code identifiers, etc. For general mathematical/ML concepts, translating to standard Chinese terminology is appropriate. Terms like "矩阵乘法" (matrix multiplication), "线性层" (linear layer), "激活函数" (activation function) are standard Chinese translations.

Let me go with natural Chinese translation using standard technical terminology.

"AB" is a math expression - keep as is.

Let me finalize.


矩阵乘法是可组合的。用矩阵 A 乘以向量，再用矩阵 B 乘，等同于直接乘以 AB。这意味着堆叠十个线性层在数学上等价于一个拥有一个大矩阵的线性层。所有这些参数，所有这些深度——都浪费了。你需要某些东西来打破这个链。这就是激活函数的作用。

<<<START>>>
矩阵乘法是可组合的。用矩阵 A 乘以向量，再用矩阵 B 乘，等同于直接乘以 AB。这意味着堆叠十个线性层在数学上等价于一个拥有一个大矩阵的线性层。所有这些参数，所有这些深度——都浪费了。你需要某些东西来打破这个链。这就是激活函数的作用。
<<<

Let me translate:

"Here is the proof. A linear layer computes f(x) = Wx + b. Stack two:"

Translation:
"这是证明。线性层计算 f(x) = Wx + b。堆叠两个："

The math f(x) = Wx + b should stay as-is. "linear layer" is a technical term but "线性层" is a common translation. Let me think - the rule says don't translate technical terms. But "linear layer" translates to "线性层" which is standard. I'll keep it translated since it's prose.

Let me provide the translation.


<<<START>>>
这是证明。线性层计算 f(x) = Wx + b。堆叠两个：
<<<

```
Layer 1: h = W1 * x + b1
Layer 2: y = W2 * h + b2
```

"Substitute:" - this is likely a command or label meaning to replace something. I'll translate as "替换:"


<<<START>>>
替换：
<<<

```
y = W2 * (W1 * x + b1) + b2
y = (W2 * W1) * x + (W2 * b1 + b2)
y = A * x + c
```

One layer. Insert a nonlinear activation g() between layers:

```
h = g(W1 * x + b1)
y = W2 * h + b2
```

现在这种代入就无法成立了。W2 * g(W1 * x + b1) + b2 无法化简为单个线性变换。网络可以表示非线性函数。每增加一个带有激活函数的层，都会增加表示能力。
<<<

>>>,,,,### Sigmoid<<<

The original activation function for neural networks.

```
sigmoid(x) = 1 / (1 + e^(-x))
```

输出范围：(0, 1)。平滑、可微，将任意实数映射为概率型值。
<<<

<<<START>>>
导数：
<<<

```
sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
```

The maximum value of this derivative is 0.25, occurring at x = 0. In backpropagation, gradients multiply through layers. Ten layers of sigmoid means the gradient gets multiplied by at most 0.25 ten times:

```
0.25^10 = 0.000000953674
```

Let me look at the text. There are no PROTECT tokens here. Let me translate.

"Less than one millionth of the original signal. This is the vanishing gradient problem. Gradients in early layers become so small that weights barely update. The network appears to learn -- loss decreases in later layers -- but the first layers are frozen. Deep sigmoid networks simply do not train."

Technical terms: "vanishing gradient problem" (vanishing gradient 消失梯度), "gradients" (梯度), "weights" (权重), "layers" (层), "loss" (损失), "sigmoid" (sigmoid 逻辑斯谛). These are technical terms I should handle carefully. The rule says don't translate technical terms, so I should keep terms like sigmoid, loss, gradient maybe. Let me think - "do not translate technical terms" - so things like "sigmoid", "loss" might be kept in English or translated as commonly done. In ML Chinese context, common terms: 梯度 (gradient), 权重 (weight), 层 (layer), 损失 (loss), 消失梯度问题 (vanishing gradient problem). 

Actually the instruction is to not translate technical terms - meaning keep them as they are. But some common technical terms have established Chinese translations. Let me be balanced - keep clearly technical/untranslatable terms like "sigmoid" in English, but translate descriptive text.

Let me translate:
"小于原始信号的百万分之一。这就是消失梯度问题。早期层的梯度变得如此之小，以至于权重几乎不更新。网络看起来在学习——后层的损失在下降——但最前几层被冻结了。深度 sigmoid 网络根本训练不起来。"

The "--" em-dashes should be preserved as line breaks/markdown. Let me keep them.


<<<START>>>
小于原始信号的百万分之一。这就是消失梯度问题。早期层的梯度变得如此之小，以至于权重几乎不更新。网络看起来在学习——后层的损失在下降——但最前几层被冻结了。深度 sigmoid 网络根本训练不起来。
<<<

Additional problem: sigmoid outputs are always positive (0 to 1), which means gradients on weights are always the same sign. This causes zig-zagging during gradient descent.

<<<START>>>
### Tanh
<<<

<<<START>>>sigmoid 的中心化版本。<<<

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

Let me translate:
- "Output range: (-1, 1)." - The output range part with math should be kept as is. "输出范围：(-1, 1)。"
- "Zero-centered, which eliminates the zig-zag problem." - "Zero-centered" is a technical term. Let me think about how to handle this. I could translate it as "以零为中心". "which eliminates the zig-zag problem" - "从而消除了锯齿问题" or "消除了锯齿状问题".

Let me produce a natural translation.

输出范围：(-1, 1)。以零为中心，从而消除了锯齿问题。


<<<START>>>输出范围：(-1, 1)。以零为中心，从而消除了锯齿问题。<<<

<<<START>>>
导数：
<<<

```
tanh'(x) = 1 - tanh(x)^2
```

最大导数为 1.0，在 x = 0 处取得——比 sigmoid 好四倍。但梯度消失问题仍然存在。对于很大的正输入或负输入，导数会趋近于零。十层网络依然会压制梯度，只是没那么激进。
<<<

### ReLU: The Breakthrough

Rectified Linear Unit. Popularized for deep learning by Nair and Hinton in 2010 (the function itself dates to Fukushima's 1969 work), it changed everything.

```
relu(x) = max(0, x)
```

输出范围：[0, infinity)。其导数平凡地简单：
<<<

```
relu'(x) = 1  if x > 0
            0  if x <= 0
```

No vanishing gradient for positive inputs. The gradient is exactly 1, passed straight through. This is why deep networks became trainable -- ReLU preserves gradient magnitude across layers.

Let me translate:

"But there is a failure mode: the dead neuron problem."
"但是存在一种失败模式：死神经元问题。"

"If a neuron's weighted input is always negative (due to a large negative bias or unfortunate weight initialization), its output is always zero, its gradient is always zero, and it never updates."
"如果一个神经元的加权输入总是负数（由于较大的负偏置或不理想的权重初始化），它的输出总是零，梯度总是零，而且从不更新。"

"It is permanently dead."
"它会永久死亡。"

"In practice, 10-40% of neurons in a ReLU network can die during training."
"在实践中，ReLU 网络中 10-40% 的神经元在训练过程中可能会死亡。"

Let me keep technical terms as-is where appropriate. "ReLU" is a technical term, keep it. "bias" = 偏置, "weight" = 权重, "gradient" = 梯度, "neuron" = 神经元. These are standard translations.

Let me write the final translation.


<<<START>>>
但是存在一种失败模式：死神经元问题。如果一个神经元的加权输入总是负数（由于较大的负偏置或不理想的权重初始化），它的输出总是零，梯度总是零，而且从不更新。它会永久死亡。在实践中，ReLU 网络中 10-40% 的神经元在训练过程中可能会死亡。
<<<

### Leaky ReLU

- No explanation or preamble

"Dead neurons" is a technical term in neural networks/ML. I should keep it as a technical term but translate appropriately. "死神经元" is the standard Chinese term for dead neurons.

Let me translate:
"The simplest fix for dead neurons." → "修复死神经元的最简单方法。"


<<<START>>>
修复死神经元的最简单方法。
<<<

```
leaky_relu(x) = x        if x > 0
                alpha * x if x <= 0
```

- Do not include any explanation or preamble

Let me translate:

"Where alpha is a small constant, typically 0.01. The negative side has a small slope instead of zero, so dead neurons still get a gradient signal and can recover."

Translation:
"其中 alpha 是一个小常数，通常为 0.01。负侧有一个小的斜率而不是零，因此死神经元仍然能获得梯度信号并可以恢复。"

Technical terms like "alpha", "gradient", "neurons" - the rule says don't translate technical terms. But alpha, gradient, neurons are English technical terms. Let me think - "gradient" is 梯度, "neurons" is 神经元. These are commonly translated in Chinese. The rule says don't translate "technical terms" but I think this refers to keeping code/math as is. Actually, let me reconsider - the instruction says "Do NOT translate code, math, links, or technical terms." 

Hmm, so technical terms should not be translated. But that seems odd for a translation task. Let me interpret this reasonably - I'll keep the formula/variable names as is (like "alpha", "0.01"), but translate the prose. Words like "gradient" (梯度) and "neurons" (神经元) are standard Chinese ML terms, I'll translate them as they're commonly done. Actually the rule says don't translate technical terms... 

Let me be pragmatic. The variable "alpha" stays as alpha. The number 0.01 stays. But gradient signal and dead neurons - these I'll translate to 梯度信号 and 神经元 since they're normal Chinese words. Actually to be safe, I could keep gradient as is. But "gradient" is really just 梯度 in Chinese ML context.

I'll translate naturally while keeping technical names. Let me write:

"其中 alpha 是一个小常数，通常为 0.01。负侧采用较小的斜率而非零斜率，因此死神经元仍能获得梯度信号，从而得以恢复。"


<<<START>>>
其中 alpha 是一个小常数，通常为 0.01。负侧采用较小的斜率而非零斜率，因此"死亡"神经元仍能获得梯度信号，从而得以恢复。
<<<

### GELU: The Modern Default

Gaussian Error Linear Unit. Introduced by Hendrycks and Gimpel in 2016. Default activation in BERT, GPT, and most modern transformers.

```
gelu(x) = x * Phi(x)
```

Where Phi(x) is the cumulative distribution function of the standard normal distribution. The approximation used in practice:

```
gelu(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

GELU is smooth everywhere, allows small negative values (unlike ReLU which hard-clips to zero), and has a probabilistic interpretation: it weights each input by how likely it is to be positive under a Gaussian distribution. This smooth gating outperforms ReLU in transformer architectures because it provides better gradient flow and avoids the dead neuron problem entirely.

<<<START>>>
### Swish / SiLU
<<<

Self-gated activation discovered by Ramachandran et al. in 2017 through automated search.

```
swish(x) = x * sigmoid(x)
```

Swish 在形式上是 x * sigmoid(x)。谷歌通过对激活函数空间进行自动化搜索发现了它——即用神经网络来设计神经网络的部分组件。
<<<

Like GELU, it is smooth, non-monotonic, and allows small negative values. The difference is subtle: Swish uses sigmoid for gating while GELU uses the Gaussian CDF. In practice, performance is nearly identical. Swish is used in EfficientNet and some vision models. GELU dominates in language models.

### Softmax：输出激活
<<<

The text: "Not used in hidden layers. Softmax converts a vector of raw scores (logits) into a probability distribution."

Technical terms like "Softmax", "logits", "hidden layers", "probability distribution" - these are technical terms. I should not translate technical terms. But "logits" is given in parentheses, which suggests it's a technical term.

Let me translate naturally while preserving technical terms:

Not used in hidden layers. - 不用于隐藏层。
Softmax converts a vector of raw scores (logits) into a probability distribution. - Softmax 将原始得分（logits）向量转换为概率分布。

Technical terms preserved: Softmax, logits, hidden layers (隐藏层 - this is a translation of a term... hmm). Actually "hidden layers" is a technical term. But translating it to 隐藏层 is standard. The rule says don't translate technical terms. But this is a common term. Let me think - "logits" is kept in parentheses as technical. "Softmax" kept. 

I'll translate "hidden layers" as 隐藏层 since it's the standard Chinese term, and keep Softmax and logits as-is.

Let me produce the translation.


<<<START>>>
不用于隐藏层。Softmax 将原始得分（logits）向量转换为概率分布。
<<<

```
softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)
```

The text:
"Every output is between 0 and 1. All outputs sum to 1. This makes it the standard final activation for multi-class classification. The largest logit gets the highest probability, but unlike argmax, softmax is differentiable and preserves information about relative confidence."

Technical terms like "logit", "argmax", "softmax", "differentiable", "activation" - I should not translate these as they're technical terms. Let me translate while keeping technical terms.

Let me translate:
"每个输出都在 0 到 1 之间。所有输出之和为 1。这使其成为多分类问题的标准最终激活函数。最大的 logit 获得最高的概率，但与 argmax 不同，softmax 是可微的，并保留了关于相对置信度的信息。"


<<<START>>>
每个输出都在 0 到 1 之间。所有输出之和为 1。这使其成为多分类问题的标准最终激活。最大的 logit 获得最高的概率，但与 argmax 不同，softmax 是可微的，并保留了关于相对置信度的信息。
<<<

### Comparison of Shapes

```mermaid
graph LR
    subgraph "Activation Functions"
        S["Sigmoid<br/>Range: (0,1)<br/>Saturates both ends"]
        T["Tanh<br/>Range: (-1,1)<br/>Zero-centered"]
        R["ReLU<br/>Range: [0,inf)<br/>Dead neurons"]
        G["GELU<br/>Range: ~(-0.17,inf)<br/>Smooth gating"]
    end
    S -->|"Vanishing gradient"| Problem["Deep networks<br/>don't train"]
    T -->|"Less severe but<br/>still vanishes"| Problem
    R -->|"Gradient = 1<br/>for x > 0"| Solution["Deep networks<br/>train fast"]
    G -->|"Smooth gradient<br/>everywhere"| Solution
```

### 梯度流比较

```mermaid
graph TD
    Input["Input Signal"] --> L1["Layer 1"]
    L1 --> L5["Layer 5"]
    L5 --> L10["Layer 10"]
    L10 --> Output["Output"]

    subgraph "Gradient at Layer 1"
        SigGrad["Sigmoid: ~0.000001"]
        TanhGrad["Tanh: ~0.001"]
        ReluGrad["ReLU: ~1.0"]
        GeluGrad["GELU: ~0.8"]
    end
```

### 何时使用哪种激活

```mermaid
flowchart TD
    Start["What are you building?"] --> Hidden{"Hidden layers<br/>or output?"}

    Hidden -->|"Hidden layers"| Arch{"Architecture?"}
    Hidden -->|"Output layer"| Task{"Task type?"}

    Arch -->|"Transformer / NLP"| GELU["Use GELU"]
    Arch -->|"CNN / Vision"| ReLU["Use ReLU or Swish"]
    Arch -->|"RNN / LSTM"| Tanh["Use Tanh"]
    Arch -->|"Simple MLP"| ReLU2["Use ReLU"]

    Task -->|"Binary classification"| Sigmoid["Use Sigmoid"]
    Task -->|"Multi-class classification"| Softmax["Use Softmax"]
    Task -->|"Regression"| Linear["Use Linear (no activation)"]
```

```figure
softmax-temperature
```

## Build It

### Step 1: Implement All Activation Functions with Derivatives

- No explanation or preamble

The text is:
"Each function takes a single float and returns a float. Each derivative function takes the same input and returns the gradient."

Let me translate this naturally.


<<<START>>>
每个函数接收单个 float，并返回一个 float。每个导数函数接收相同的输入，并返回梯度。
<<<

```python
import math

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_act(x):
    return math.tanh(x)

def tanh_derivative(x):
    t = math.tanh(x)
    return 1 - t * t

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x, alpha=0.01):
    return 1.0 if x > 0 else alpha

def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def gelu_derivative(x):
    phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return phi + x * pdf

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def softmax(xs):
    max_x = max(xs)
    exps = [math.exp(x - max_x) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]
```

### Step 2: Visualize Where Gradients Die

在从 -5 到 5 的 100 个均匀分布的点上计算梯度。打印一个文本直方图，显示每个激活的梯度接近于零的位置。
<<<

```python
def gradient_scan(name, derivative_fn, start=-5, end=5, n=100):
    step = (end - start) / n
    near_zero = 0
    healthy = 0
    for i in range(n):
        x = start + i * step
        g = derivative_fn(x)
        if abs(g) < 0.01:
            near_zero += 1
        else:
            healthy += 1
    pct_dead = near_zero / n * 100
    print(f"{name:15s}: {healthy:3d} healthy, {near_zero:3d} near-zero ({pct_dead:.0f}% dead zone)")

gradient_scan("Sigmoid", sigmoid_derivative)
gradient_scan("Tanh", tanh_derivative)
gradient_scan("ReLU", relu_derivative)
gradient_scan("Leaky ReLU", leaky_relu_derivative)
gradient_scan("GELU", gelu_derivative)
gradient_scan("Swish", swish_derivative)
```

### Step 3: Vanishing Gradient Experiment

Forward-pass a signal through N layers using sigmoid vs ReLU. Measure how the activation magnitude changes.

```python
import random

def vanishing_gradient_experiment(activation_fn, name, n_layers=10, n_inputs=5):
    random.seed(42)
    values = [random.gauss(0, 1) for _ in range(n_inputs)]

    print(f"\n{name} through {n_layers} layers:")
    for layer in range(n_layers):
        weights = [random.gauss(0, 1) for _ in range(n_inputs)]
        z = sum(w * v for w, v in zip(weights, values))
        activated = activation_fn(z)
        magnitude = abs(activated)
        bar = "#" * int(magnitude * 20)
        print(f"  Layer {layer+1:2d}: magnitude = {magnitude:.6f} {bar}")
        values = [activated] * n_inputs

vanishing_gradient_experiment(sigmoid, "Sigmoid")
vanishing_gradient_experiment(relu, "ReLU")
vanishing_gradient_experiment(gelu, "GELU")
```

### Step 4: Dead Neuron Detector

Create a ReLU network, pass random inputs through it, count how many neurons never fire.

```python
def dead_neuron_detector(n_inputs=5, hidden_size=20, n_samples=1000):
    random.seed(0)
    weights = [[random.gauss(0, 1) for _ in range(n_inputs)] for _ in range(hidden_size)]
    biases = [random.gauss(0, 1) for _ in range(hidden_size)]

    fire_counts = [0] * hidden_size

    for _ in range(n_samples):
        inputs = [random.gauss(0, 1) for _ in range(n_inputs)]
        for neuron_idx in range(hidden_size):
            z = sum(w * x for w, x in zip(weights[neuron_idx], inputs)) + biases[neuron_idx]
            if relu(z) > 0:
                fire_counts[neuron_idx] += 1

    dead = sum(1 for c in fire_counts if c == 0)
    rarely_fire = sum(1 for c in fire_counts if 0 < c < n_samples * 0.05)
    healthy = hidden_size - dead - rarely_fire

    print(f"\nDead Neuron Report ({hidden_size} neurons, {n_samples} samples):")
    print(f"  Dead (never fired):     {dead}")
    print(f"  Barely alive (<5%):     {rarely_fire}")
    print(f"  Healthy:                {healthy}")
    print(f"  Dead neuron rate:       {dead/hidden_size*100:.1f}%")

    for i, c in enumerate(fire_counts):
        status = "DEAD" if c == 0 else "WEAK" if c < n_samples * 0.05 else "OK"
        bar = "#" * (c * 40 // n_samples)
        print(f"  Neuron {i:2d}: {c:4d}/{n_samples} fires [{status:4s}] {bar}")

dead_neuron_detector()
```

### Step 5: Training Comparison -- Sigmoid vs ReLU vs GELU

Let me translate this.

"在圆形数据集上训练同一个两层网络（圆内的点 = 类别 1，圆外的点 = 类别 0），使用三种不同的激活函数。比较收敛速度。"

Technical terms like "network", "activations", "convergence speed" - "activations" here refers to activation functions. Let me keep it accurate. "activation" is a technical term but I can translate "activations" as "激活函数" which is the standard term. Actually the rule says don't translate technical terms, but activation functions are commonly translated in Chinese as 激活函数. I'll translate it naturally.

Let me finalize.


<<<START>>>
在圆形数据集上训练同一个两层网络（圆内的点为类别 1，圆外的点为类别 0），分别使用三种不同的激活函数，并比较收敛速度。
<<<

```python
def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class ActivationNetwork:
    def __init__(self, activation_fn, activation_deriv, hidden_size=8, lr=0.1):
        random.seed(0)
        self.act = activation_fn
        self.act_d = activation_deriv
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
            self.h.append(self.act(z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        error = self.out - target
        d_out = error * self.out * (1 - self.out)

        for i in range(self.hidden_size):
            d_h = d_out * self.w2[i] * self.act_d(self.z1[i])
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def train(self, data, epochs=200):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += (pred - y) ** 2
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append(avg_loss)
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses


data = make_circle_data()

configs = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("ReLU", relu, relu_derivative),
    ("GELU", gelu, gelu_derivative),
]

results = {}
for name, act_fn, act_d_fn in configs:
    print(f"\n=== Training with {name} ===")
    net = ActivationNetwork(act_fn, act_d_fn, hidden_size=8, lr=0.1)
    losses = net.train(data, epochs=200)
    results[name] = losses

print("\n=== Final Loss Comparison ===")
for name, losses in results.items():
    print(f"  {name:10s}: start={losses[0]:.4f} -> end={losses[-1]:.4f} (improvement: {(1 - losses[-1]/losses[0])*100:.1f}%)")
```

## 使用它
<<<

Let me translate this.

PyTorch - keep as is (technical term/product name)
functional and module forms - these are technical terms, "functional" and "module" in PyTorch context. I should keep technical terms untranslated but translate the surrounding text.

"PyTorch provides all of these as both functional and module forms:"

Translation: "PyTorch 同时提供了这些的功能形式和模块形式："

Let me refine. "functional and module forms" - functional form = 函数形式, module form = 模块形式. But these are technical terms. Hmm, the rule says don't translate technical terms. But "functional and module forms" here refers to the functional API and nn.Module API in PyTorch. 

I think translating as "函数形式和模块形式" is acceptable and clearer. Actually, "functional" and "module" are PyTorch-specific terms. Let me keep them reasonable. I'll translate the general sentence structure.

"PyTorch 同时提供这些功能的函数形式和模块形式："


<<<START>>>
PyTorch 同时提供了这些的函数形式和模块形式：
<<<

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(4, 10)

relu_out = F.relu(x)
gelu_out = F.gelu(x)
sigmoid_out = torch.sigmoid(x)
swish_out = F.silu(x)

logits = torch.randn(4, 5)
probs = F.softmax(logits, dim=1)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.GELU(),
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Linear(32, 5),
)
```

- No explanation or preamble

Let me translate this text about activation functions.

"Hidden layers in a transformer: GELU. Hidden layers in a CNN: ReLU. Output layer for classification: softmax. Output layer for regression: none (linear). Output layer for probabilities: sigmoid. That's it. Start with these defaults. Change them only when you have evidence."

Technical terms to keep: transformer, GELU, CNN, ReLU, softmax, sigmoid - these are technical terms, keep them.

Let me translate:

Transformer中的隐藏层：GELU。CNN中的隐藏层：ReLU。分类的输出层：softmax。回归的输出层：无（线性）。概率输出层：sigmoid。就这么些。先使用这些默认值。只有在有证据时才更改。


<<<START>>>
Transformer中的隐藏层：GELU。CNN中的隐藏层：ReLU。分类的输出层：softmax。回归的输出层：无（线性）。概率输出层：sigmoid。就这么些。先使用这些默认值。只有在有证据时才更改。
<<<

RNNs and LSTMs use tanh for hidden state and sigmoid for gates, but if you're building from scratch today, you're probably not using RNNs. If neurons are dying in your ReLU network, switch to GELU. Don't reach for Leaky ReLU unless you have a specific reason -- GELU solves the dead neuron problem and gives better gradient flow.

## Ship It

Let me translate:
"This lesson produces:" → "本课将产出："
"a reusable prompt" → "一个可复用的提示词"
"that helps you pick the right activation function for any architecture" → "它帮助你为任何架构选择合适的激活函数"

Let me put it together while preserving the markdown structure.


<<<START>>>
本课将产出：
- `outputs/prompt-activation-selector.md` -- 一个可复用的提示词，帮助你为任何架构选择合适的激活函数
<<<

## 练习

<<<

Technical terms like "Parametric ReLU", "PReLU", "Leaky ReLU", "alpha" should be kept as technical terms. But wait, the instruction says don't translate technical terms. However, "PReLU", "Leaky ReLU" are proper names of activation functions. Let me think about how to handle these.

The rule says "Do NOT translate code, math, links, or technical terms." So I should keep technical terms. But these are activation function names. I'll keep PReLU, Leaky ReLU as is. "alpha" is a variable/math term, keep as is.

Let me translate:
"1. 实现参数化 ReLU（PReLU），其中负斜率 alpha 是一个可学习参数。在 circle 数据集上训练它，并与固定的 Leaky ReLU 进行比较。"

Actually, for technical terms I should keep them. Let me reconsider. "Parametric ReLU" - this is a technical term. PReLU is the abbreviation. I'll keep these. "Leaky ReLU" is a technical term.

Let me write the translation:

"1. 实现参数化 ReLU（PReLU），其中负斜率 alpha 为可学习参数。在 circle 数据集上训练该模型，并与固定 Leaky ReLU 进行比较。"

This looks good. Let me make sure I preserve the markdown (the "1." list item) and line breaks.

I'll wrap it between <<<START>>> and <<<

2. 使用 50 层而非 10 层运行消失梯度实验。绘制 sigmoid、tanh、ReLU 和 GELU 在各层的幅度。每个激活函数的信号在第几层会 effectively 降至零？
<<<

<<<START>>>
3. 实现 ELU（指数线性单元）：elu(x) = x（当 x > 0 时），alpha * (e^x - 1)（当 x <= 0 时）。在相同网络中，将其神经元死亡比率与 ReLU 进行比较。
<<<

4. 构建一个在训练过程中运行的“梯度健康监控器”：在每个 epoch，计算每一层的平均梯度幅值。当任意一层的梯度低于 0.001 或超过 100 时打印警告。
<<<

Let me translate this:

"5. Modify the training comparison to use the XOR dataset from Lesson 01 instead of circles. Which activation converges fastest on XOR? Why does this differ from the circle results?"

Translation:
"5. 修改训练比较，改用 Lesson 01 中的 XOR 数据集，而不是 circles。哪种激活函数在 XOR 上收敛最快？这与圆环结果有何不同？"

Let me keep technical terms like "XOR", "Lesson 01", "activation", "converges" appropriately. The instructions say don't translate technical terms. "Activation" is a technical term (激活函数). Let me think - "activation" in neural networks context is "激活函数" or "激活". This is a technical term. I could keep it as "activation" or translate. The rule says don't translate technical terms. But activation is a somewhat standard term. Let me keep it minimal.

Actually, let me reconsider. The rule is to not translate technical terms. So "activation" → maybe keep as "activation" or use "激活函数". In Chinese ML context, "激活" is commonly used. Let me translate it reasonably while keeping key terms.

Let me write:
"5. 修改训练对比，改用 Lesson 01 中的 XOR 数据集，而不是 circles。哪种 activation 在 XOR 上收敛最快？这与 circle 结果有何不同？"

Hmm, but "circles" and "circle" - are these technical terms? They refer to the circles dataset. Let me keep "circles" as is since it's a dataset name.

Let me refine:
"5. 修改训练对比，改用 Lesson 01 中的 XOR 数据集，而不是 circles。哪种 activation 在 XOR 上收敛最快？这与 circle 的结果有何不同？"

This seems good. Let me finalize.


<<<START>>>
5. 修改训练对比，改用 Lesson 01 中的 XOR 数据集，而不是 circles。哪种 activation 在 XOR 上收敛最快？这与 circle 的结果有何不同？
<<<

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Activation function | "The nonlinear part" | A function applied to each neuron's output that breaks linearity, enabling the network to learn nonlinear mappings |
| Vanishing gradient | "Gradients disappear in deep networks" | Gradients shrink exponentially through layers when the activation's derivative is less than 1, making early layers untrainable |
| Exploding gradient | "Gradients blow up" | Gradients grow exponentially through layers when the effective multiplier exceeds 1, causing unstable training |
| Dead neuron | "A neuron that stopped learning" | A ReLU neuron whose input is permanently negative, producing zero output and zero gradient |
| Sigmoid | "Squishes values to 0-1" | The logistic function 1/(1+e^-x), historically important but causes vanishing gradients in deep networks |
| ReLU | "Clips negatives to zero" | max(0, x) -- the activation that made deep learning practical by preserving gradient magnitude |
| GELU | "The transformer activation" | Gaussian Error Linear Unit, a smooth activation that weights inputs by their probability of being positive |
| Swish/SiLU | "Self-gated ReLU" | x * sigmoid(x), discovered through automated search, used in EfficientNet |
| Softmax | "Turns scores into probabilities" | Normalizes a vector of logits into a probability distribution where all values are in (0,1) and sum to 1 |
| Leaky ReLU | "ReLU that doesn't die" | max(alpha*x, x) where alpha is small (0.01), preventing dead neurons by allowing small negative gradients |
| Saturation | "The flat part of sigmoid" | Regions where an activation's derivative approaches zero, blocking gradient flow |
| Logit | "The raw score before softmax" | The unnormalized output of the final layer before applying softmax or sigmoid |

## Further Reading

- Nair & Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines" (2010) -- the paper that introduced ReLU and enabled training of deep networks
- Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (2016) -- introduced the activation function that became the default for transformers
- Ramachandran et al., "Searching for Activation Functions" (2017) -- used automated search to discover Swish, showing that activation design can be automated
- Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010) -- the paper that diagnosed vanishing/exploding gradients and proposed Xavier initialization
- Goodfellow, Bengio, Courville, "Deep Learning" Chapter 6.3 (https://www.deeplearningbook.org/) -- rigorous treatment of hidden units and activation functions
