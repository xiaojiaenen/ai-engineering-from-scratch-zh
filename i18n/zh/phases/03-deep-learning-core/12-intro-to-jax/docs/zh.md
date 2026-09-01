<<<START>>>
# JAX 简介
<<<

> PyTorch 会修改张量。TensorFlow 构建图。JAX 编译纯函数。最后这一点改变了你对深度学习的思考方式。
<<<

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## Learning Objectives

- Write pure-function neural network code using JAX's functional API (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Explain the key design difference between PyTorch's eager mutation and JAX's functional compilation model
- Apply jit compilation and vmap vectorization to accelerate training loops compared to naive Python
- Train a simple network in JAX and contrast the explicit state management with PyTorch's object-oriented approach

## 问题
<<<

你知道如何在 PyTorch 中构建神经网络。你定义一个 `nn.Module`，调用 `.backward()`，然后让优化器执行一步。它能用。数百万人都在使用它。
<<<

But PyTorch has a constraint baked into its DNA: it traces operations eagerly, one at a time, in Python. Every `tensor + tensor` is a separate kernel launch. Every training step re-interprets the same Python code. This works fine until you need to train a 540-billion-parameter model across 2,048 TPUs. Then the overhead kills you.

Google DeepMind trains Gemini on JAX. Anthropic trained Claude on JAX. These are not small operations -- they are the largest neural network training runs on Earth. They chose JAX because it treats your training loop as a compilable program, not a sequence of Python calls.

Let me translate the text while keeping technical terms like JAX, NumPy, JIT, XLA, autograd etc. intact.

"JAX is NumPy with three superpowers: automatic differentiation, JIT compilation to XLA, and automatic vectorization."
→ "JAX 是拥有三项超能力的 NumPy：自动微分、向 XLA 的 JIT 编译，以及自动向量化。"

"You write a function that processes one example."
→ "你编写一个处理单个样本的函数。"

"JAX gives you a function that processes a batch, computes gradients, compiles to machine code, and runs across multiple devices."
→ "JAX 给你处理整个批次、计算梯度、编译成机器码，并在多个设备上运行的函数。"

"All without changing the original function."
→ "而无需改变原始函数。"

Let me keep the line breaks as they are - it's all one paragraph here.


<<<START>>>
JAX 是拥有三项超能力的 NumPy：自动微分、向 XLA 的 JIT 编译，以及自动向量化。你编写一个处理单个样本的函数。JAX 给你一个处理整个批次、计算梯度、编译成机器码并在多个设备上运行的函数。而无需改动原始函数。
<<<

## 概念
<<<

### The JAX Philosophy

JAX is a functional framework. No classes, no mutable state, no `.backward()` method. Instead:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

This is not a style preference. It is a compiler constraint. JIT compilation requires pure functions -- same inputs always produce same outputs, no side effects. That restriction is what makes 100x speedups possible.

### jax.numpy: The Familiar Surface

JAX reimplements the NumPy API on accelerators:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

- No explanation or preamble

Technical terms like GPU/TPU should not be translated. Let me translate:

"Same function names." → "相同的函数名。"
"Same broadcasting rules." → "相同的广播规则。"
"Same slicing semantics." → "相同的切片语义。"
"But the arrays live on GPU/TPU, and every operation is traceable by the compiler." → "但这些数组运行在 GPU/TPU 上，每一个操作都能被编译器追踪。"

Let me put it together.


<<<START>>>
相同的函数名。相同的广播规则。相同的切片语义。但这些数组运行在 GPU/TPU 上，每一个操作都能被编译器追踪。
<<<

One critical difference: JAX arrays are immutable. No `a[0] = 5`. Instead: `a = a.at[0].set(5)`. This feels awkward for a week, then it clicks -- immutability is what makes transformations like `grad`, `jit`, and `vmap` composable.

### jax.grad: Functional Autodiff

PyTorch attaches gradients to tensors (`.grad`). JAX attaches gradients to functions.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接收一个函数，并返回一个计算梯度的新函数。无需 `.backward()` 调用。张量上不存储计算图。梯度只是另一个你可以调用、组合或 JIT-编译的函数。
<<<

This composes arbitrarily:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Second derivatives. Third derivatives. Jacobians. Hessians. All by composing `grad`. PyTorch can do this too (`torch.autograd.functional.hessian`), but it is bolted on. In JAX, it is the foundation.

The constraint: `grad` only works on pure functions. No print statements inside (they run during tracing, not execution). No mutation of external state. No random number generation without explicit key management.

### jit: Compile to XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

On the first call, JAX traces the function -- it records which operations happen, without executing them. Then it hands that trace to XLA (Accelerated Linear Algebra), Google's compiler for TPUs and GPUs. XLA fuses operations, eliminates redundant memory copies, and generates optimized machine code.

>>>START<<<
后续调用完全跳过 Python。编译后的代码在加速器上以 C++ 速度运行。
>>>

When JIT helps:
- Training steps (same computation repeated thousands of times)
- Inference (same model, different inputs)
- Any function called more than once with similar-shaped inputs

When JIT hurts:
- Functions with Python control flow that depends on values (`if x > 0` where x is a traced array)
- One-shot computations (compilation overhead exceeds runtime)
- Debugging (tracing hides the actual execution)

控制流限制是真实存在的。`jax.lax.cond` 替换 `if/else`。`jax.lax.scan` 替换 `for` 循环。这些不是可选的 —— 它们是编译的代价。
<<<

Let me translate this. "vmap" is a technical term (likely a function name like PyTorch's vmap), so I should keep it as is. "Automatic Vectorization" is a technical concept.

The heading: "### vmap: Automatic Vectorization"

Translation: "### vmap：自动向量化"

Wait, the rules say "Do NOT translate code, math, links, or technical terms." "Automatic Vectorization" is a technical term/concept. Hmm, but it's part of the heading text. Let me think.

Actually "vmap" is a code/technical term to keep. "Automatic Vectorization" - vectorization is a technical term. But the heading is describing what vmap does. I think it's reasonable to translate "Automatic Vectorization" as "自动向量化" since it's describing the concept, but "向量化" and "自动" are fairly standard translations.

Let me keep it as a translation of the heading. The instruction is to translate English into Simplified Chinese, preserving technical terms. The technical terms are "vmap". "Automatic Vectorization" describes the feature - I'll translate it.

Let me produce: "### vmap：自动向量化"

I'll wrap in the tags.


### vmap：自动向量化

你编写一个函数来处理单个示例：
<<<

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` lifts it to process a batch:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 表示：不要对 `params`（共享）进行批处理，而对 `x` 的 axis 0 进行批处理。不要手动 `for` 循环。不要 reshaping。不要批维度 threading。JAX 会推断出批维度并对整个计算进行向量化。
<<<

这不是语法糖。`vmap` 生成融合的向量化代码，运行速度比 Python 循环快 10-100 倍。并且它可以与 `jit` 和 `grad` 组合：
<<<

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Technical terms like "PyTorch" should be preserved. "Per-example gradients" is a technical term but I can translate it as "每个样本的梯度" or keep it. Let me think about what counts as a technical term that shouldn't be translated.

"Per-example gradients" - this is a concept. I could translate it as "逐样本梯度" or "每样本梯度". I think translating the general phrase is fine since it's not a proper noun or code.

"PyTorch" is a technical term (library name) - keep as is.

"hacks" - could be translated as "技巧" or "hack手段".

Let me translate:
"逐样本梯度。一行。没有这些 hack 手段，在 PyTorch 里几乎不可能做到。"

This preserves the structure with the periods and line breaks. Let me finalize.


<<<START>>>
逐样本梯度。一行。没有这些 hack 手段，在 PyTorch 里几乎不可能实现。
<<<

### pmap：跨设备的数据并行
<<<

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` replicates the function across all available devices (GPUs/TPUs) and splits the batch. Inside the function, `jax.lax.pmean` and `jax.lax.psum` synchronize gradients across devices.

Google trains Gemini across thousands of TPU v5e chips using `pmap` (and its successor `shard_map`). The programming model: write the single-device version, wrap with `pmap`, done.

### Pytrees: The Universal Data Structure

JAX operates on "pytrees" -- nested combinations of lists, tuples, dicts, and arrays. Your model parameters are a pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Every JAX transformation -- `grad`, `jit`, `vmap` -- knows how to traverse pytrees. `jax.tree.map(f, tree)` applies `f` to every leaf. This is how optimizers update all parameters at once:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

No `.parameters()` method. No parameter registration. The tree structure is the model.

### Functional vs Object-Oriented

PyTorch 将状态存储在对象内部：
<<<

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用具有显式状态的纯函数：
<<<

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数是传入的。不会存储任何东西。也不会修改任何东西。这让每个函数都变得可测试、可组合、可编译。这也意味着你要自己管理这些参数——或者像 Flax 或 Equinox 那样使用现成的库。

<<<

### The JAX Ecosystem

JAX gives you primitives. Libraries give you ergonomics:

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax is the standard optimizer library. It separates the gradient transformation (Adam, SGD, clipping) from the parameter update, making it trivial to compose:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### When to Use JAX vs PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class (Google built both) | Community-maintained (torch_xla) |
| GPU support | Good (CUDA via XLA) | Best-in-class (native CUDA) |
| Debugging | Hard (tracing + compilation) | Easy (eager, line-by-line) |
| Ecosystem | Research-focused (Flax, Equinox) | Massive (HuggingFace, torchvision, etc.) |
| Hiring | Niche (Google/DeepMind/Anthropic) | Mainstream (everywhere) |
| Large-scale training | Superior (XLA, pmap, mesh) | Good (FSDP, DeepSpeed) |
| Prototyping speed | Slower (functional overhead) | Faster (mutate and go) |
| Production inference | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Who uses it | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

诚实的答案：除非你有特定原因需要使用 JAX，否则使用 PyTorch。那些理由是——TPU 访问、需要逐样本梯度、大规模多设备训练，或者在 Google/DeepMind/Anthropic 工作。
<<<

### Random Numbers in JAX

JAX does not have a global random state. Every random operation requires an explicit PRNG key:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

This is annoying at first. But it guarantees reproducibility across devices and compilations -- a property that PyTorch's `torch.manual_seed` cannot guarantee in multi-GPU settings.

```figure
batchnorm-effect
```

## Build It

### 第 1 步：配置与数据

我们将使用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP。784 个输入，两个隐藏层分别有 256 和 128 个神经元，10 个输出类别。
<<<

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Step 2: Initialize Parameters

没有类。只是一个返回 pytree 的函数：
<<<

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

Let me translate:

"He-initialization, done manually. Three PRNG keys split from one seed. Every weight is an immutable array in a nested dict."

Technical terms like "He-initialization", "PRNG", "immutable array", "nested dict", "seed", "weight" should NOT be translated.

Let me translate:
- "He-initialization, done manually." → "He初始化，手动完成。"
- "Three PRNG keys split from one seed." → "从单个种子拆分成三个 PRNG 密钥。"
- "Every weight is an immutable array in a nested dict." → "每个权重都是一个嵌套字典中的不可变数组。"

Let me preserve the line breaks (these seem to be separate sentences/lines).

I'll write:
<<<START>>>
He初始化，手动完成。从单个种子拆分成三个 PRNG 密钥。每个权重都是一个嵌套字典中的不可变数组。
<<<

### Step 3: Forward Pass

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Pure functions. Params in, prediction out. No `self`, no stored state. `loss_fn` computes cross-entropy from scratch -- softmax, log, negative mean.

### Step 4: JIT-Compiled Training Step

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` returns both the loss value and the gradients in one pass. The `@jax.jit` decorator compiles both functions to XLA. After the first call, each training step runs without touching Python.

### 步骤 5：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 epochs. ~97% test accuracy. The first epoch is slow (JIT compilation). Epochs 2-10 are fast.

注意缺失了什么：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新是一个组合函数调用。梯度被计算、由 Adam 转换、并应用于参数——全部在 `train_step` 内部完成。
<<<

## 使用它
<<<

### Flax: The Google Standard

<<<START>>>
Flax 是最常见的 JAX 神经网络库。它重新加入了 `nn.Module`，但采用了显式的状态管理：
<<<

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构与 PyTorch 相同，但 `params` 与模型分离。`model.init()` 创建参数。`model.apply(params, x)` 执行前向传递。模型对象没有状态。
<<<

### Equinox: The Pythonic Alternative

Equinox（由 Patrick Kidger 开发）将模型表示为 pytrees：
<<<

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

The model itself is a pytree. No `.apply()` needed. Parameters are just the model's leaves. This is closer to how JAX thinks.

### Optax: 可组合优化器

Optax decouples the gradient transformation from the update:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient clipping, learning rate warmup, weight decay -- all composed as a chain of transforms. Each transform sees the gradients, modifies them, and passes them to the next. No monolithic optimizer class.

## Ship It

**Installation:**

```bash
pip install jax jaxlib optax flax
```

对于 GPU 支持：
<<<

```bash
pip install jax[cuda12]
```

>>>对于 TPU（Google Cloud）：<<<

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Performance gotchas:**

- First JIT call is slow (compilation). Warm up before benchmarking.
- Avoid Python loops over JAX arrays inside JIT. Use `jax.lax.scan` or `jax.lax.fori_loop`.
- `jax.debug.print()` works inside JIT. Regular `print()` does not.
- Profile with `jax.profiler` or TensorBoard. XLA compilation can hide bottlenecks.
JAX 默认预分配 75% 的 GPU 显存。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可禁用。
<<<

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**This lesson produces:**
- `outputs/prompt-jax-optimizer.md` -- 用于选择正确的 JAX 优化器配置的提示
- `outputs/skill-jax-patterns.md` -- 涵盖 JAX 中函数式模式的技能
<<<

## 练习

<<<

Let me translate this text.


1. 给 MLP 添加 dropout。在 JAX 中，dropout 需要一个 PRNG key——在前向传播中传递一个 key，并为每个 dropout 层进行 split。比较有和无 dropout 时的测试准确率。

2. Use `jax.vmap` to compute per-example gradients for a batch of 32 MNIST images. Compute the gradient norm for each example. Which examples have the largest gradients, and why?

3. Replace the manual forward function with a generic `mlp_forward(params, x)` that works for any number of layers. Use `jax.tree.leaves` to determine the depth automatically.

4. Benchmark the training step with and without `@jax.jit`. Time 100 steps of each. How large is the speedup on your hardware? What is the compilation overhead on the first call?

5. Implement gradient clipping by composing `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Train with and without clipping. Plot the gradient norm over training to see the effect.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| XLA | "The thing that makes JAX fast" | Accelerated Linear Algebra -- a compiler that fuses operations and generates optimized GPU/TPU kernels from a computation graph |
| JIT | "Just-in-time compilation" | JAX traces the function on first call, compiles to XLA, then runs the compiled version on subsequent calls |
| Pure function | "No side effects" | A function where the output depends only on inputs -- no global state, no mutation, no randomness without explicit keys |
| vmap | "Auto-batching" | Transforms a function that processes one example into one that processes a batch, without rewriting |
| pmap | "Auto-parallelism" | Replicates a function across multiple devices and splits the input batch |
| Pytree | "Nested dict of arrays" | Any nested structure of lists, tuples, dicts, and arrays that JAX can traverse and transform |
| Tracing | "Recording the computation" | JAX executes the function with abstract values to build a computation graph, without computing real results |
| Functional autodiff | "grad of a function" | Computing derivatives by transforming functions, not by attaching gradient storage to tensors |
| Optax | "JAX's optimizer library" | A composable library of gradient transformations -- Adam, SGD, clipping, scheduling -- that chain together |
| Flax | "JAX's nn.Module" | Google's neural network library for JAX, adding layer abstractions while keeping state explicit |

## Further Reading

- JAX documentation: https://jax.readthedocs.io/ -- the official docs, with excellent tutorials on grad, jit, and vmap
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- the original paper explaining the design philosophy
- Flax documentation: https://flax.readthedocs.io/ -- Google's neural network library for JAX
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- the Pythonic alternative to Flax
- DeepMind, "Optax: composable gradient transformation and optimisation" -- the standard optimizer library
- "You Don't Know JAX" (Colin Raffel, 2020) -- a practical guide to JAX gotchas and patterns, from one of the T5 authors
