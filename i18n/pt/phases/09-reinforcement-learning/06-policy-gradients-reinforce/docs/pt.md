# Política Gradiente  REINFORCE do zero

> Para de estimar o valor. Parametrizar a política diretamente, calcular o gradiente do retorno esperado, passo para cima. Williams (1992) escreveu em um teorema. É por isso que PPO, GRPO e cada ciclo LLM RL existem.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 minutes

## O problema

Q-learning e DQN paramétricos da função *valor*.`argmax Q`O que é bom para ações discretas e estados discretos.`argmax`sobre um torque de 10 dimensões?) ou quando você quer uma política estocástica (`argmax`É determinista por construção).

Os gradientes de política parametrizam a *política* em vez disso. `π_θ(a | s)`A rede neural é uma rede neural que produz uma distribuição sobre ações.`θ`- Passa para cima.`argmax`Não há recursão de Bellman, só ascensão de gradiente.`J(θ) = E_{π_θ}[G]`- Não .

O teorema de REINFORCE (Williams 1992) diz que este gradiente é computavel: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`- Exerça um episódio, calcula o retorno, multiplica por`∇ log π_θ(a | s)`- A média, a ascensão gradual, pronto.

Cada algoritmo LLM-RL em 2026  PPO, DPO, GRPO  é um aperfeiçoamento da REINFORCE.

## O conceito

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**The policy gradient theorem.**Para qualquer política`π_θ`Parametrizado por `θ`- Não .

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

onde`G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}`é o retorno descontado do passo `t`A expectativa é de que as trajetórias sejam completas .`τ`amostragem de `π_θ`- Não .

**The proof is short.**Diferenciar `J(θ) = Σ_τ P(τ; θ) G(τ)`- Não é o que se espera.`∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`Factor `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`Os termos de ambiente desaparecem. Duas linhas de álgebra dão-lhe o teorema.

**Variance reduction tricks.**A Vanilla ReINFORCE tem uma variância homicida. Os retornos são ruidosos.`∇ log π`O produto é muito barulhento.

1. **Baseline subtraction.**Substitui`G_t`com`G_t - b(s_t)`para qualquer linha de base `b(s_t)`que não depende de `a_t`- Imparcial porque`E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Escolha típica: `b(s_t) = V̂(s_t)`aprendizado por um crítico → ator-crítico (Lessão 07).
2. **Reward-to-go.**Substitui`Σ_t G_t · ∇ log π_θ(a_t | s_t)`com`Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`. Apenas os retornos futuros são importantes para uma determinada ação  Os retornos passados contribuem com ruído zero-médio.

Combinados, obtém:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

que é REINFORCE com uma linha de base  o ancestral directo do A2C (Lessão 07) e do PPO (Lessão 08).

**Softmax policy parameterization.**Para ações discretas, a escolha padrão:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

onde`f_θ`É qualquer rede neural que produz uma pontuação por ação.

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

Por exemplo, o resultado da acção tomada menos o seu valor esperado no âmbito da política.

**Gaussian policy for continuous actions.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`- Não .`∇ log N(a; μ, σ)`A fase 9 · 07 é a fase de SAC.

```figure
policy-gradient-landscape
```

## Construí-lo

### Passo 1: rede de políticas softmax

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Use uma política linear (um vetor de peso por ação) para um envelope tabuleiro. Para Atari, troque em uma CNN e mantenha a cabeça softmax.

### Passo 2: amostragem e probabilidade de registro

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Passo 3: lançamento com log-probes capturados

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Passo 4: Atualização da REINFORCE

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

O gradiente`∇ log π(a|s) = e_a - π(·|s)`(exceto em`a`- probabilidades) é o coração dos gradientes de política softmax.

### Passo 5: Linhas de base

Uma média corrente de `G`Em relação aos episódios recentes, a redução de variância é suficiente para que um GridWorld 4×4 seja executado; é preciso ~ 500 episódios para convergir.`V̂(s)`E você tem crítico de ator.

## Encurralagens

- **Exploding gradients.**Os rendimentos podem ser enormes.`G`- Não .`~N(0, 1)`através do lote antes de multiplicar por `∇ log π`- Não .
- **Entropy collapse.**A política converge para uma ação quase determinista muito cedo, deixa de explorar, fica presa.`β · H(π(·|s))`Para o objectivo.
- **High variance.**A Vanilla REINFORCE precisa de milhares de episódios.
- **Sample inefficiency.**A correcção extra-política através da amostragem de importância traz dados, ao custo da variação (a taxa da OPP é um peso de IS reduzido).
- **Non-stationary gradients.**O mesmo gradiente de há 100 episódios usa o antigo .`π`Os métodos de política atualizam cada algumas implementações por este motivo.
- **Credit assignment.**Sem recompensa, recompensas passadas contribuem com ruído.

## Usá-lo

Em 2026, o REINFORCE raramente é executado diretamente, mas a sua fórmula de gradiente está em toda parte:

| Use case | Derived method |
|----------|---------------|
| Continuous control | PPO / SAC with Gaussian policy |
| LLM RLHF | PPO with KL penalty, running on token-level policy |
| LLM reasoning (DeepSeek) | GRPO — REINFORCE with group-relative baseline, no critic |
| Multi-agent | Centralized-critic REINFORCE (MADDPG, COMA) |
| Discrete action robotics | A2C, A3C, PPO |
| Preference-only settings | DPO — REINFORCE rewritten as a preference-likelihood loss, no sampling |

Quando você lê`loss = -advantage * log_prob`Os trabalhos completos (DPO, GRPO, RLOO) são truques de redução de variância no topo desta linha.

## Envia-o

Salva como`outputs/skill-policy-gradient-trainer.md`- Não .

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Exercícios

1. **Easy.**Implementar REINFORCE no 4×4 GridWorld com uma política de softmax linear. Treinar por 1.000 episódios sem uma linha de base. Planejar a curva de aprendizagem; medir a variância (std de retornos).
2. **Medium.**Adicione uma linha de base de corrida média. Treine novamente. Compare a eficiência da amostra e a variância com a corrida de baunilha. Em quanto a linha de base reduz os passos para a convergência?
3. **Hard.**Adicionar um bônus de entropia `β · H(π)`- Esvaziar .`β ∈ {0, 0.01, 0.1, 1.0}`O que é que é o ponto de encontro desta tarefa?

## Termos-chave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy gradient | "Train the policy directly" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; derived from the log-derivative trick. |
| REINFORCE | "The original PG algorithm" | Williams (1992); Monte Carlo returns multiplied by log-policy gradient. |
| Log-derivative trick | "Score function estimator" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; makes gradients of expectations tractable. |
| Baseline | "Variance reduction" | Any `b(s)` subtracted from `G`; unbiased because `E[b · ∇ log π] = 0`. |
| Reward-to-go | "Only future returns count" | `G_t^{from t}` instead of the full `G_0`; correct and lower-variance. |
| Entropy bonus | "Encourage exploration" | `+β · H(π(·\|s))` term keeps the policy from collapsing. |
| On-policy | "Train on what you just saw" | Gradient expectation is w.r.t. the current policy — cannot reuse old data directly. |
| Advantage | "How much better than average" | `A(s, a) = G(s, a) - V(s)`; the signed quantity REINFORCE-with-baseline multiplies. |

## Mais leitura

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) o papel original do REINFORCE.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) o teorema moderno da política-gradiente com aproximação de funções.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) apresentação de livros didáticos.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) Exposição pedagógica clara com código PyTorch.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) A redução da variância e a visão natural-gradiente que liga a REINFORCE à família da região de confiança (TRPO, PPO).
