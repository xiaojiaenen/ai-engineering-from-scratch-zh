# A família de otimização das preferências diretas

> Rafailov e outros. (2023) mostrou que o ótimo da RLHF tem uma forma fechada em termos de dados de preferência, para que você possa ignorar o modelo de recompensa explícito e otimizar a política diretamente. Essa percepção gerou uma família  IPO, KTO, SimPO, ORPO, BPO  cada um corrigindo um modo de falha de DPO. Em 2026, os algoritmos de alinhamento direto enviam mais corridas de pós-treino fronteiriças do que o PPO. Mas a curva de otimização excessiva da lição 2 ainda se aplica: os DAAs não escapam do Goodhart, eles apenas se movem onde morde.

**Type:** Learn
**Languages:** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time:** ~75 minutes

## Objetivos de aprendizagem

- Derivar o formato fechado do DPO do RLHF-with-KL-optimum.
- Indicar o modo de falha de cada uma das correções de IPO, KTO, SimPO, ORPO e BPO no DPO.
- Distinguir "margem de recompensa implícita" da "força de preferência" e explicar por que o mapeamento da identidade da OPI é importante.
- Explique por que Rafailov et al. (NeurIPS 2024) provam que os DAAs são excessivamente óptimas apesar de não ter RM explícita.

## O problema

O objectivo do RLHF (Lessão 1):

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

tem um óptimo conhecido:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

Assim, a recompensa é definida implícitamente pela relação entre a política ideal e a referência:

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

Substitua isso na probabilidade de preferência Bradley-Terry e na função de partição .`Z(x)`cancelas porque depende apenas de`x`O que resta é uma perda nos parâmetros da política sozinho.

A redução: a derivação assume que o ótimo é alcançável, os dados de preferência são distribuídos e a política de referência é a âncora de modo verdadeiro. Nenhum destes se aplica exatamente.

## O conceito

### DPO (Rafailov et al., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

O que pode dar errado:

- A diferença implícita de recompensa .`beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)`Uma pequena preferência pode produzir uma lacuna arbitrariamente grande.
- O disco de perda seleciona e rejeita log-probs em direções opostas. Pode empurrar o log-prob absoluto escolhido para baixo, desde que o rejeitado cai mais rápido.
- As preferências fora da distribuição (par raro raro vs par raro raro) produzem recompensas implícitas arbitrárias.

### O IPO (Azar et al., 2024)

A Optimização de Preferências de Identidade substitui o log-sigmoid por um mapeamento de identidade na probabilidade de preferência. A perda se torna um erro quadrado em um alvo limitado:

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

A margem é limitada por `1/(2 beta)`A força de preferência e a diferença implícita entre a recompensa são proporcionais.

### O TCO (Ethayarajh et al., 2024)

A otimização de Kahneman-Tversky elimina completamente a estrutura em pares. Dado uma saída única rotulada e um sinal binário "desejável" ou "indesejável", ele mapeia para um utilitário de teoria prospectiva:

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

A vantagem: pode utilizar dados não emparejados, o que é muito mais abundante.

### SimPO (Meng et al., 2024)

Otimizar as preferências simples alinha o sinal de treinamento com a geração.

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

com uma margem `gamma`A normalização da comprimento elimina o incentivo a explorar o modo de falha de viés de comprimento do DPO (mais`y_w`dá uma maior lacuna de log-prob por construção).

### ORPO (Hong et al., 2024)

Otimizar a preferência de odds-ratio adiciona um termo de preferência à probabilidade de registro negativo de SFT padrão:

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

Não há política de referência  o termo SFT é o regulador. Trein em uma única etapa do modelo base ao modelo alinhado. Não há ponto de controle separado SFT.

### BPO (submissão ICLR 2026, OpenReview id=b97EwMUWu7)

Identifica o problema de respostas degradadas escolhidas: DPO preserva o ranking `y_w > y_l`Mas o log-prob absoluto de `y_w`BPO adiciona uma correção de linha única que penaliza os movimentos para baixo na resposta escolhida.

### O resultado universal: os DAAs ainda se optimizam demais

Rafailov et al. "Lei de Escalada para a Optimização Excessiva de Modelos de Recompensa em Algoritmos de Alinhamento Direto" (NeurIPS 2024) treinou políticas com DPO, IPO, SLiC em múltiplos conjuntos de dados em todos os orçamentos KL. As curvas ouro-recompensa-versus-KL têm a mesma forma Gao et al. Pico e colapso. A busca implícita de recompensa de amostras fora da distribuição durante o treinamento; regularização KL não estabiliza isso.

Os DAAs não escapam de Goodhart. Eles mudam a superfície onde morde de "modelo de recompensa super-otimizado" para "ratio de política de referência super-otimizado".

### Escolher entre eles (2026)

- Se tiver grandes dados de preferência em pares: DPO com beta conservador, SimPO se for evidente o viés de comprimento.
- Se tiver feedback binário não pareado: KTO.
- Se quiser um gasoduto de uma única etapa de um modelo base: ORPO.
- Se virem registos degradados de registos escolhidos nos registos do DPO: BPO.
- Se as preferências variarem muito e o DPO estiver saturado: IPO.

Cada laboratório corre todos os cinco em uma bateria e escolhe o vencedor por tarefa. Não há razão para o ótimo ser o mesmo para o raciocínio matemático e segurança.

```figure
dpo-margin
```

## Usá-lo

`code/main.py`Comparar seis perdas (DPO, IPO, KTO, SimPO, ORPO, BPO) em um conjunto de dados de preferências de brinquedos onde a força de preferência real varia por pares. Cada perda é otimizada contra a mesma amostra de 500 pares com uma pequena política de softmax.

## Envia-o

Esta lição produz`outputs/skill-preference-loss-selector.md`. Tendo em conta as estatísticas dos conjuntos de dados (paradas versus não paradas, variáveis versus preferências uniformes, distribuição de comprimento) e um alvo (estágio único ou SFT-then-preferência), recomendamos uma perda de preferência e relatamos o modo de falha contra o qual protege.

## Exercícios

1. Corra .`code/main.py`. Relatar a queda final de registro-probes escolhido para DPO e BPO.

2. Modifique os dados de preferência para que todos os pares tenham a mesma força. Qual dos seis métodos é mais robusto? Qual degrada? Explique a vantagem da IPO aqui.

3. Faça as respostas rejeitadas em média 2 vezes mais longas do que as escolhidas.

4. Rafailov et al. (NeurIPS 2024) alegam que os DAAs otimizam demais. Reproduzir uma versão de ponto único: gráfico escolhido-menos-rejeitado divergência KL e observar o excesso de otimização no DPO em beta grande.

5. Leia o resumo do documento do BPO (OpenReview b97EwMUWu7).`code/main.py`- Não .

## Termos-chave

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| DPO | "RLHF without a reward model" | Loss derived from the closed-form RLHF optimum; policy parameters only |
| Implicit reward | "the log-ratio" | `beta * log(pi(y\|x) / pi_ref(y\|x))` — the DPO-implied reward |
| IPO | "bounded DPO" | Replaces log-sigmoid with identity; implicit reward gap capped by `1/(2 beta)` |
| KTO | "unpaired DPO" | Prospect-theory utility over single labels with loss aversion |
| SimPO | "reference-free DPO" | Length-normalized log-likelihood + margin; no reference policy |
| ORPO | "one-stage DPO" | NLL + odds-ratio preference term; trains from base model in one pass |
| BPO | "chosen-preserving DPO" | DPO plus a penalty for decreasing the chosen response's absolute log-prob |
| Degraded Chosen | "chosen goes down" | DPO decreases chosen log-prob so long as rejected falls faster |
| DAA | "direct alignment algorithm" | Any preference-loss method that skips an explicit RM |

## Mais leitura

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) OPI
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
