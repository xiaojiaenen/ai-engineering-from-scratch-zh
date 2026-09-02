# A interface das ferramentas  Por que os agentes precisam de uma entrada/entrada estruturada

> Um modelo de linguagem produz tokens. Um programa realiza ações. A diferença entre esses dois é a interface de ferramentas: um contrato que permite que o modelo solicite uma ação e o host a executar.`tools/call`A A2A é uma codificação diferente do mesmo ciclo de quatro passos.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 minutes

## Objetivos de aprendizagem

- Explique por que um Mestrado em Direito que só pode gerar texto não pode, por si só, tomar medidas contra o mundo real.
- Desenhe o ciclo de chamada de ferramenta de quatro passos (descreva → decide → execute → observe) e nomee quem possui cada passo.
- Escreva uma descrição de ferramenta como três partes: nome, entrada de JSON Schema e função de executor determinista.
- Distinguir entre as ferramentas puras e as que produzem efeitos colaterais e explicar por que a divisão é importante para a segurança.

## O problema

Um LLM emite uma distribuição de probabilidade sobre o próximo token. Isso é toda a superfície de saída. Se você perguntar a um modelo de chat "o que é o tempo em Bengaluru agora", ele pode escrever uma frase plausível, mas não pode marcar em uma API meteorológica. A frase pode ser correta por coincidência ou três dias de esgotamento.

Fechar essa lacuna é o propósito da interface da ferramenta. O programa host  o seu agente runtime, Claude Desktop, ChatGPT, Cursor, ou um script personalizado  anuncia uma lista de ferramentas chamáveis para o modelo. O modelo, quando decide que uma ação é necessária, emite uma carga útil estruturada que nomeia uma ferramenta e os seus argumentos. O anfitrião analisa a carga útil, executa a ferramenta de forma real e realimenta o resultado. O ciclo continua até que o modelo decida que não são necessárias mais chamadas.

A primeira versão deste contrato foi lançada em junho de 2023 como o parâmetro de "funções" da OpenAI.`tool_use`Blocos em Claude 2.1.`functionDeclarations`A A2A (abril de 2026, v1.0) colocou a mesma primitiva para a delegação de agente a agente.

O ciclo de quatro passos é a invariante por baixo de tudo isto.

## O conceito

### Passo um: descrever

O anfitrião declara cada ferramenta com três campos.

- **Name.**Um identificador estável e legível por máquina.`get_weather`Não é "o tempo".
- **Description.**Um breve texto em um parágrafo em língua natural. "Use quando o usuário perguntar sobre as condições atuais de uma cidade específica.
- **Input schema.**Um objeto de JSON Schema (projeto 2020-12) descrevendo os argumentos da ferramenta.

Os provedores modernos seriam essas declarações no prompt do sistema usando um modelo específico do fornecedor, para que você, como o chamador, apenas trate o formulário estruturado.

### Passo dois: decidir

Dada a mensagem do usuário e as ferramentas disponíveis, o modelo escolhe um dos três comportamentos.

1. **Answer directly**Não há chamada de ferramentas.
2. **Call one or more tools.**Emite objetos de chamada estruturados.`parallel_tool_calls: true`(por padrão em OpenAI e Gemini, opta-se em Anthropic) o modelo pode emitir várias chamadas em uma vez.
3. **Refuse.**As saídas estruturadas de modo rigoroso podem produzir um tipo de`refusal`Bloqueio em vez de chamada.

Uma carga útil de chamada de ferramenta tem três campos estáveis: uma chamada `id`, uma ferramenta`name`, e um JSON `arguments`O id existe para que o host possa correlacionar o resultado posterior com a chamada específica, o que importa quando chamadas paralelas voltam fora de ordem.

### Passo três: Execução

O host recebe a chamada, valida argumentos contra o esquema declarado e executa o executor. Argumentos inválidos significam que o modelo alucinou um campo ou usou o tipo errado  um modo de falha muito comum em modelos fracos. Os hosts de produção fazem uma das três coisas em argumentos inválidos: falham rapidamente e superficialmente o erro para o modelo, reparam o JSON com um parser restrito, ou tentam novamente o modelo com o erro de validação incluído no prompt.

O executor em si é código comum. Python, TypeScript, um comando shell, uma consulta de banco de dados. Ele produz um resultado, que geralmente é uma cadeia, mas pode ser qualquer valor JSON ou um bloco de conteúdo estruturado (texto, imagem ou referência de recursos em MCP). O resultado deve ser serializable.

### Passo quatro: observar

O anfitrião adiciona o resultado da ferramenta à conversa (como um `tool`mensagem de papel com correspondência `id`O modelo agora tem a saída da ferramenta no contexto e pode produzir uma resposta final ou solicitar mais chamadas.

### A confiança se divide.

As ferramentas têm dois sabores que são importantes para a segurança.

- **Pure.**Somente de leitura, determinista, sem efeitos secundários. `get_weather`- Não .`search_docs`- Não .`get_current_time`É seguro ligar especulativamente.
- **Consequential.**Mutar o estado, gastar dinheiro, tocar dados do usuário. `send_email`- Não .`delete_file`- Não .`execute_trade`- Deve estar fechado.

A "Relação de dois" de Meta para segurança de agente 2026 diz que uma única vez pode combinar no máximo duas de: entrada não confiável, dados sensíveis, ação consequente. A interface da ferramenta é onde você impõe essa regra  rejeitando chamadas, exigindo confirmação do usuário ou escalando os escopo. Veja a Fase 13 · 15 para o capítulo completo de segurança e Fase 14 · 09 para políticas de permissão de nível de agente.

### Onde vive o ciclo

| Context | Who describes | Who decides | Who executes |
|---------|---------------|-------------|--------------|
| Single-turn function calling (OpenAI/Anthropic/Gemini) | App developer | LLM | App developer |
| MCP | MCP server | LLM via MCP client | MCP server |
| A2A | Agent Card publisher | Calling agent | Called agent |
| Web browser (function-calling agent) | Browser extension / WebMCP | LLM | Browser runtime |

Em todos os lugares, os mesmos quatro passos.

### Porque não pedir ao modelo para emitir JSON?

"Pedir ao modelo para responder em JSON" foi o padrão de chamada pré-função. Falha de ~5 a 15% do tempo em modelos de fronteira e muito mais em modelos menores. Os modos de falha incluem aparelhos faltantes, vírgulas traseiras, campos alucinados e tipos errados. Você então precisa de um pass de reparo JSON, uma retratação ou um decodificador restrito.

Chamadas nativas são melhores por três razões. Primeiro, o provedor treina o modelo de ponta a ponta na forma exata da chamada, de modo que a taxa válida de JSON sobe para 98 a 99% no modo rigoroso. Em segundo lugar, a carga útil da chamada fica em sua própria fenda de protocolo, não dentro do texto livre , de modo que uma chamada de ferramenta nunca vai vazando para a resposta visível ao usuário. Em terceiro lugar, os provedores exigem o cumprimento dos esquemas com decodificação restrita (modo rigoroso da OpenAI, modo de Antropic `tool_use`, dos Gémeos.`responseSchema`O resultado é garantido para validar.

A fase 13 · 02 acompanha as três APIs dos provedores lado a lado.

### Fusões de circuitos

O ciclo termina quando o modelo deixa de emitir chamadas ou o host atinge uma contagem máxima de voltas. Os hosts de produção definem isso entre 5 e 20 voltas. Além disso, você está quase certamente em um ciclo que o modelo não pode sair. Claude Code é padrão para 20; OpenAI Assistentes para 10; Cursor's agente modo para 25.

A alternativa  loops ilimitados  aparece a cada seis meses como "agente gastou $400 em chamadas API durante a noite" pós-mortem. Não enviar sem um limite.

A fase 14 · 12 abrange a recuperação de erros e a auto-cura em profundidade; a fase 17 abrange os limites da taxa de produção.

### A partir daqui, a Fase 13 vai

- Lições 02 a 05 poliram a superfície de chamada de ferramentas a nível do fornecedor.
- As lições 06 a 14 generalizar o ciclo em MCP.
- Lições 15 a 18 defendem o loop contra servidores hostis, usuários adversários e superfícies autênticas remotas não autenticadas.
- As lições 19 a 22 estendem o padrão à colaboração agente-agente, observabilidade, roteamento e embalagem.
- A lição 23 cria um ecossistema completo usando cada primitivo.

Cada lição restante é uma elaboração deste ciclo de quatro passos.

```figure
tp-tool-loop
```

## Usá-lo

`code/main.py`executa o ciclo de quatro passos sem um LLM. Uma função de "decisionador" falso simula o modelo combinando padrões na mensagem do usuário; o executor, o validador de esquema e o arnês observe-step são reais.

O que ver:

- O registro de ferramentas contém três campos por ferramenta: nome, descrição, esquema e referência de executor.
- O validador é um subconjunto mínimo de JSON Schema (tipos, requeridos, enum, min/max) escrito apenas em stdlib.
- O número de iterações do circuito limita a cinco.

## Envia-o

Esta lição produz`outputs/skill-tool-interface-reviewer.md`. Dada uma definição de ferramenta de projeto (nome + descrição + esquema + esboço do executor), a habilidade auditoria para a adequação do loop: é o nome máquina-estavel, é a descrição um uso completo breve, o esquema usa JSON Schema 2020-12 corretamente, e é a classificação pura versus consecuente explícito.

## Exercícios

1. Adicionar uma quarta ferramenta para `code/main.py`chamados`get_stock_price(ticker)`. Escreva a sua descrição como "Use quando o usuário solicita um preço atual de ação por ticker. Não use para preços históricos ou resumos de mercado".

2. Desligar o validador de esquema.`arguments`O objeto está faltando um campo necessário, e confirme que o host o rejeita antes da execução. Então passe uma chamada com um campo desconhecido extra. Decida: o host deve rejeitar ou ignorar? Justifique sua escolha com um argumento de segurança.

3. Classificar cada ferramenta no arnes como pura ou consequente.`consequential: true`A partir daí, o sistema de verificação de dados é usado para fazer uma seleção de dados de verificação de dados.

4. Desenhe o ciclo de quatro etapas em papel com a tabela de coluna do provedor acima preenchida para o seu cliente favorito (Claude Desktop, Cursor, ChatGPT ou uma pilha personalizada).

5. Leia o guia de chamadas de função do OpenAI de cima para baixo. Identifique o campo que está na solicitação, mas não no loop de quatro passos como apresentado aqui. Explique o que ele adiciona e por que é conveniente em vez de essencial.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tool | "A thing the model can call" | A triple of name + JSON-Schema-typed input + executor function |
| Function calling | "Native tool use" | Provider-level API support for emitting structured tool calls instead of prose |
| Tool call | "The model's request to act" | A JSON payload with `id`, `name`, `arguments` emitted by the model |
| Tool result | "What the tool returned" | The executor's output, wrapped in a `tool` role message with matching id |
| Parallel tool calls | "Many calls at once" | Multiple call objects in one model turn, independent and orderable by id |
| Strict mode | "Guaranteed JSON" | Constrained decoding that forces the model's output to validate against the declared schema |
| Pure tool | "Read-only tool" | No side effects; safe to re-run |
| Consequential tool | "Action tool" | Mutates external state; requires gate, audit, or user confirmation |
| Four-step loop | "The tool-call cycle" | describe → decide → execute → observe |
| Host | "Agent runtime" | The program that holds the tool registry, calls the model, and runs the executor |

## Mais leitura

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) Referência canónica para declarações de ferramentas e formas de chamada de estilo OpenAI
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)- O Claude.`tool_use`- Não .`tool_result`formato de bloco
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling)- Não .`functionDeclarations`e semântica paralela em Gemini
- [Model Context Protocol — Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) a atual generalização sem estado e agnóstica do fornecedor da interface da ferramenta
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) o dialeto esquema todas as ferramentas modernas API fala
