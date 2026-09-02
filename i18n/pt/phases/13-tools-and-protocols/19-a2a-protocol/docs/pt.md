# A2A  Protocolo de agente a agente

> O MCP é agente-to-tool. A2A (Agent2Agent) é um protocolo aberto para permitir que agentes opacos construídos em diferentes estruturas colaborem. Lançado pelo Google em abril de 2025, doado à Linux Foundation em junho de 2025, alcançando v1.0 em abril de 2026 com mais de 150 apoiadores, incluindo AWS, Cisco, Microsoft, Salesforce, SAP e ServiceNow. Absorveu o ACP da IBM e acrescentou a extensão dos pagamentos AP2. Esta lição fala sobre o Cartão de Agente, o ciclo de vida da tarefa e as duas ligações de transporte.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Objetivos de aprendizagem

- Distinguir casos de utilização de agente para ferramenta (MCP) de casos de utilização de agente para agente (A2A).
- Publica um cartão de agente em `/.well-known/agent.json`com competências e metadados de ponto final.
- Caminhar o ciclo de vida da tarefa (submetido → trabalhando → requerido → concluído / falhado / cancelado / rejeitado).
- Use Mensagens com Partes (texto, arquivo, dados) e Artefactos como saídas.

## O problema

Um agente de atendimento ao cliente precisa delegar a redação de relatórios a um agente de escritores especializado.

- Funciona, mas cada emparejamento é único.
- Base de código compartilhada, exige que os dois agentes executem o mesmo quadro.
- MCP: não se encaixa: MCP é para chamar ferramentas, não para dois agentes colaborando enquanto preservam o raciocínio interno opaco de cada agente.

A2A preenche a lacuna. Modela a interação como um agente enviando uma tarefa para outro, com um ciclo de vida, mensagens e artefatos. O estado interno do agente chamado permanece opaco.

A A2A é o protocolo "deixe os agentes através de frameworks falarem uns com os outros".

## O conceito

### Agente Card

Todos os agentes que cumprem os requisitos A2A publicam um cartão em `/.well-known/agent.json`- Não .

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

A descoberta é baseada em URL: trazer o cartão, aprender a URL do ponto final A2A, enumerar habilidades.

### Cartões de Agente assinados (AP2)

A extensão AP2 (septembro de 2025) adiciona assinaturas criptográficas aos cartões de agentes. Um editor assinou seu próprio cartão com um JWT; os consumidores verificam.

### Ciclo de vida da tarefa

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

Os clientes iniciam com `tasks/send`- O agente chamado passa por estados; os clientes subscrevem as atualizações de estado através da SSE ou da pesquisa.

### Mensagens e partes

Uma mensagem contém uma ou mais Partes:

- `text` conteúdo simples.
- `file`Base64 blob com mimeType.
- `data` carregamento útil JSON (entrada estruturada para o agente chamado).

Exemplo:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifactos

As saídas são artefatos, não cordas brutas.

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Os artefatos podem ser transmitidos como pedaços.

### Duas obrigações de transporte

1. **JSON-RPC over HTTP.** `/a2a`Endpoint, POST para pedidos, SSE opcional para streaming.
2. **gRPC.**Para ambientes empresariais onde o gRPC é nativo.

Ambas as ligações têm a mesma forma lógica de mensagem.

### Preservação da opacidade

Um princípio de design chave: o estado interno do agente chamado é opaco. O chamador vê o estado da tarefa e artefatos. A cadeia de pensamento do agente chamado, suas chamadas de ferramenta, sua delegação de sub-agente  são todas invisíveis. Isso é diferente do MCP, onde as chamadas de ferramentas são transparentes.

A A2A pode ser "chamá-lo para o agente de atendimento ao cliente" sem que o chamador aprenda como esse agente implementa o serviço.

### Linha de tempo

- **2025-04-09.**O Google anuncia A2A.
- **2025-06-23.**Doado à Fundação Linux.
- **2025-08.**Absorve o ACP da IBM.
- **2025-09.**Naves de extensão AP2 (pagamentos por agentes).
- **2026-04.**V1.0 lançado com mais de 150 organizações de apoio.

### Relação com a MCP

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

Use MCP quando quiser invocar uma ferramenta específica. Use A2A quando quiser delegar uma tarefa inteira a outro agente. Muitos sistemas de produção usam ambos: um agente usa MCP para sua camada de ferramentas e A2A para sua camada de colaboração.

```figure
a2a-task-lifecycle
```

## Usá-lo

`code/main.py`Implementa um mínimo de A2A: um agente de investigação publica o seu cartão, um agente de redacção recebe um `tasks/send`com partes incluindo um PDF e uma instrução de texto, transições através de trabalho → input_required → working → completado, e retorna um artefato de texto.

O que ver:

- Forma de cartão JSON.
- Associação de ID de tarefa e transições de estado.
- Mensagens com peças de tipo misto.
- Requerido para entrada de ramo no meio da tarefa.
- O artefato retorna ao término.

## Envia-o

Esta lição produz`outputs/skill-a2a-agent-spec.md`. Dado um novo agente que deve ser chamado por outros agentes, a habilidade produz o JSON do Cartão do Agente, esquema de habilidades e plano de ponto final.

## Exercícios

1. Corra .`code/main.py`. Rastrear todo o ciclo de vida da tarefa, incluindo a pausa de entrada necessária, quando o agente chamado pedir uma clarificação.

2. Adicione um cartão de agente assinado, assine com HMAC sobre o JSON canônico do cartão, escreva um verificador e confirme que falha em um cartão mutado.

3. Implementar o streaming de tarefas: o agente de redação emite três fragmentos de artefatos incrementais sobre o SSE e o chamador acumula-los.

4. Desenhar um agente A2A que envolva um servidor MCP. mapear cada ferramenta MCP para uma habilidade A2A. Observe as compensações  que opacidade é perdida?

5. Leia o anúncio A2A v1.0 e identifique a única característica que ainda não foi implementada por nenhuma estrutura a partir de abril de 2026.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent protocol" | Open protocol for opaque agent collaboration |
| Agent Card | "`.well-known/agent.json`" | Published metadata describing an agent's skills and endpoint |
| Skill | "A callable unit" | A named operation the agent supports (analog to MCP tool) |
| Task | "Unit of delegation" | A work item with a lifecycle and final artifact |
| Message | "Task input" | Carries Parts (text, file, data) |
| Part | "Typed chunk" | `text` / `file` / `data` element of a message |
| Artifact | "Task output" | Named, typed output returned on completion |
| AP2 | "Agent Payments Protocol" | Signed Agent Cards extension for trust and payments |
| Opacity | "Black-box collaboration" | Called agent's internals are hidden from caller |
| Input-required | "Task pause" | Lifecycle state when the agent needs more info |

## Mais leitura

- [a2a-protocol.org](https://a2a-protocol.org/latest/) especificação canónica A2A
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) Implementações de referência e KDS
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) Transferência de governança de Junho de 2025
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) Mapa de estrada e impulso dos parceiros
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) Nota de liberação e orientação compatível para trás
