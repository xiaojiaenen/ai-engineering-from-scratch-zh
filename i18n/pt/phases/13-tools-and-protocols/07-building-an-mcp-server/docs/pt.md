# Construir um servidor MCP: Python sem estado e TypeScript

> Um servidor MCP moderno não se lembra de um aperto de mão. Valida os metadados em cada solicitação, executa um processador e retorna um resultado digitado.

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Lesson 06
**Time:** ~85 minutes

## Objetivos de aprendizagem

- Implementação obrigatória `server/discover`para MCP `2026-07-28`- Não .
- Validar a versão do protocolo e as capacidades do cliente em cada pedido.
- Expor ferramentas, recursos e instruções com ordem de lista determinista.
- Retorno .`resultType`, identidade do servidor, e cache indica os resultados corretos.
- Servir o mesmo contrato sem estado sobre estúdio de linha nova e limitada em Python e TypeScript.

## O problema

Um servidor que armazena capacidades do cliente após a primeira mensagem é fácil de construir e difícil de operar. O mesmo processo pode servir clientes sequenciais. Uma solicitação remota pode aterrar em um trabalhador diferente. Uma declaração de capacidade obsoleta pode vazamento de comportamento através de limites de autorização.

MCP `2026-07-28`O aplicativo ainda pode manter notas duradouras, trabalhos ou manuais de estado explícito. O que não pode manter é o estado de protocolo oculto que muda a forma como uma solicitação posterior é decodificada.

Esta lição constrói um servidor de notas duas vezes. As versões Python e TypeScript usam apenas suas bibliotecas padrão para o núcleo do protocolo. Ambos expõem os mesmos métodos e aplicam o mesmo contrato de fio.

## O conceito

### O moderno circuito de despacho

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

Três regras do estúdio ainda são importantes:

- Escreva apenas mensagens JSON-RPC para stdout. Envie diagnósticos para stderr.
- Delimite as mensagens com uma linha nova e coloque em branco cada resposta.
- Saia imediatamente quando o STD chegar à EOF.

A vida útil do processo é uma vida útil do transporte.

### Requisito de validação

Cada pedido deve ter:

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

São necessários os dois primeiros campos. `clientInfo`É recomendável validar uma forma de identidade actual, mas não a tratar como autenticação.

Se a versão não for suportada, retorne o código `-32022`com`requested`E ...`supported`. Metadados de pedido faltantes são parâmetros inválidos, código `-32602`Nunca preencha campos faltantes de uma chamada anterior.

### A descoberta obrigatória

Servidores modernos devem implementar `server/discover`. Um resultado completo de descoberta inclui versões modernas suportadas, recursos, instruções opcionais, dicas de cache e identidade do servidor no resultado `_meta`- Não .

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

O Discovery não desbloqueia o servidor.`tools/list`sem chamar descoberta porque`tools/list`já contém os mesmos metadados da solicitação.

### Ferramentas

`tools/list`Retorna uma lista determinista de descriptórios de ferramentas. A ordem estável melhora o caching de resposta e mantém o contexto do modelo estável. O resultado também requer `ttlMs`E ...`cacheScope`- Não .

`tools/call`Retorna blocos de conteúdo e `isError`. Use um erro JSON-RPC quando o protocolo envolvente ou os parâmetros do método são inválidos.`isError: true`Quando uma invocação de ferramenta válida for executada mas a própria ferramenta falhar.

As anotações das ferramentas continuam a ser sugestões, não medidas de execução:

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

O servidor deve ainda aplicar a autorização real.

### Recursos

`resources/list`Retorna descriptores URI estáveis. `resources/read`Retorna conteúdo digitado. Ambos são caché em `2026-07-28`, por isso ambas incluem`ttlMs`E ...`cacheScope`- Não .

Utilização`cacheScope: "private"`Para dados de notas específicos do usuário. Um cache compartilhado não deve reutilizar uma resposta privada em contextos de autorização.

A transferência moderna não utiliza `resources/subscribe`Um cliente abre .`subscriptions/listen`e pedidos `resourceSubscriptions`A lição 10 construi esse fluxo.

### Instruções

`prompts/list`é cacheable e determinista. `prompts/get`O resultado do prompt renderado é completo, mas não é um dos resultados de leitura ou cache que requer sugestões de cache.

### Cada resultado bem sucedido é digitado

Os exemplos usam uma embalagem para cada sucesso:

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

Lista, leitura e manipulação de descobertas adicionar `ttlMs`- E mais .`cacheScope`Centralizando esta embalagem impede que um manipulador omita silenciosamente os campos de resultados modernos.

### Não há solicitações iniciadas pelo servidor

Um servidor moderno pode enviar notificações relacionadas a uma solicitação do cliente ou notificações em um cliente aberto `subscriptions/listen`Não deve enviar a sua própria solicitação JSON-RPC.

Quando um processador precisa de amostragem, elicitação ou entrada de raízes, ele retorna um `input_required`Resultado. O cliente atende as solicitações de entrada embutidas e retrata o método original com um novo id de solicitação.

### Compatibilidade explícita com o legado

Um servidor de dupla era pode também implementar o `2025-11-25`Ele escolhe um comportamento moderno quando necessário.`_meta`campos estão presentes e comportamento legado quando recebe `initialize`- Não .

Não coloque um `2026-07-28`Não se preencha modernos.`resultType`O código nesta lição é deliberadamente moderno apenas para que suas invariantes permaneçam visíveis.

```figure
t3-dispatch-loop
```

## Usá-lo

Execute a demonstração e testes finitos do servidor Python:

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

Execute a porta TypeScript com um executador TypeScript:

```bash
npx tsx main.ts --demo
```

A demonstração envia .`server/discover`O sistema de dados de cada um dos servidores, que é um sistema de dados de base, é um sistema de dados de base, que permite que cada um dos servidores seja identificado como um servidor.

## Envia-o

Esta lição vai avançar .`outputs/skill-mcp-server-scaffolder.md`. Produz um plano de servidor moderno com um contrato de descoberta, validação por solicitação, listas deterministas cacheáveis e um adaptador isolado herdado opcional.

## Exercícios

1. Remover recursos de uma solicitação e provar que o servidor não reutiliza a declaração da solicitação anterior.
2. Reverte o `TOOLS`- Não .`PROMPTS`Confirme que todos os resultados da lista permanecem estáveis.
3. Adicione um destrutivo .`notes_delete`O sistema de verificação de autorização é um instrumento de verificação de autorização dentro do executor.`destructiveHint`Só como uma sugestão de experiência.
4. Adicionar`resources/templates/list`com`ttlMs`- Não .`cacheScope`, e ordem determinista.
5. Construa um adaptador de legado separado para `2025-11-25`Adicionar testes que provem que uma solicitação moderna nunca entra nela.

## Termos-chave

| Term | Meaning |
|------|---------|
| Stateless server | Handles each request from its own metadata without protocol-session memory |
| `server/discover` | Mandatory modern method that advertises versions and capabilities |
| Complete result | Successful modern result with `resultType: "complete"` |
| Cacheable result | Discovery, list, or resource-read result with `ttlMs` and `cacheScope` |
| Deterministic list | Same logical registry produces the same item order |
| Server identity | Recommended `io.modelcontextprotocol/serverInfo` in result `_meta` |
| Tool error | Valid tool call that returns content with `isError: true` |
| Protocol error | Invalid JSON-RPC or MCP request returned through `error` |

## Mais leitura

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
