# Capstone: Envie um pacote de banco de trabalho de agentes reutilizáveis

> A mini-track termina com um pacote que você deixa em qualquer repo.`cp -r`A pedra angular é o artefato que este currículo trata.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phases 14 · 31 to 14 · 41
**Time:** ~75 minutes

## Objetivos de aprendizagem

- Envolva as sete superfícies de banco de trabalho num único diretório.
- Enfiar os esquemas, scripts e modelos para que um novo repo obtenha uma linha de base conhecida.
- Adicionar um único script de instalação que coloque o pacote idempotentemente.
- Decida o que fica no rebanho e o que fica fora, defendendo o corte para cada um.

## O problema

Um banco de trabalho que vive em um Google Doc, um histórico de bate-papo e três scripts semi-lembrados é um banco de trabalho que é reconstruído a cada trimestre. A cura é um pacote de versões: um repo ou diretório com as superfícies, os esquemas, os scripts e um instalador de um comando.

Terás de terminar esta lição com:`outputs/agent-workbench-pack/`enviado em disco e um `bin/install.sh`que o coloca em qualquer repo alvo.

## O conceito

```mermaid
flowchart TD
  Pack[agent-workbench-pack/] --> Docs[AGENTS.md + docs/]
  Pack --> Schemas[schemas/]
  Pack --> Scripts[scripts/]
  Pack --> Bin[bin/install.sh]
  Bin --> Repo[target repo]
  Repo --> Surfaces[all seven workbench surfaces wired]
```

### O layout do pacote

```
outputs/agent-workbench-pack/
├── AGENTS.md
├── docs/
│   ├── agent-rules.md
│   ├── reliability-policy.md
│   ├── handoff-protocol.md
│   └── reviewer-rubric.md
├── schemas/
│   ├── agent_state.schema.json
│   ├── task_board.schema.json
│   └── scope_contract.schema.json
├── scripts/
│   ├── init_agent.py
│   ├── run_with_feedback.py
│   ├── verify_agent.py
│   └── generate_handoff.py
├── bin/
│   └── install.sh
└── README.md
```

### O que fica dentro, o que fica fora

Em:

- Esquemas de superfície, são o contrato.
- Os quatro guiões acima são o tempo de execução.
- São as regras e a rubrica.

Fora:

- As tarefas pertencem ao banco de reservas do alvo, não ao pacote.
- O pacote é agnóstico.
- O grupo vive ao lado do grupo, não dentro dele.

### O instalador

Um breve .`bin/install.sh`(ou `bin/install.py`):

1. Recusar-se a instalar numa embalagem existente sem `--force`- Não .
2. Copia o pacote para o repo alvo.
3. - Cable para CI se um`.github/workflows/`- Não existe.
4. Imprima os próximos passos: preencha o quadro, define comandos de aceitação, execute o script init.

### Edição de versões

O pacote transporta um`VERSION`Os erros de esquema e alterações de script que exigem migrações aumentam o maior.`agent_state.json`Registros de que versão de pacote foi iniciado contra.

```figure
wb-pack-install
```

## Construí-lo

`code/main.py`reúne a embalagem em `outputs/agent-workbench-pack/`ao lado da lição, sementeado com os esquemas e roteiros das lições anteriores nesta mini-track e os documentos que já escreveu.

- É o que é ?

```
python3 code/main.py
```

O script copia e pinha as superfícies, escreve o README, imprime a árvore de pacotes e sai do zero.

## Padrões de produção em silêncio

Um pacote só é valioso se sobreviver a garfos, atualizações e um ambiente hostil.

**`VERSION` is the contract, not the marketing.**Os problemas principais exigem uma migração de estado, os problemas menores exigem uma revisão, os problemas de parche são apenas de documentos.`.workbench-version`no repo-alvo em cada instalação; `lint_pack.py`recusa-se a enviar se a fechadura do alvo não estiver de acordo com a do pacote `VERSION`É assim que ...`npm`- Não .`Cargo`, e `pyproject.toml`sobrevivem a 10 anos de guerra, nada nos agentes muda as regras.

**Single source for cross-tool distribution.**Nx navios um `nx ai-setup`que descreve`AGENTS.md`- Não .`CLAUDE.md`- Não .`.cursor/rules/`- Não .`.github/copilot-instructions.md`O pacote deve fazer o mesmo; o instalador emite os links de sim (`ln -s AGENTS.md CLAUDE.md`A forma de forjar o pacote para apoiar uma ferramenta sobre outra é um modo de falha.

**`uninstall.sh` that refuses on non-trivial state.**Desinstalar o pacote não deve excluir os dados do utilizador `agent_state.json`- Não .`task_board.json`, ou `outputs/`O desinstalador remove os esquemas, scripts, documentos e...`AGENTS.md`(com `--keep-agents-md`O Estado pertence ao utilizador, o pacote não o possui.

**Skill-as-publishable. SkillKit-style distribution.**As embarcações de pacote como habilidade do SkillKit: `skillkit install agent-workbench-pack`O pacote repo é a fonte da verdade; o SkillKit é o canal de distribuição. O bloqueio do vendedor desmorona; as sete superfícies permanecem as mesmas.

## Usá-lo

Três lugares os navios de embalagem:

- **As a directory you drop into a repo.** `cp -r outputs/agent-workbench-pack /path/to/repo`- Não .
- **As a public template repo.**Forca e personalização, com `VERSION`Controlar a deriva.
- **As a SkillKit skill.**Conectado ao seu produto de agente para que um único comando o descreva.

O pacote é a receita, cada instalação é uma porção.

## Envia-o

`outputs/skill-workbench-pack.md`gera um pacote de projetos: regras aprimoradas para a história da equipa, globos de alcance combinados com o repo, dimensões rubricas estendidas com uma entrada específica de domínio.

## Exercícios

1. Decida qual o quinto documento opcional merece a promoção para o pacote canônico.
2. Reescrever o instalador como Python com um `--dry-run`Compare a ergonomia com a bash.
3. Adicionar um`bin/uninstall.sh`O que é considerado não trivial?
4. Adicionar um`lint_pack.py`que falha quando o pacote se desloca de `VERSION`Entregue-o para a CI para o repo do grupo.
5. Autor do manual de migração de um banco de trabalho rolado à mão para este pacote.

## Termos-chave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Workbench pack | "The starter kit" | A versioned directory carrying all seven surfaces |
| Installer | "Setup script" | `bin/install.sh` that lays the pack down idempotently |
| Pack version | "VERSION" | Major bumps for schema/script changes, patch for doc-only |
| Drop-in pack | "cp -r and go" | Pack works without per-repo customization on day one |
| Forkable template | "GitHub template" | Public repo that GitHub's "Use this template" can clone from |

## Mais leitura

- Fases 14 · 31 a 14 · 41  cada superfície que este pacote agrega
- [SkillKit](https://github.com/rohitg00/skillkit) instalar esta habilidade em 32 agentes de IA
- [Nx Blog, Teach Your AI Agent How to Work in a Monorepo](https://nx.dev/blog/nx-ai-agent-skills) Gerador de fonte única em seis ferramentas
- [agents.md — the open spec](https://agents.md/) o que o roteador da sua embalagem deve implementar
- [HKUDS/OpenHarness](https://github.com/HKUDS/OpenHarness) Implementação de referência de um equivalente de embalagem
- [andrewgarst/agentic_harness](https://github.com/andrewgarst/agentic_harness) Referência com back-up redis com suite eval
- [Augment Code, A good AGENTS.md is a model upgrade](https://www.augmentcode.com/blog/how-to-write-good-agents-dot-md-files) embalagem de documentos bar de qualidade
- [Anthropic, Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Anthropic, Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Fase 14 · 30  Desenvolvimento de agente orientado para avaliação que consome o portal de verificação da embalagem
- Fase 14 · 41  o índice de referência antes/após esta embalagem melhora em
