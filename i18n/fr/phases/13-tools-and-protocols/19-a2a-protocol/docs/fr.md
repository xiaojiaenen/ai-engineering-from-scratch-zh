# A2A  Protocole agent-agent

> MCP est agent-outil. A2A (Agent2Agent) est un protocole ouvert permettant aux agents opaques construits sur différents cadres de collaborer. Sorti par Google en avril 2025, donné à la Fondation Linux en juin 2025, atteint la version 1.0 en avril 2026 avec plus de 150 supporters, notamment AWS, Cisco, Microsoft, Salesforce, SAP et ServiceNow. Il a absorbé l'ACP d'IBM et a ajouté l'extension des paiements AP2. Cette leçon traverse la carte d'agent, le cycle de vie de la tâche et les deux liaisons de transport.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Objectifs d'apprentissage

- Distinguer les cas d'utilisation de l'agent à l'outil (MCP) des cas d'utilisation de l'agent à l'agent (A2A).
- Publier une carte d' agent à `/.well-known/agent.json`avec des compétences et des métadonnées de point final.
- Suivez le cycle de vie de la tâche (s' est soumis → en cours de travail → requis → terminé / échoué / annulé / rejeté).
- Utilisez les messages avec des pièces (texte, fichier, données) et des objets comme sorties.

## Le problème

Un agent de service client doit déléguer la rédaction de rapports à un agent spécialisé en rédaction.

- L'API REST personnalisée fonctionne, mais chaque couplage est unique.
- Une base de code partagée, exige que les deux agents exécutent le même cadre.
- MCP: pas adapté: MCP est pour appeler des outils, pas pour deux agents collaborant tout en préservant le raisonnement interne opaque de chaque agent.

A2A remplit le vide. Il modélise l'interaction en tant qu'agent envoyer une tâche à un autre, avec un cycle de vie, des messages et des objets. L'état interne de l'agent appelé reste opaque  l'appelant ne voit que les transitions de l'état de tâche et les sorties éventuelles.

A2A est le protocole "laissez les agents à travers les cadres s'entretenir" qui ne remplace pas le MCP, les deux sont complémentaires.

## Le concept

### Agent Card

Chaque agent conforme aux A2A publie une carte à l' adresse `/.well-known/agent.json`- Le numéro de la liste:

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

La découverte est basée sur l'URL: récupérer la carte, apprendre l'URL du point d'extrémité A2A, énumérer les compétences.

### Carte d'agent signée (AP2)

L'extension AP2 (septembre 2025) ajoute des signatures cryptographiques aux cartes d'agent. Un éditeur signe sa propre carte avec un JWT; les consommateurs vérifient.

### Cycle de vie des tâches

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

Les clients commencent par `tasks/send`Les agents appelés passent par les États; les clients s'abonnent aux mises à jour de l'État via SSE ou sondage.

### Messages et parties

Un message contient une ou plusieurs parties:

- `text` contenu clair.
- `file` base64 avec mimeType.
- `data` chargement utile JSON typé (entrée structurée pour l'agent appelé).

Exemple:

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

### Les objets

Les sorties sont des artifacts, pas des chaînes brutes.

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Les objets peuvent être diffusés en morceaux.

### Deux obligations de transport

1. **JSON-RPC over HTTP.** `/a2a`Point final, POST pour les demandes, SSE optionnel pour le streaming.
2. **gRPC.**Pour les environnements d'entreprise où le gRPC est natif.

Les deux liaisons ont la même forme logique de message.

### Préservation de l'ouverture

Un principe de conception clé: l'état interne de l'agent appelé est opaque. L'appelant voit l'état de la tâche et les artefacts. La chaîne de pensée de l'agent appelé, ses appels à l'outil, sa délégation de sous-agent  sont tous invisibles. Cela est différent de MCP, où les appels à l'outil sont transparents.

Rationale: A2A permet aux concurrents de collaborer sans révéler les informations internes. A2A peut être "appeler cet agent de service client" sans que l'appelant apprenne comment cet agent implemente le service.

### L'année

- **2025-04-09.**Google annonce A2A.
- **2025-06-23.**Donné à la Fondation Linux.
- **2025-08.**Il absorbe l'ACP d'IBM.
- **2025-09.**Les navires de l'extension AP2 (paiements par agent).
- **2026-04.**V1.0 est sorti avec plus de 150 organisations de soutien.

### Relation avec le PCM

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

Utilisez MCP lorsque vous voulez invoquer un outil spécifique. Utilisez A2A lorsque vous voulez déléguer une tâche entière à un autre agent. De nombreux systèmes de production utilisent les deux: un agent utilise MCP pour sa couche d'outils et A2A pour sa couche de collaboration.

```figure
a2a-task-lifecycle
```

## Utilisez-le

`code/main.py`Il implique un harnais A2A minimal: un agent de recherche publie sa carte, un agent de rédaction reçoit une`tasks/send`avec des parties comprenant un PDF et une instruction de texte, les transitions à travers le travail → input_required → working → terminées, et renvoie un artefact de texte.

À quoi regarder:

- La forme de la carte agent JSON.
- Célébration des tâches et transitions d'état.
- Messages avec des pièces de type mixte.
- Réglage de la branche à la mi-taxe.
- Le retour de l'artefact à la fin.

## La faire partir

Cette leçon produit `outputs/skill-a2a-agent-spec.md`. Étant donné qu'un nouvel agent doit être appelé par d'autres agents, la compétence produit le JSON de la carte agent, le schéma de compétences et le schéma des points d'extrémité.

## Exercices

1. On court .`code/main.py`.Récouvrez l'ensemble du cycle de vie de la tâche, y compris la pause requise pour l'entrée lorsque l'agent appelé demande une clarification.

2. Ajouter une carte d'agent signée, signer avec HMAC sur le JSON canonique de la carte, écrire un vérificateur et confirmer qu'il échoue sur une carte mutée.

3. Implementation de flux de tâches: l'agent de rédaction émet trois fragments d'artefact supplémentaires sur SSE et l'appelant les accumule.

4. Conceptionner un agent A2A qui embrasse un serveur MCP. Mape chaque outil MCP à une compétence A2A. Notez les compromis  quelle opacité est perdue?

5. Lisez l'annonce d'A2A v1.0 et identifiez la seule fonctionnalité qui n'est pas encore mise en œuvre par aucun cadre à partir d'avril 2026.

## Les termes clés

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

## Pour en savoir plus

- [a2a-protocol.org](https://a2a-protocol.org/latest/) spécification canonique A2A
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) Implémentations de référence et KDD
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) Transfert de gouvernance en juin 2025
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) feuille de route et dynamique des partenaires
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) note de libération v1.0 et orientation rétrocompatible
