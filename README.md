# NEC AI

NEC est un **agent IA personnel**, dans l'esprit de JARVIS. Ce n'est pas un simple chatbot : il comprend une mission, fait un plan, choisit ses outils (recherche web, lecture de pages, navigateur, etc.), enchaîne plusieurs étapes, vérifie ce qu'il trouve et rend une réponse claire avec ses sources.

```
Vous > Trouve-moi les 3 meilleures solutions de centre de contact pour 50 agents et compare-les.
🤖 Compris.
🧠 Analyse...
📋 Plan :
     1. Identifier les solutions pertinentes
     2. Chercher les prix et fonctionnalités sur les sites officiels
     3. Comparer et recommander
🔎 Recherche web : « meilleures solutions centre de contact 50 agents 2026 »
🌐 Consultation de www.odigo.com
🌐 Consultation de www.genesys.com
...
NEC > | Solution | Prix / agent | Points forts | ... |
```

## Sommaire

1. [État du projet](#1-état-du-projet)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Configuration `.env`](#4-configuration-env)
5. [Lancement](#5-lancement)
6. [Utilisation](#6-utilisation)
7. [Ajouter un outil](#7-ajouter-un-outil)
8. [Ajouter un fournisseur LLM](#8-ajouter-un-fournisseur-llm)
9. [Ajouter un fournisseur de recherche, STT ou TTS](#9-ajouter-un-fournisseur-de-recherche-stt-ou-tts)
10. [Serveur API, client Windows, VPS](#10-serveur-api-client-windows-vps)
11. [Sécurité](#11-sécurité)
12. [Tests](#12-tests)

---

## 1. État du projet

| Étape | Contenu | État |
|---|---|---|
| 0 | Restructuration (package `nec_ai/`), nettoyage | ✅ |
| 1 | Configuration `.env`, logs, système d'outils + permissions | ✅ |
| 2 | Couche LLM : Gemini, OpenAI, faux LLM de test | ✅ |
| 3 | Boucle d'agent, planification, événements, traces, mémoire courte | ✅ |
| 4 | Recherche web multi-fournisseurs, lecture de pages, CLI → **MVP texte** | ✅ |
| 5 | Planification avancée | ✅ (outil `update_plan`, livré avec l'étape 3) |
| 6 | API REST FastAPI + streaming + authentification + rate limiting | ⏳ |
| 7 | Outils fichiers et terminal sécurisés | ⏳ |
| 8 | Navigateur branché sur le nouveau cœur | ⏳ |
| 9 | Mémoire long terme | ⏳ |
| 10 | Voix : LiveKit branché sur le nouveau cœur, fournisseurs STT/TTS, wake word | ⏳ |
| 11 | Client Windows, déploiement VPS | ⏳ |

L'agent vocal LiveKit d'origine fonctionne toujours tel quel (voir [Voix](#voix-livekit)). Son navigateur (Playwright, 27 outils) est conservé et sera branché sur le nouveau cœur à l'étape 8.

## 2. Architecture

```
nec_ai/
├── agent/
│   ├── core.py        boucle THINK → ACT → OBSERVE, limite d'itérations, erreurs
│   ├── planner.py     outil update_plan : plan écrit puis révisé par le LLM
│   ├── events.py      agent.started / thinking / plan / tool.* / final / error
│   └── context.py     mémoire de travail d'une tâche
├── llm/
│   ├── base.py        interface LLMProvider, Message, ToolCall, retries
│   ├── gemini.py      Google Gemini (défaut)
│   ├── openai.py      OpenAI ou tout endpoint compatible
│   ├── fake.py        LLM scripté pour les tests
│   └── prompts.py     prompt système en blocs (texte / voix)
├── tools/
│   ├── base.py        Tool, ToolResult, RiskLevel (SAFE / CONFIRMATION_REQUIRED / BLOCKED)
│   ├── registry.py    découverte + exécution sûre (validation, permissions, timeout)
│   ├── web_search/    abstraction + fournisseurs DuckDuckGo, Brave, Serper
│   ├── fetch.py       fetch_url : lire une page web en texte
│   └── browser/       navigateur Playwright (hérité, à brancher à l'étape 8)
├── memory/
│   └── short_term.py  historique par conversation
├── security/
│   ├── untrusted.py   marquage <page_data> du contenu externe
│   └── net.py         politique réseau (anti-SSRF)
├── observability/
│   ├── logging.py     logs horodatés
│   └── trace.py       une trace JSONL par requête
├── voice/
│   └── livekit_agent.py  agent vocal LiveKit (d'origine)
├── config/settings.py toute la configuration, lue depuis .env
├── app.py             assemblage : settings → LLM + outils → Agent
└── cli.py             interface terminal
```

Fonctionnement d'une requête :

```
Utilisateur ─► Agent ─► LLM : « que faire ? »
                 │         └─► appelle des outils (web_search, fetch_url, update_plan...)
                 │                 └─► ToolRegistry : validation → permission → exécution → résultat
                 │◄──────────────── résultat renvoyé au LLM (observation)
                 └─► ... jusqu'à une réponse sans appel d'outil = réponse finale
```

Principes :

- **Le LLM décide *quoi* faire, les outils exécutent *comment*.** Un outil ne parle jamais au LLM.
- **Tout passe par le registre.** C'est lui qui valide les arguments, applique les permissions, gère les timeouts et transforme toute exception en résultat lisible.
- **Rien n'est codé en dur.** Le LLM, le moteur de recherche et les limites se changent dans `.env`.
- **Le contenu externe est une donnée, jamais un ordre.** Pages web, résultats de recherche et fichiers sont encadrés par `<page_data>`.

## 3. Installation

Prérequis : **Python 3.11 à 3.14** et **[uv](https://docs.astral.sh/uv/)**.

Windows (PowerShell) :

```powershell
winget install --id=astral-sh.uv -e
git clone https://github.com/debssevices-debug/ai.git NEC_AI
cd NEC_AI
uv sync
copy .env.example .env
notepad .env
```

Linux / macOS :

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/debssevices-debug/ai.git NEC_AI && cd NEC_AI
uv sync
cp .env.example .env
```

## 4. Configuration `.env`

Seule une clé LLM est obligatoire. Tout le reste a une valeur par défaut sûre.

| Variable | Rôle | Défaut |
|---|---|---|
| `LLM_PROVIDER` | `gemini` ou `openai` | `gemini` |
| `LLM_MODEL` | modèle précis (vide = défaut du fournisseur) | `gemini-flash-latest` / `gpt-4.1-mini` |
| `GOOGLE_API_KEY` | clé Gemini ([aistudio.google.com](https://aistudio.google.com/apikey)) | — |
| `OPENAI_API_KEY` | clé OpenAI | — |
| `MAX_AGENT_ITERATIONS` | étapes maximum par demande | `20` |
| `SEARCH_PROVIDER` | `duckduckgo` (gratuit), `brave`, `serper` | `duckduckgo` |
| `SEARCH_API_KEY` | clé du fournisseur de recherche payant | — |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `TRACE_ENABLED` | trace JSONL de chaque requête dans `data/traces/` | `true` |

La liste complète, commentée, se trouve dans [`.env.example`](.env.example). **Ne commitez jamais `.env`** : il est exclu par `.gitignore`.

## 5. Lancement

```powershell
uv run nec              # conversation interactive
uv run nec ask "Quelles sont les dernières nouvelles concernant Microsoft ?"
uv run nec tools        # liste des outils disponibles
uv run nec -v           # affiche aussi les logs horodatés
uv run nec --plain      # sans emoji (anciens terminaux)
```

`python -m nec_ai` fonctionne aussi.

### Voix (LiveKit)

L'agent vocal d'origine (Gemini Live, voix française, navigateur) se lance comme avant. Il faut `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` et `GOOGLE_API_KEY` dans `.env` ou `.env.local`.

```powershell
uv run python -m nec_ai.voice.livekit_agent console   # dans le terminal
uv run python -m nec_ai.voice.livekit_agent dev       # connecté à LiveKit Cloud
```

## 6. Utilisation

Exemples de demandes :

- « Cherche-moi les dernières informations sur Odigo. »
- « Compare Aircall, Ringover et Diabolocom pour 50 agents, avec les prix. »
- « Va lire cette page et fais-moi un résumé : https://… »
- « Regarde plusieurs sources et donne-moi une synthèse sur … »
- « Continue » (la conversation garde le contexte des échanges précédents)

Pour chaque requête, une trace est écrite dans `data/traces/<date>/<id>.jsonl` (`USER_REQUEST → PLAN_CREATED → TOOL_SELECTED → TOOL_RESULT → FINAL_RESPONSE`). Elle permet de comprendre après coup pourquoi l'agent a fait telle action.

## 7. Ajouter un outil

1. Créez une classe dans `nec_ai/tools/` :

```python
from pydantic import BaseModel, Field
from nec_ai.tools.base import PermissionDecision, Tool, ToolContext, ToolResult


class WeatherTool(Tool):
    name = "weather"
    description = "Donner la météo actuelle d'une ville."
    untrusted_output = True          # la sortie vient d'Internet

    class Input(BaseModel):          # sert à la fois de schéma pour le LLM et de validation
        city: str = Field(..., description="Nom de la ville")

    def check_permission(self, args, ctx):   # optionnel : SAFE par défaut
        return PermissionDecision.safe()

    async def run(self, args: Input, ctx: ToolContext) -> ToolResult:
        ...
        return ToolResult.success(f"Il fait 21 °C à {args.city}.", data={"temp": 21})
```

2. Enregistrez-la dans `build_registry()` (`nec_ai/app.py`).
3. Ajoutez un test (voir `tests/test_tool_registry.py`).

Les permissions : `PermissionDecision.safe()`, `.confirm("raison")` (l'utilisateur doit accepter cet appel précis) ou `.block("raison")` (jamais exécuté).

## 8. Ajouter un fournisseur LLM

1. Créez `nec_ai/llm/mon_fournisseur.py` avec une sous-classe de `LLMProvider` qui implémente `complete(messages, tools, temperature)` et renvoie un `LLMResponse` (texte et/ou `tool_calls`).
2. Traduisez les erreurs : `LLMUnavailableError` pour ce qui peut être réessayé (timeout, 429, 5xx), `LLMError` pour le reste.
3. Ajoutez le cas dans `create_llm()` (`nec_ai/llm/__init__.py`) et la valeur dans `llm_provider` (`config/settings.py`).

`gemini.py` et `openai.py` servent d'exemples. Pour un modèle compatible OpenAI (OpenRouter, Mistral, serveur local), il suffit de régler `LLM_PROVIDER=openai` et `OPENAI_BASE_URL`.

## 9. Ajouter un fournisseur de recherche, STT ou TTS

- **Recherche** : sous-classe de `SearchProvider` (`nec_ai/tools/web_search/base.py`), avec une méthode `search()` qui renvoie des `SearchResult`. Ajoutez-la ensuite dans `create_provider()`. Brave et Serper, dans `providers.py`, sont de bons modèles.
- **STT / TTS / wake word** : les interfaces `STTProvider`, `TTSProvider` et `WakeWordDetector` arrivent à l'étape 10, sur le même principe.

## 10. Serveur API, client Windows, VPS

Architecture cible (étapes 6 et 11) :

```
Windows PC (client) ──HTTPS + clé API──► VPS : API NEC (FastAPI) ──► LLM / Web / Mémoire / Outils
                    ◄── flux d'événements (SSE) : agent.thinking, tool.started, agent.final…
```

Le cœur est déjà prêt pour ça. `Agent.run()` produit exactement les événements que l'API transmettra au client (`agent.started`, `agent.thinking`, `tool.started`, `tool.completed`, `confirmation.required`, `agent.final`, `agent.error`). Les commandes de lancement du serveur et le guide de déploiement VPS seront ajoutés à l'étape 6.

## 11. Sécurité

- Aucune clé dans le code. Les secrets sont lus depuis `.env` et masqués dans les logs (`SecretStr`).
- Permissions par outil : `SAFE`, `CONFIRMATION_REQUIRED` ou `BLOCKED`. Sans quelqu'un pour confirmer, une action risquée est refusée.
- Réseau (`fetch_url`, navigateur) : http/https uniquement. Les endpoints de métadonnées cloud et les adresses privées sont bloqués, y compris après résolution DNS et à chaque redirection. La taille téléchargée est limitée.
- Protection contre l'injection de prompt : le contenu externe est encadré par `<page_data>` et le prompt interdit d'y obéir.
- Limites : nombre d'étapes (`MAX_AGENT_ITERATIONS`), timeouts LLM et outils, taille des sorties d'outils, appels identiques non répétés.
- À venir avec l'API : authentification, rate limiting, écoute sur `127.0.0.1` par défaut.

## 12. Tests

```powershell
uv run pytest          # toute la suite (sans réseau ni clé API)
uv run ruff check .    # lint
uv run ruff format .   # formatage
```

Les tests du navigateur « live » se lancent automatiquement si Chromium est installé (`uv run playwright install chromium`). Sinon ils sont ignorés.

## Licence

MIT, voir [LICENSE](LICENSE).
