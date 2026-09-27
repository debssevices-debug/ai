# Déployer NEC sur un VPS

Architecture :

```
PC Windows (nec remote / future app)
        │  HTTPS + clé API
        ▼
Caddy (port 443, certificat HTTPS automatique)
        │  http://127.0.0.1:8000
        ▼
NEC API (nec serve, service systemd) ──► LLM · recherche web · mémoire · outils
```

NEC n'écoute que sur `127.0.0.1` : seul Caddy est exposé sur Internet, et toutes les routes `/v1` exigent une clé.

Prérequis : un VPS Ubuntu 24.04 (2 vCPU / 4 Go de RAM suffisent si le LLM n'est pas local), un nom de domaine qui pointe vers son IP (enregistrement DNS `A`, par exemple `nec.mondomaine.com`).

## 1. Sécuriser le serveur

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl ufw
sudo ufw allow OpenSSH
sudo ufw allow 80,443/tcp
sudo ufw enable
sudo adduser --disabled-password --gecos "" nec
```

## 2. Installer NEC (utilisateur `nec`)

```bash
sudo -iu nec
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
git clone https://github.com/debssevices-debug/ai.git NEC_AI
cd NEC_AI
uv sync
cp .env.example .env
uv run nec new-key          # copie la clé affichée
nano .env
```

Dans `.env` :

```
API_HOST=127.0.0.1
API_PORT=8000
API_KEYS=<la clé générée>
RATE_LIMIT_PER_MINUTE=30
LLM_PROVIDER=...            # voir étape 3
```

## 3. Choisir le LLM

- **Abonnement Claude, sans clé API (`LLM_PROVIDER=claude_code`)**, toujours en tant qu'utilisateur `nec` :
  ```bash
  curl -fsSL https://claude.ai/install.sh | bash
  claude          # puis /login : ouvre le lien affiché sur ton PC pour valider, puis /exit
  ```
  Usage **personnel uniquement** : n'ouvre pas ce serveur à d'autres personnes avec ton abonnement.
- **Clé API** : `LLM_PROVIDER=claude` + `ANTHROPIC_API_KEY`, ou `gemini` + `GOOGLE_API_KEY`.

Test rapide : `uv run nec ask "bonjour"`.

## 4. Lancer NEC comme service

`exit` pour revenir à ton utilisateur admin, puis crée `/etc/systemd/system/nec.service` :

```ini
[Unit]
Description=NEC AI API
After=network-online.target
Wants=network-online.target

[Service]
User=nec
WorkingDirectory=/home/nec/NEC_AI
ExecStart=/home/nec/.local/bin/uv run nec serve
Restart=on-failure
RestartSec=5
Environment=PATH=/home/nec/.local/bin:/usr/local/bin:/usr/bin:/bin
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nec
sudo systemctl status nec
journalctl -u nec -f        # logs en direct
```

## 5. HTTPS avec Caddy

```bash
sudo apt install -y caddy
sudo nano /etc/caddy/Caddyfile
```

```
nec.mondomaine.com {
    reverse_proxy 127.0.0.1:8000 {
        flush_interval -1      # le streaming des événements passe sans attendre
    }
}
```

```bash
sudo systemctl reload caddy
curl https://nec.mondomaine.com/health
```

## 6. Se connecter depuis le PC Windows

Dans le `.env` du PC :

```
SERVER_URL=https://nec.mondomaine.com
NEC_API_KEY=<la même clé que dans API_KEYS du serveur>
```

```powershell
uv run nec remote
```

L'affichage est le même qu'en local (🔎, 🌐, ✅). Les actions risquées te sont demandées sur le PC, et l'agent attend ta réponse.

## API en bref

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | état du serveur (sans clé) |
| GET | `/v1/tools` | outils disponibles |
| POST | `/v1/chat` | `{"message": "...", "session_id": "..."}` → réponse finale + événements |
| POST | `/v1/chat/stream` | idem, en direct (Server-Sent Events) |
| POST | `/v1/confirmations/{id}` | `{"approve": true}` pour autoriser une action risquée |
| DELETE | `/v1/sessions/{id}` | oublier une conversation |

Toutes les routes `/v1` demandent l'en-tête `Authorization: Bearer <clé>`.

```bash
curl -N https://nec.mondomaine.com/v1/chat/stream \
  -H "Authorization: Bearer $NEC_API_KEY" -H "Content-Type: application/json" \
  -d '{"message": "Dernières nouvelles de Microsoft ?"}'
```

## Mettre à jour

```bash
sudo -iu nec
cd NEC_AI && git pull && uv sync
exit
sudo systemctl restart nec
```

## Sécurité : récapitulatif

- NEC écoute uniquement en local ; seul Caddy (HTTPS) est exposé.
- Clés longues (`nec new-key`), comparées en temps constant, jamais journalisées. Le serveur refuse de démarrer sans clé.
- Limite de requêtes par clé (`RATE_LIMIT_PER_MINUTE`) et nombre de tâches simultanées (`API_MAX_CONCURRENT_RUNS`).
- Conversations isolées par clé. Une action risquée n'est jamais exécutée sans ta confirmation.
- Pas de documentation interactive (`/docs`) quand des clés sont configurées.
- Pour révoquer un accès, retire la clé de `API_KEYS` puis `sudo systemctl restart nec`.
