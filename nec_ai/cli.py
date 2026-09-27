"""Command line: talk to NEC in the terminal.

python -m nec_ai                 interactive chat
python -m nec_ai ask "question"  one question, one answer
python -m nec_ai tools           list the available tools
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any
from urllib.parse import urlsplit

from nec_ai.agent.events import AgentEvent, EventType
from nec_ai.config.settings import get_settings
from nec_ai.llm import LLMError
from nec_ai.observability.logging import setup_logging
from nec_ai.tools.registry import ConfirmationRequest

EXIT_WORDS = {"exit", "quit", "q", "bye", "au revoir", "stop"}


class EventPrinter:
    """Shows what the agent is doing, one short line per step."""

    def __init__(self, verbose: bool = False, plain: bool = False) -> None:
        self.verbose = verbose
        self.plain = plain

    def _icon(self, emoji: str, fallback: str) -> str:
        return fallback if self.plain else emoji

    def __call__(self, event: AgentEvent) -> None:
        line = self.describe(event)
        if line:
            print(line, flush=True)

    def describe(self, event: AgentEvent) -> str | None:
        d: dict[str, Any] = event.data
        t = event.type
        if t is EventType.STARTED:
            return self._icon("🤖", "[agent]") + " Compris."
        if t is EventType.THINKING:
            if d.get("iteration", 1) == 1 and not self.verbose:
                return self._icon("🧠", "[..]") + " Analyse..."
            return (
                self._icon("🧠", "[..]") + f" Réflexion (étape {d.get('iteration')})..."
            )
        if t is EventType.PLAN:
            steps = "\n".join(
                f"     {i}. {s}" for i, s in enumerate(d.get("steps", []), 1)
            )
            title = "Plan révisé" if d.get("revised") else "Plan"
            return self._icon("📋", "[plan]") + f" {title} :\n{steps}"
        if t is EventType.WAITING:
            return self._icon("⏳", "[wait]") + " " + self._waiting(d)
        if t is EventType.MESSAGE:
            return self._icon("💬", "[msg]") + f" {d.get('text', '')}"
        if t is EventType.TOOL_STARTED:
            return self._tool_line(d.get("tool", ""), d.get("arguments") or {})
        if t is EventType.TOOL_COMPLETED:
            if not self.verbose:
                return None
            return (
                self._icon("✅", "[ok]")
                + f" {d.get('tool')} ({d.get('duration', 0):.1f}s)"
            )
        if t is EventType.TOOL_FAILED:
            error = d.get("error") or "échec"
            return self._icon("⚠️", "[!]") + f" {d.get('tool')} : {error}"
        if t is EventType.ERROR:
            return self._icon("❌", "[x]") + f" {d.get('error')}"
        return None

    @staticmethod
    def _waiting(d: dict[str, Any]) -> str:
        seconds = f"{d.get('seconds', 0):.0f} s"
        reason = d.get("reason")
        if reason == "throttle":
            return f"Pause de {seconds} pour rester sous le quota du modèle..."
        if reason == "rate_limit":
            return f"Quota du modèle atteint, nouvel essai dans {seconds}..."
        if reason == "overloaded":
            return f"Modèle surchargé, nouvel essai dans {seconds}..."
        return f"Modèle indisponible, nouvel essai dans {seconds}..."

    def _tool_line(self, tool: str, args: dict[str, Any]) -> str | None:
        if tool == "update_plan":
            return None
        if tool == "web_search":
            return (
                self._icon("🔎", "[search]")
                + f" Recherche web : « {args.get('query', '')} »"
            )
        if tool == "fetch_url":
            return self._icon("🌐", "[web]") + f" Consultation de {_short_url(args)}"
        return self._icon("🛠️", "[tool]") + f" {tool}"


def _short_url(args: dict[str, Any]) -> str:
    """``www.apple.com/fr/iphone-18-pro`` rather than just the host."""
    url = str(args.get("url", ""))
    parts = urlsplit(url)
    if not parts.hostname:
        return url
    path = parts.path.rstrip("/")
    text = parts.hostname + (path if len(path) <= 50 else path[:47] + "...")
    start = args.get("start")
    return text + (f" (suite, à partir de {start})" if start else "")


async def ask_confirmation(request: ConfirmationRequest) -> bool:
    print(
        f"\n⚠️  Confirmation requise : {request.summary}\n   Raison : {request.reason}"
    )
    answer = await asyncio.to_thread(input, "   Autoriser ? [o/N] ")
    return answer.strip().lower() in {"o", "oui", "y", "yes"}


async def _ask(question: str, printer: EventPrinter) -> int:
    from nec_ai.app import build_agent

    agent = build_agent()
    answer = await agent.ask(question, confirm=ask_confirmation, on_event=printer)
    print("\n" + answer)
    return 0


async def _chat(printer: EventPrinter) -> int:
    from nec_ai.app import build_agent

    agent = build_agent()
    print("NEC AI — tape ta demande (ou 'exit' pour quitter).\n")
    while True:
        try:
            text = await asyncio.to_thread(input, "Vous > ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text.strip().lower() in EXIT_WORDS:
            return 0
        if not text.strip():
            continue
        answer = await agent.ask(
            text, session_id="cli", confirm=ask_confirmation, on_event=printer
        )
        print(f"\nNEC > {answer}\n")


async def _remote(printer: EventPrinter, server: str | None, key: str | None) -> int:
    """Chat with a NEC server (e.g. on a VPS) instead of a local agent."""
    from nec_ai.client import RemoteClient, RemoteError

    settings = get_settings()
    url = server or settings.server_url
    api_key = key or (
        settings.nec_api_key.get_secret_value() if settings.nec_api_key else None
    )
    async with RemoteClient(url, api_key) as client:
        try:
            info = await client.health()
        except RemoteError as exc:
            print(f"❌ {exc}", file=sys.stderr)
            return 2
        print(
            f"NEC AI — connecté à {url} (v{info.get('version', '?')}). 'exit' pour quitter.\n"
        )
        while True:
            try:
                text = await asyncio.to_thread(input, "Vous > ")
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if text.strip().lower() in EXIT_WORDS:
                return 0
            if not text.strip():
                continue
            answer = ""
            try:
                async for event in client.stream(text, session_id="cli"):
                    printer(event)
                    if event.type is EventType.CONFIRMATION_REQUIRED:
                        request = ConfirmationRequest(
                            tool=event.data.get("tool", ""),
                            arguments=event.data.get("arguments") or {},
                            reason=event.data.get("reason", ""),
                            summary=event.data.get("summary", ""),
                            id=event.data.get("id", ""),
                        )
                        approved = await ask_confirmation(request)
                        await client.confirm(request.id, approve=approved)
                    elif event.type is EventType.FINAL:
                        answer = event.data.get("answer", "")
            except RemoteError as exc:
                print(f"❌ {exc}")
                continue
            print(f"\nNEC > {answer}\n")


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn

    from nec_ai.api.security import auth_problem
    from nec_ai.api.server import create_app

    settings = get_settings()
    if host:
        settings = settings.model_copy(update={"api_host": host})
    if port:
        settings = settings.model_copy(update={"api_port": port})
    problem = auth_problem(settings)
    if problem:
        print(f"Le serveur ne démarre pas : {problem}", file=sys.stderr)
        return 2
    setup_logging(settings.log_level)
    print(f"NEC API sur http://{settings.api_host}:{settings.api_port}")
    uvicorn.run(
        create_app(settings),
        host=settings.api_host,
        port=settings.api_port,
        log_level="warning",
        proxy_headers=True,
    )
    return 0


def _new_key() -> int:
    import secrets

    key = secrets.token_urlsafe(32)
    print(key)
    print(
        "\nServeur : ajoute-la à API_KEYS dans son .env."
        "\nClient  : mets-la dans NEC_API_KEY. Ne la partage pas.",
        file=sys.stderr,
    )
    return 0


def _tools() -> int:
    from nec_ai.app import build_registry

    for spec in build_registry(get_settings()).specs():
        print(f"- {spec.name}: {spec.description.splitlines()[0]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nec", description="NEC AI, agent personnel")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="afficher chaque étape"
    )
    parser.add_argument("--plain", action="store_true", help="sans emoji")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("chat", help="conversation interactive (défaut)")
    ask = sub.add_parser("ask", help="poser une seule question")
    ask.add_argument("question", nargs="+")
    sub.add_parser("tools", help="lister les outils disponibles")
    serve = sub.add_parser("serve", help="lancer l'API serveur")
    serve.add_argument("--host", help="adresse d'écoute (défaut : API_HOST)")
    serve.add_argument("--port", type=int, help="port (défaut : API_PORT)")
    remote = sub.add_parser("remote", help="discuter avec un serveur NEC distant")
    remote.add_argument("--server", help="URL du serveur (défaut : SERVER_URL)")
    remote.add_argument("--key", help="clé API (défaut : NEC_API_KEY)")
    sub.add_parser("new-key", help="générer une clé API solide")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # The progress lines already show waits and errors; the timestamped logs
    # would repeat them, so they only appear with -v.
    setup_logging("INFO" if args.verbose else "CRITICAL")
    printer = EventPrinter(verbose=args.verbose, plain=args.plain)

    try:
        if args.command == "tools":
            return _tools()
        if args.command == "new-key":
            return _new_key()
        if args.command == "serve":
            return _serve(args.host, args.port)
        if args.command == "remote":
            return asyncio.run(_remote(printer, args.server, args.key))
        if args.command == "ask":
            return asyncio.run(_ask(" ".join(args.question), printer))
        return asyncio.run(_chat(printer))
    except LLMError as exc:
        print(f"Configuration incomplète : {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
