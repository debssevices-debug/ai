"""Claude through the local Claude Code CLI: your Claude subscription, no API key.

This is what T3 Code does too: it drives the ``claude`` program installed on
the machine, which is already logged in with the user's Claude plan. Each agent
step runs ``claude -p`` once in headless mode:

* every built-in Claude Code tool is switched off (``--tools ""``): Claude only
  *decides*; NEC's own tools execute, under NEC's permissions;
* the whole step (rules, available tools, conversation so far) goes in on
  stdin, which avoids the Windows command-line length limit;
* Claude answers with one JSON object: tool calls to make, or the final answer.

Requirements: Claude Code installed and logged in once (run ``claude``, then
``/login``). Meant for personal use on your own machine, like T3 Code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from typing import Any

from nec_ai.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    LLMUnavailableError,
    Message,
    ToolCall,
)
from nec_ai.tools.base import ToolSpec

logger = logging.getLogger("nec.llm")

ENGINE_PROMPT = (
    "You are the reasoning engine of NEC, a personal agent. You cannot run any "
    "tool yourself: you decide, NEC executes. Follow the instructions given in "
    "the input and always reply with a single JSON object, nothing else."
)

PROTOCOL = """\
# FORMAT DE RÉPONSE (obligatoire)
Tu ne peux pas exécuter d'outil toi-même : tu décides, NEC exécute et te renvoie
les résultats à l'étape suivante. Réponds UNIQUEMENT avec un objet JSON, sans
texte avant ni après, sans bloc de code :
{"tool_calls": [{"name": "<outil>", "arguments": {...}}], "answer": ""}
- Pour agir : mets un ou plusieurs appels dans "tool_calls" (plusieurs quand ils
  sont indépendants) ; "answer" peut contenir une phrase courte d'annonce.
- Pour répondre définitivement : "tool_calls": [] et "answer" = ta réponse
  complète pour l'utilisateur (Markdown autorisé)."""

FINAL_ONLY = """\
# FORMAT DE RÉPONSE (obligatoire)
Aucun outil n'est disponible pour cette étape. Réponds UNIQUEMENT avec l'objet
JSON {"tool_calls": [], "answer": "<ta réponse complète>"}."""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ClaudeCodeProvider(LLMProvider):
    name = "claude_code"

    def __init__(self, model: str = "", executable: str = "claude") -> None:
        super().__init__(model or "")
        self.executable = executable or "claude"

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
    ) -> LLMResponse:
        prompt = render_prompt(messages, tools)
        args = [
            "-p",
            "--output-format",
            "json",
            "--tools",
            "",
            "--system-prompt",
            ENGINE_PROMPT,
            "--no-session-persistence",
            "--max-turns",
            "2",
        ]
        if self.model:
            args += ["--model", self.model]

        returncode, stdout, stderr = await self._run(args, prompt)
        return parse_cli_output(returncode, stdout, stderr, bool(tools))

    async def _run(self, args: list[str], stdin: str) -> tuple[int, str, str]:
        command = _command(self.executable) + args
        # An empty working directory: no project CLAUDE.md or settings leak in.
        with tempfile.TemporaryDirectory(prefix="nec-claude-") as workdir:
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workdir,
                    env=os.environ.copy(),
                )
            except FileNotFoundError as exc:
                raise LLMError(
                    "Claude Code n'est pas installé ou introuvable. Installe-le "
                    "(https://claude.com/claude-code), lance `claude` une fois et "
                    "connecte-toi avec /login, ou indique son chemin dans "
                    "CLAUDE_CODE_PATH."
                ) from exc
            try:
                out, err = await process.communicate(stdin.encode("utf-8"))
            except asyncio.CancelledError:
                process.kill()
                raise
        return (
            process.returncode or 0,
            out.decode("utf-8", errors="replace"),
            err.decode("utf-8", errors="replace"),
        )


def _command(executable: str) -> list[str]:
    """Resolve the CLI; Windows npm installs are .cmd files that need cmd.exe."""
    resolved = shutil.which(executable) or executable
    if os.name == "nt" and resolved.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", resolved]
    return [resolved]


def render_prompt(messages: list[Message], tools: list[ToolSpec] | None) -> str:
    parts: list[str] = [PROTOCOL if tools else FINAL_ONLY]
    if tools:
        listing = [
            f"- {t.name} : {t.description}\n  paramètres (JSON Schema) : "
            f"{json.dumps(t.parameters, ensure_ascii=False)}"
            for t in tools
        ]
        parts.append("# OUTILS DISPONIBLES\n" + "\n".join(listing))

    system = "\n\n".join(m.content for m in messages if m.role == "system")
    if system:
        parts.append("# CONSIGNES DE L'AGENT\n" + system)

    transcript: list[str] = []
    for m in messages:
        if m.role == "user":
            transcript.append(f"[UTILISATEUR]\n{m.content}")
        elif m.role == "assistant":
            text = m.content
            for call in m.tool_calls:
                text += (
                    f"\n(appel d'outil : {call.name} "
                    f"{json.dumps(call.arguments, ensure_ascii=False)})"
                )
            transcript.append(f"[TOI, NEC]\n{text.strip()}")
        elif m.role == "tool":
            transcript.append(f"[RÉSULTAT DE {m.name}]\n{m.content}")
    parts.append("# CONVERSATION\n" + "\n\n".join(transcript))
    parts.append(
        "Quelle est ta prochaine étape ? Réponds avec l'objet JSON uniquement."
    )
    return "\n\n".join(parts)


def parse_cli_output(
    returncode: int, stdout: str, stderr: str, tools_allowed: bool
) -> LLMResponse:
    envelope: dict[str, Any] | None = None
    try:
        envelope = (
            json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else None
        )
    except (json.JSONDecodeError, IndexError):
        envelope = None

    if envelope is None:
        raise _cli_error(stderr or stdout or f"exit code {returncode}")
    result = str(envelope.get("result") or "")
    if envelope.get("is_error") or returncode != 0:
        raise _cli_error(result or stderr or str(envelope.get("subtype", "error")))

    decision = _extract_json(result)
    if decision is None:
        # The model answered in plain text: take it as the final answer.
        logger.debug("claude_code: non-JSON reply taken as final answer")
        return LLMResponse(text=result.strip())

    calls: list[ToolCall] = []
    if tools_allowed:
        for raw in decision.get("tool_calls") or []:
            if isinstance(raw, dict) and raw.get("name"):
                arguments = raw.get("arguments")
                calls.append(
                    ToolCall(
                        name=str(raw["name"]),
                        arguments=arguments if isinstance(arguments, dict) else {},
                    )
                )
    usage = envelope.get("usage") or {}
    return LLMResponse(
        text=str(decision.get("answer") or "").strip(),
        tool_calls=calls,
        usage={
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
        },
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _cli_error(detail: str) -> LLMError:
    text = " ".join(detail.split())[:300]
    lowered = text.lower()
    if "login" in lowered or "api key" in lowered or "authenticat" in lowered:
        return LLMError(
            "Claude Code n'est pas connecté à ton compte. Ouvre un terminal, lance "
            f"`claude`, tape /login et suis les étapes. (Détail : {text})"
        )
    if "limit" in lowered and ("usage" in lowered or "reached" in lowered):
        # Subscription usage window: waiting seconds will not help.
        return LLMUnavailableError(
            f"Limite d'utilisation de ton abonnement Claude atteinte ({text}). "
            "Elle se réinitialise après quelques heures.",
            retry_after=3600,
            kind="rate_limit",
        )
    if "overloaded" in lowered or "529" in lowered:
        return LLMUnavailableError(f"Claude surchargé : {text}", kind="overloaded")
    return LLMError(f"Claude Code a échoué : {text}")
