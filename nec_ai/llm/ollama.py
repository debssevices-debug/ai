"""Local models through Ollama: no API key, no quota, nothing leaves the PC.

Uses Ollama's native ``/api/chat`` endpoint rather than its OpenAI-compatible
one, because only the native API lets us set the context window (``num_ctx``):
Ollama's small default silently truncates the system prompt and tool results.

Setup (Windows): install from https://ollama.com, then ``ollama pull qwen3:8b``.
The model must support tool calling (qwen3, qwen2.5, llama3.1, mistral-nemo...).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

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

#: Good tool calling, runs on a PC with ~8 GB of free RAM (faster with a GPU).
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_URL = "http://localhost:11434"

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str = "",
        base_url: str = DEFAULT_URL,
        num_ctx: int = 16_384,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(model or DEFAULT_MODEL)
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.num_ctx = num_ctx
        self._client = client

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
    ) -> LLMResponse:
        options: dict[str, Any] = {"num_ctx": self.num_ctx}
        if temperature is not None:
            options["temperature"] = temperature
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_ollama_messages(messages),
            "stream": False,
            "options": options,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        url = f"{self.base_url}/api/chat"
        try:
            if self._client is not None:
                response = await self._client.post(url, json=payload)
            else:
                # Local generation can be slow on CPU: the agent's own LLM
                # timeout bounds the call, so no extra limit here.
                async with httpx.AsyncClient(timeout=None) as client:
                    response = await client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Ollama ne répond pas sur {self.base_url}. Installe-le depuis "
                "https://ollama.com et vérifie qu'il est lancé (icône dans la barre "
                "des tâches, ou `ollama serve`)."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Ollama error: {type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise _http_error(response, self.model)

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMUnavailableError("Ollama returned an invalid response") from exc
        return from_ollama_response(data)


def _http_error(response: httpx.Response, model: str) -> LLMError:
    try:
        detail = str(response.json().get("error", ""))
    except ValueError:
        detail = response.text[:200]
    lowered = detail.lower()
    if response.status_code == 404 or "not found" in lowered:
        return LLMError(
            f"Le modèle {model!r} n'est pas installé dans Ollama. "
            f"Lance : ollama pull {model}"
        )
    if "does not support tools" in lowered:
        return LLMError(
            f"Le modèle {model!r} ne sait pas utiliser d'outils. Choisis-en un qui "
            "le supporte, par exemple LLM_MODEL=qwen3:8b ou llama3.1:8b."
        )
    if response.status_code >= 500:
        return LLMUnavailableError(f"Ollama error {response.status_code}: {detail}")
    return LLMError(f"Ollama error {response.status_code}: {detail}")


def to_ollama_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {"role": "tool", "content": m.content, "tool_name": m.name or ""}
            )
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [
                        {"function": {"name": c.name, "arguments": c.arguments}}
                        for c in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def from_ollama_response(data: dict[str, Any]) -> LLMResponse:
    message = data.get("message") or {}
    calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        calls.append(ToolCall(name=str(function.get("name", "")), arguments=arguments))

    text = _THINK.sub("", str(message.get("content") or "")).strip()
    usage = {
        "input_tokens": int(data.get("prompt_eval_count") or 0),
        "output_tokens": int(data.get("eval_count") or 0),
    }
    return LLMResponse(text=text, tool_calls=calls, usage=usage)
