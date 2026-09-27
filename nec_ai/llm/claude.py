"""Anthropic Claude through the official ``anthropic`` SDK (Messages API).

* Adaptive thinking is on: Claude decides how much to reason per step. Its
  thinking blocks are passed back unchanged on the next turn, as the API
  requires within one conversation.
* Server-side refusal fallbacks are enabled (``fallbacks: "default"``): if a
  safety classifier declines a request, the API re-runs it on the recommended
  fallback model inside the same call instead of returning a refusal.
* The SDK's own retries are off (``max_retries=0``): the agent's retry loop
  already waits for ``retry-after`` and shows the wait to the user.
* No ``temperature``: current Claude models reject sampling parameters.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic

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

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16_000


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(
        self,
        api_key: str,
        model: str = "",
        *,
        fallbacks: bool = True,
        max_tokens: int = MAX_TOKENS,
        client: Any = None,
    ) -> None:
        super().__init__(model or DEFAULT_MODEL)
        self.fallbacks = fallbacks
        self.max_tokens = max_tokens
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key, max_retries=0
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,  # ignored: rejected by current models
    ) -> LLMResponse:
        system, claude_messages = to_claude_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": claude_messages,
            "thinking": {"type": "adaptive"},
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]
        if self.fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        try:
            response = await self._client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMError(
                "Claude a refusé la clé API (401). Vérifie ANTHROPIC_API_KEY dans le .env."
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError(
                f"Claude : accès refusé pour cette clé ({exc.message})."
            ) from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(
                f"Claude : modèle {self.model!r} introuvable pour cette clé "
                "(corrige LLM_MODEL ou laisse-le vide)."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailableError(
                f"Claude rate limit (429): {exc.message}",
                retry_after=_retry_after(exc),
                kind="rate_limit",
            ) from exc
        except anthropic.InternalServerError as exc:  # 5xx, including 529 overloaded
            raise LLMUnavailableError(
                f"Claude error {exc.status_code}: {exc.message}",
                retry_after=_retry_after(exc),
                kind="overloaded",
            ) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code == 402:
                raise LLMError(
                    "Claude : problème de facturation sur le compte (crédits épuisés ?)."
                ) from exc
            raise LLMError(f"Claude error {exc.status_code}: {exc.message}") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMUnavailableError(
                "Claude did not answer in time", kind="timeout"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailableError(f"Claude unreachable: {exc}") from exc

        return from_claude_response(response)


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after", ""))
    except (TypeError, ValueError, AttributeError):
        return None


def to_claude_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Translate neutral messages. Tool results of one turn share one user message."""
    system: list[str] = []
    out: list[dict[str, Any]] = []

    def user_blocks() -> list[dict[str, Any]]:
        """The content list of a trailing user message, created if needed."""
        if out and out[-1]["role"] == "user":
            content = out[-1]["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
                out[-1]["content"] = content
            return content
        out.append({"role": "user", "content": []})
        return out[-1]["content"]

    for m in messages:
        if m.role == "system":
            system.append(m.content)
        elif m.role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content,
            }
            if m.content.startswith("ERROR:"):
                block["is_error"] = True
            user_blocks().append(block)
        elif m.role == "assistant":
            if m.provider == ClaudeProvider.name and isinstance(m.provider_data, list):
                # Claude's own blocks, thinking included, echoed back unchanged.
                out.append({"role": "assistant", "content": m.provider_data})
                continue
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for call in m.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            out.append(
                {
                    "role": "assistant",
                    "content": blocks or [{"type": "text", "text": "…"}],
                }
            )
        else:
            if out and out[-1]["role"] == "user":
                # e.g. an agent note right after tool results: same user turn.
                user_blocks().append({"type": "text", "text": m.content})
            else:
                out.append({"role": "user", "content": m.content})
    return "\n\n".join(s for s in system if s), out


def from_claude_response(response: Any) -> LLMResponse:
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise LLMError(
            "Claude a refusé cette demande"
            + (f" (catégorie : {category})" if category else "")
            + ". Reformule-la ou passe par un autre modèle."
        )

    texts: list[str] = []
    calls: list[ToolCall] = []
    echo: list[dict[str, Any]] = []
    for block in response.content:
        if block.type == "fallback":
            continue  # audit marker only; not needed in the history
        echo.append(block.model_dump(mode="json", exclude_none=True))
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            arguments = block.input if isinstance(block.input, dict) else {}
            calls.append(ToolCall(name=block.name, arguments=arguments, id=block.id))

    usage: dict[str, int] = {}
    if getattr(response, "usage", None) is not None:
        usage = {
            "input_tokens": response.usage.input_tokens or 0,
            "output_tokens": response.usage.output_tokens or 0,
        }
    return LLMResponse(
        text="".join(texts).strip(), tool_calls=calls, usage=usage, provider_data=echo
    )
