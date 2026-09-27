"""OpenAI (and any OpenAI-compatible endpoint) through the Chat Completions API."""

from __future__ import annotations

import json
import logging
from typing import Any

import openai

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

DEFAULT_MODEL = "gpt-4.1-mini"


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "",
        base_url: str | None = None,
        client: Any = None,
    ) -> None:
        super().__init__(model or DEFAULT_MODEL)
        self._client = client or openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
        }
        if tools:
            kwargs["tools"] = [
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
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        ) as exc:
            raise LLMUnavailableError(f"OpenAI unavailable: {exc}") from exc
        except openai.InternalServerError as exc:
            raise LLMUnavailableError(f"OpenAI server error: {exc}") from exc
        except openai.OpenAIError as exc:
            raise LLMError(f"OpenAI error: {exc}") from exc

        choice = response.choices[0].message
        calls: list[ToolCall] = []
        for raw in choice.tool_calls or []:
            try:
                arguments = json.loads(raw.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": raw.function.arguments}
            calls.append(
                ToolCall(name=raw.function.name, arguments=arguments, id=raw.id)
            )

        usage = {}
        if response.usage is not None:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return LLMResponse(
            text=(choice.content or "").strip(), tool_calls=calls, usage=usage
        )


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
            )
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(
                                    c.arguments, ensure_ascii=False
                                ),
                            },
                        }
                        for c in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out
