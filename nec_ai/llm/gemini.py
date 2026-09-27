"""Google Gemini through the ``google-genai`` SDK.

Automatic function calling is switched off: the SDK must never run a tool by
itself. The model proposes calls, the agent loop decides and executes them.
"""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import errors, types

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

#: Google's rolling alias for the current Flash model, so the default keeps
#: working when a dated model is retired. Pin one with LLM_MODEL if needed.
DEFAULT_MODEL = "gemini-flash-latest"

_RETRYABLE = {408, 429, 500, 502, 503, 504}


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str = "", client: Any = None) -> None:
        super().__init__(model or DEFAULT_MODEL)
        self._client = client or genai.Client(api_key=api_key)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
    ) -> LLMResponse:
        system, contents = to_gemini_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            tools=[types.Tool(function_declarations=[to_declaration(t) for t in tools])]
            if tools
            else None,
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            message = f"Gemini error {code}: {getattr(exc, 'message', exc)}"
            if code in _RETRYABLE:
                raise LLMUnavailableError(message) from exc
            if code == 404:
                message += f" (model {self.model!r} not found; set LLM_MODEL)"
            raise LLMError(message) from exc
        except (OSError, TimeoutError) as exc:
            raise LLMUnavailableError(f"Gemini unreachable: {exc}") from exc

        return from_gemini_response(response)


def to_declaration(spec: ToolSpec) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=spec.name,
        description=spec.description,
        parameters_json_schema=spec.parameters,
    )


def to_gemini_contents(messages: list[Message]) -> tuple[str, list[types.Content]]:
    """Translate neutral messages. Consecutive tool results share one turn."""
    system_parts: list[str] = []
    contents: list[types.Content] = []

    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
            continue

        if message.role == "tool":
            part = types.Part(
                function_response=types.FunctionResponse(
                    id=message.tool_call_id,
                    name=message.name or "tool",
                    response={"result": message.content},
                )
            )
            last = contents[-1] if contents else None
            if last is not None and last.role == "user" and _is_function_turn(last):
                last.parts.append(part)
            else:
                contents.append(types.Content(role="user", parts=[part]))
            continue

        if message.role == "assistant":
            if message.provider == GeminiProvider.name and isinstance(
                message.provider_data, types.Content
            ):
                # Reuse the exact content: it carries thought signatures that
                # Gemini requires back on the next turn.
                contents.append(message.provider_data)
                continue
            parts: list[types.Part] = []
            if message.content:
                parts.append(types.Part(text=message.content))
            for call in message.tool_calls:
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call.id, name=call.name, args=call.arguments
                        )
                    )
                )
            contents.append(
                types.Content(role="model", parts=parts or [types.Part(text="")])
            )
            continue

        contents.append(
            types.Content(role="user", parts=[types.Part(text=message.content)])
        )

    return "\n\n".join(p for p in system_parts if p), contents


def _is_function_turn(content: types.Content) -> bool:
    return bool(content.parts) and all(p.function_response for p in content.parts)


def from_gemini_response(response: Any) -> LLMResponse:
    candidates = getattr(response, "candidates", None) or []
    if not candidates or candidates[0].content is None:
        feedback = getattr(response, "prompt_feedback", None)
        reason = getattr(feedback, "block_reason", None) if feedback else None
        if reason:
            raise LLMError(f"Gemini refused the request ({reason}).")
        return LLMResponse(text="")

    content = candidates[0].content
    texts: list[str] = []
    calls: list[ToolCall] = []
    for part in content.parts or []:
        if part.function_call is not None:
            fc = part.function_call
            call = ToolCall(name=fc.name or "", arguments=dict(fc.args or {}))
            if fc.id:
                call.id = fc.id
            calls.append(call)
        elif part.text and not part.thought:
            texts.append(part.text)

    usage: dict[str, int] = {}
    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        usage = {
            "input_tokens": meta.prompt_token_count or 0,
            "output_tokens": meta.candidates_token_count or 0,
        }
    return LLMResponse(
        text="".join(texts).strip(),
        tool_calls=calls,
        usage=usage,
        provider_data=content,
    )
