"""Provider-neutral LLM interface.

The agent only ever sees :class:`Message`, :class:`ToolCall` and
:class:`LLMResponse`. Each provider translates them to and from its own API, so
switching from Gemini to OpenAI (or anything else) is a configuration change,
not a rewrite.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from nec_ai.tools.base import ToolSpec

logger = logging.getLogger("nec.llm")

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:10]}")


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    """For ``role="tool"``: which call this answers."""

    name: str | None = None
    """For ``role="tool"``: the tool's name."""

    provider_data: Any = None
    """Opaque provider payload (e.g. Gemini thought signatures). Only reused by
    the provider that produced it."""

    provider: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls("assistant", content)

    @classmethod
    def tool_result(cls, call: ToolCall, content: str) -> Message:
        return cls("tool", content, tool_call_id=call.id, name=call.name)


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    provider_data: Any = None

    def as_message(self, provider: str) -> Message:
        return Message(
            "assistant",
            self.text,
            tool_calls=list(self.tool_calls),
            provider_data=self.provider_data,
            provider=provider,
        )


class LLMError(Exception):
    """The LLM call failed and retrying will not help (bad key, bad request)."""


class LLMUnavailableError(LLMError):
    """Transient failure: timeout, rate limit, 5xx. Worth retrying."""


class LLMProvider(ABC):
    name: str = "llm"

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
    ) -> LLMResponse:
        """One model turn. Raises :class:`LLMError` / :class:`LLMUnavailableError`."""

    async def aclose(self) -> None:  # pragma: no cover - optional
        return None


async def complete_with_retry(
    provider: LLMProvider,
    messages: list[Message],
    tools: list[ToolSpec] | None = None,
    *,
    timeout: float = 60.0,
    retries: int = 2,
    temperature: float | None = None,
) -> LLMResponse:
    """Call the model with a timeout and exponential backoff on transient errors."""
    attempt = 0
    while True:
        try:
            return await asyncio.wait_for(
                provider.complete(messages, tools, temperature=temperature),
                timeout=timeout,
            )
        except TimeoutError:
            error: LLMError = LLMUnavailableError(
                f"the model did not answer within {timeout:g}s"
            )
        except LLMUnavailableError as exc:
            error = exc
        if attempt >= retries:
            raise error
        delay = 2**attempt
        attempt += 1
        logger.warning(
            "LLM unavailable (%s); retry %d/%d in %ds", error, attempt, retries, delay
        )
        await asyncio.sleep(delay)
