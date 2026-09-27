"""Provider-neutral LLM interface.

The agent only ever sees :class:`Message`, :class:`ToolCall` and
:class:`LLMResponse`. Each provider translates them to and from its own API, so
switching from Gemini to OpenAI (or anything else) is a configuration change,
not a rewrite.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
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

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        kind: str = "unavailable",
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        """Seconds the provider asked us to wait, when it said so."""

        self.kind = kind
        """``rate_limit`` (quota), ``overloaded`` (5xx), ``timeout`` or ``unavailable``."""


class RequestThrottle:
    """Keeps requests under N per minute, so a free-tier quota is never hit.

    Waiting a few seconds before a request is much better than a 429: some
    providers count rejected requests against the quota too.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._sent: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, on_wait: Callable[[float], None] | None = None) -> None:
        if self.per_minute <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            while self._sent and now - self._sent[0] >= 60:
                self._sent.popleft()
            if len(self._sent) >= self.per_minute:
                wait = 60 - (now - self._sent[0]) + 0.2
                if on_wait is not None:
                    on_wait(wait)
                logger.info(
                    "LLM throttle: waiting %.1fs (limit %d/min)", wait, self.per_minute
                )
                await asyncio.sleep(wait)
                self._sent.popleft()
            self._sent.append(time.monotonic())


#: Called before each retry with (error, seconds to wait, attempt number).
RetryCallback = Callable[[LLMUnavailableError, float, int], None]


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
    retries: int = 3,
    temperature: float | None = None,
    max_wait: float = 60.0,
    throttle: RequestThrottle | None = None,
    on_retry: RetryCallback | None = None,
    on_throttle: Callable[[float], None] | None = None,
) -> LLMResponse:
    """Call the model with a timeout, retrying transient errors.

    The wait before a retry is what the provider asked for (``retry_after``,
    e.g. a quota reset in 7 s) or else an exponential backoff (2, 4, 8 s...).
    When the provider asks for longer than ``max_wait`` (a daily quota), the
    error is raised at once rather than blocking the user for minutes.
    """
    attempt = 0
    while True:
        if throttle is not None:
            await throttle.acquire(on_throttle)
        try:
            return await asyncio.wait_for(
                provider.complete(messages, tools, temperature=temperature),
                timeout=timeout,
            )
        except TimeoutError:
            error = LLMUnavailableError(
                f"the model did not answer within {timeout:g}s", kind="timeout"
            )
        except LLMUnavailableError as exc:
            error = exc
        if attempt >= retries:
            raise error
        if error.retry_after is not None:
            delay = error.retry_after + 0.5
        else:
            delay = min(2.0 * 2**attempt, 30.0)
        if delay > max_wait:
            raise error
        attempt += 1
        logger.warning(
            "LLM unavailable (%s); retry %d/%d in %.0fs", error, attempt, retries, delay
        )
        if on_retry is not None:
            on_retry(error, delay, attempt)
        await asyncio.sleep(delay)
