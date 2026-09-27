"""A scripted LLM for tests and offline demos.

    llm = FakeLLM([
        LLMResponse(tool_calls=[ToolCall("web_search", {"query": "x"})]),
        LLMResponse(text="Voici la réponse."),
    ])

Each ``complete`` call returns the next scripted response (or the result of a
callable, which receives the messages). Every call is recorded for assertions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from nec_ai.llm.base import LLMError, LLMProvider, LLMResponse, Message
from nec_ai.tools.base import ToolSpec

Script = LLMResponse | Exception | Callable[[list[Message]], LLMResponse]


@dataclass
class RecordedCall:
    messages: list[Message]
    tools: list[ToolSpec] | None


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(
        self, script: Iterable[Script] = (), default: str | None = None
    ) -> None:
        super().__init__("fake")
        self._script = list(script)
        self._default = default
        self.calls: list[RecordedCall] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append(RecordedCall(list(messages), tools))
        if not self._script:
            if self._default is not None:
                return LLMResponse(text=self._default)
            raise LLMError("FakeLLM script exhausted")
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step(messages)
        return step
