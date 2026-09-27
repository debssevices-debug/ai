"""The generic tool contract.

A tool is four things: a ``name``, a ``description`` the LLM reads, an input
model (a pydantic class, which gives both the JSON schema shown to the LLM and
the validation of what it sends back) and an async ``run``.

    class Echo(Tool):
        name = "echo"
        description = "Repeat a text."

        class Input(BaseModel):
            text: str

        async def run(self, args: Input, ctx: ToolContext) -> ToolResult:
            return ToolResult.success(args.text)

Tools never talk to the LLM and never decide *what* to do: the agent decides,
the tool executes. Tools also never raise to the agent: the registry turns any
exception into a failed :class:`ToolResult` the model can read and recover from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from nec_ai.agent.context import TaskContext
    from nec_ai.config.settings import Settings


class RiskLevel(StrEnum):
    """What the permission layer decides for one tool call."""

    SAFE = "safe"
    """Runs without asking."""

    CONFIRMATION_REQUIRED = "confirmation_required"
    """Runs only after the user explicitly approves this exact call."""

    BLOCKED = "blocked"
    """Never runs, whatever the user or the model says."""


@dataclass(frozen=True)
class PermissionDecision:
    level: RiskLevel
    reason: str = ""

    @classmethod
    def safe(cls) -> PermissionDecision:
        return cls(RiskLevel.SAFE)

    @classmethod
    def confirm(cls, reason: str) -> PermissionDecision:
        return cls(RiskLevel.CONFIRMATION_REQUIRED, reason)

    @classmethod
    def block(cls, reason: str) -> PermissionDecision:
        return cls(RiskLevel.BLOCKED, reason)


@dataclass
class ToolResult:
    """What a tool hands back.

    ``content`` is the text the LLM reads. ``data`` is an optional structured
    payload for API clients (search results, file listings) that is never sent
    to the model as-is.
    """

    ok: bool
    content: str
    data: Any = None
    error: str | None = None

    @classmethod
    def success(cls, content: str, data: Any = None) -> ToolResult:
        return cls(ok=True, content=content, data=data)

    @classmethod
    def failure(cls, error: str, data: Any = None) -> ToolResult:
        return cls(ok=False, content=f"ERROR: {error}", data=data, error=error)


@dataclass
class ToolContext:
    """Everything a tool may need besides its arguments."""

    settings: Settings
    session_id: str = "default"
    task: TaskContext | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral description of a tool, as shown to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]


class NoInput(BaseModel):
    """Input model for tools that take no argument."""


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    Input: ClassVar[type[BaseModel]] = NoInput

    timeout: ClassVar[float | None] = None
    """Seconds before the call is cancelled. None uses settings.tool_timeout."""

    untrusted_output: ClassVar[bool] = False
    """Wrap the output as untrusted data (web pages, search snippets, files)."""

    def check_permission(self, args: BaseModel, ctx: ToolContext) -> PermissionDecision:
        """Decide whether this call may run. Override for risky tools."""
        return PermissionDecision.safe()

    @abstractmethod
    async def run(self, args: Any, ctx: ToolContext) -> ToolResult:
        """Execute the call. ``args`` is an instance of ``self.Input``."""

    def describe_call(self, args: BaseModel) -> str:
        """One human line for UIs and confirmations: ``web_search: "..."``."""
        values = ", ".join(f"{k}={v!r}" for k, v in args.model_dump().items())
        return f"{self.name}({values})"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description.strip(),
            parameters=clean_schema(self.Input.model_json_schema()),
        )


def clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip pydantic's ``title`` noise so every tool costs fewer tokens."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                key: walk(value)
                for key, value in node.items()
                if not (key == "title" and isinstance(value, str))
            }
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    cleaned = walk(schema)
    cleaned.setdefault("type", "object")
    cleaned.setdefault("properties", {})
    return cleaned
