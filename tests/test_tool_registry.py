"""The tool registry: discovery, validation, permissions, errors, timeouts."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from nec_ai.config.settings import Settings
from nec_ai.tools.base import (
    PermissionDecision,
    RiskLevel,
    Tool,
    ToolContext,
    ToolResult,
)
from nec_ai.tools.registry import ConfirmationRequest, ToolRegistry


class Echo(Tool):
    name = "echo"
    description = "Repeat a text."

    class Input(BaseModel):
        text: str
        times: int = 1

    async def run(self, args: Input, ctx: ToolContext) -> ToolResult:
        return ToolResult.success(args.text * args.times)


class Dangerous(Tool):
    name = "dangerous"
    description = "Needs a yes."

    class Input(BaseModel):
        target: str

    def check_permission(self, args: Input, ctx: ToolContext) -> PermissionDecision:
        if args.target == "system":
            return PermissionDecision.block("touches the system")
        return PermissionDecision.confirm("deletes something")

    async def run(self, args: Input, ctx: ToolContext) -> ToolResult:
        return ToolResult.success(f"deleted {args.target}")


class Crashing(Tool):
    name = "crash"
    description = "Always raises."

    async def run(self, args, ctx):
        raise RuntimeError("boom")


class Slow(Tool):
    name = "slow"
    description = "Never finishes in time."
    timeout = 0.05

    async def run(self, args, ctx):
        await asyncio.sleep(5)
        return ToolResult.success("late")


class Web(Tool):
    name = "web"
    description = "Returns page text."
    untrusted_output = True

    async def run(self, args, ctx):
        return ToolResult.success("x" * 50)


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(settings=Settings(_env_file=None))


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry([Echo(), Dangerous(), Crashing(), Slow(), Web()])


def test_tools_are_discoverable(registry: ToolRegistry) -> None:
    assert registry.names == ["crash", "dangerous", "echo", "slow", "web"]
    spec = next(s for s in registry.specs() if s.name == "echo")
    assert spec.description == "Repeat a text."
    assert spec.parameters["type"] == "object"
    assert set(spec.parameters["properties"]) == {"text", "times"}
    assert spec.parameters["required"] == ["text"]
    assert "title" not in spec.parameters


def test_duplicate_names_are_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(ValueError):
        registry.register(Echo())


async def test_a_safe_tool_runs(registry: ToolRegistry, ctx: ToolContext) -> None:
    run = await registry.execute("echo", {"text": "ab", "times": 2}, ctx)
    assert run.result.ok
    assert run.result.content == "abab"
    assert run.decision.level is RiskLevel.SAFE


async def test_unknown_tool_lists_the_available_ones(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    run = await registry.execute("nope", {}, ctx)
    assert not run.result.ok
    assert "echo" in run.result.content


async def test_invalid_arguments_are_explained(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    run = await registry.execute("echo", {"times": "many"}, ctx)
    assert not run.result.ok
    assert "text" in run.result.content
    assert "times" in run.result.content


async def test_blocked_calls_never_run(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    asked: list[ConfirmationRequest] = []

    async def yes(request: ConfirmationRequest) -> bool:
        asked.append(request)
        return True

    run = await registry.execute("dangerous", {"target": "system"}, ctx, confirm=yes)
    assert not run.result.ok
    assert run.decision.level is RiskLevel.BLOCKED
    assert asked == []  # blocked means the user is not even asked


async def test_confirmation_required_without_a_handler_is_refused(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    run = await registry.execute("dangerous", {"target": "tmp"}, ctx)
    assert not run.result.ok
    assert run.approved is False
    assert "not approve" in run.result.content


async def test_confirmation_yes_runs_the_tool(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    async def yes(request: ConfirmationRequest) -> bool:
        assert request.tool == "dangerous"
        assert request.reason == "deletes something"
        return True

    run = await registry.execute("dangerous", {"target": "tmp"}, ctx, confirm=yes)
    assert run.result.ok
    assert run.approved is True


async def test_confirmation_no_does_not_run_the_tool(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    async def no(request: ConfirmationRequest) -> bool:
        return False

    run = await registry.execute("dangerous", {"target": "tmp"}, ctx, confirm=no)
    assert not run.result.ok
    assert "deleted" not in run.result.content


async def test_a_crashing_confirmation_handler_means_no(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    async def broken(request: ConfirmationRequest) -> bool:
        raise RuntimeError("ui gone")

    run = await registry.execute("dangerous", {"target": "tmp"}, ctx, confirm=broken)
    assert not run.result.ok


async def test_exceptions_become_failed_results(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    run = await registry.execute("crash", {}, ctx)
    assert not run.result.ok
    assert "boom" in run.result.content


async def test_timeouts_become_failed_results(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    run = await registry.execute("slow", {}, ctx)
    assert not run.result.ok
    assert "timed out" in run.result.content


async def test_output_is_bounded_and_marked_untrusted(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    small = ToolContext(settings=Settings(_env_file=None, max_tool_output_chars=20))
    run = await registry.execute("web", {}, small)
    assert run.result.content.startswith("<page_data>")
    assert run.result.content.endswith("</page_data>")
    assert "truncated" in run.result.content
