"""Tool discovery and safe execution.

The registry is the only way the agent runs a tool, which makes it the one
place that enforces the rules for every tool alike:

1. the tool exists and is enabled;
2. the arguments validate against the tool's input model;
3. the permission layer allows the call (SAFE / CONFIRMATION_REQUIRED / BLOCKED);
4. the call finishes within its timeout;
5. any exception becomes a failed :class:`ToolResult`, never a crash;
6. the output is bounded, and wrapped as untrusted data when it came from outside.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from nec_ai.security.untrusted import wrap_untrusted
from nec_ai.tools.base import (
    PermissionDecision,
    RiskLevel,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)

logger = logging.getLogger("nec.tools")

_SAFE = PermissionDecision.safe()


@dataclass
class ConfirmationRequest:
    """A risky call waiting for the user's yes or no."""

    tool: str
    arguments: dict[str, Any]
    reason: str
    summary: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


#: Asks the user, returns True only on an explicit yes.
ConfirmationHandler = Callable[[ConfirmationRequest], Awaitable[bool]]


@dataclass
class ToolExecution:
    """The full record of one call, for events and traces."""

    tool: str
    arguments: dict[str, Any]
    result: ToolResult
    decision: PermissionDecision
    duration: float
    confirmation: ConfirmationRequest | None = None
    approved: bool | None = None


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    # ── discovery ───────────────────────────────────────────────────────
    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"a tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec() for name in self.names]

    # ── execution ───────────────────────────────────────────────────────
    async def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        ctx: ToolContext,
        confirm: ConfirmationHandler | None = None,
    ) -> ToolExecution:
        arguments = dict(arguments or {})
        started = time.monotonic()

        def done(
            result: ToolResult,
            decision: PermissionDecision = _SAFE,
            request: ConfirmationRequest | None = None,
            approved: bool | None = None,
        ) -> ToolExecution:
            return ToolExecution(
                tool=name,
                arguments=arguments,
                result=result,
                decision=decision,
                duration=time.monotonic() - started,
                confirmation=request,
                approved=approved,
            )

        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self.names) or "none"
            return done(
                ToolResult.failure(
                    f"Unknown tool {name!r}. Available tools: {available}."
                )
            )

        try:
            args = tool.Input.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'input'}: {err['msg']}"
                for err in exc.errors()
            )
            return done(ToolResult.failure(f"Invalid arguments for {name}: {problems}"))

        try:
            decision = tool.check_permission(args, ctx)
        except Exception as exc:  # a broken policy must fail closed
            logger.exception("permission check crashed for %s", name)
            decision = PermissionDecision.block(f"permission check failed: {exc}")

        request: ConfirmationRequest | None = None
        approved: bool | None = None
        if decision.level is RiskLevel.BLOCKED:
            logger.warning("TOOL %s blocked: %s", name, decision.reason)
            return done(
                ToolResult.failure(
                    f"This action is blocked by the security policy: {decision.reason}. "
                    "Do not retry it; tell the user it is not allowed."
                ),
                decision,
            )
        if decision.level is RiskLevel.CONFIRMATION_REQUIRED:
            request = ConfirmationRequest(
                tool=name,
                arguments=arguments,
                reason=decision.reason,
                summary=tool.describe_call(args),
            )
            if confirm is None:
                approved = False
            else:
                try:
                    approved = bool(await confirm(request))
                except Exception:
                    logger.exception("confirmation handler failed for %s", name)
                    approved = False
            if not approved:
                logger.info("TOOL %s refused by the user", name)
                return done(
                    ToolResult.failure(
                        f"The user did not approve this action ({decision.reason}). "
                        "It was not carried out. Do not claim it was done."
                    ),
                    decision,
                    request,
                    approved,
                )

        timeout = (
            tool.timeout if tool.timeout is not None else ctx.settings.tool_timeout
        )
        logger.info("TOOL %s started", name)
        try:
            result = await asyncio.wait_for(tool.run(args, ctx), timeout=timeout)
        except TimeoutError:
            logger.warning("TOOL %s timed out after %gs", name, timeout)
            result = ToolResult.failure(
                f"{name} timed out after {timeout:g} seconds."
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("TOOL %s crashed", name)
            result = ToolResult.failure(f"{name} failed: {type(exc).__name__}: {exc}")

        if not isinstance(result, ToolResult):
            result = ToolResult.failure(f"{name} returned an invalid result.")

        result.content = self._bound(result.content, ctx.settings.max_tool_output_chars)
        if tool.untrusted_output and result.ok:
            result.content = wrap_untrusted(result.content)

        logger.info(
            "TOOL %s %s (%.2fs)",
            name,
            "completed" if result.ok else "failed",
            time.monotonic() - started,
        )
        return done(result, decision, request, approved)

    @staticmethod
    def _bound(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"
