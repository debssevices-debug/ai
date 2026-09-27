"""The agent loop: THINK → ACT → OBSERVE, until a final answer.

    agent = Agent(llm, registry, settings)
    async for event in agent.run("Compare Odigo, Genesys et Aircall"):
        print(event.type, event.data)

One request can take many steps. Each iteration asks the LLM what to do next;
if it calls tools, they run through the registry (validation, permissions,
confirmation, timeouts) and their results go back to the LLM; when it answers
without calling a tool, that answer is final.

Guarantees:
* never more than ``MAX_AGENT_ITERATIONS`` LLM turns; at the limit the agent is
  asked to answer with what it has rather than stopping silently;
* an identical tool call repeated is not executed again (loop protection);
* an LLM outage or a crashing tool produces an ``agent.error`` / failed tool
  event and a readable answer, never an exception out of ``run``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from nec_ai.agent.context import TaskContext
from nec_ai.agent.events import AgentEvent, EventType
from nec_ai.agent.planner import PLAN_TOOL
from nec_ai.config.settings import Settings
from nec_ai.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    LLMUnavailableError,
    Message,
    RequestThrottle,
    ToolCall,
    complete_with_retry,
)
from nec_ai.llm.prompts import system_prompt
from nec_ai.memory.short_term import ConversationMemory
from nec_ai.observability.trace import TraceWriter
from nec_ai.tools.base import ToolContext
from nec_ai.tools.registry import ConfirmationHandler, ConfirmationRequest, ToolRegistry

logger = logging.getLogger("nec.agent")

#: How many times the exact same call may run in one task.
MAX_IDENTICAL_CALLS = 1

LIMIT_NOTICE = (
    "Tu as atteint la limite d'étapes pour cette demande. N'appelle plus d'outil : "
    "réponds maintenant avec ce que tu as trouvé, et indique clairement ce qui "
    "manque ou reste à vérifier."
)

FALLBACK_ANSWER = (
    "Je n'ai pas pu terminer cette demande : le modèle de langage est indisponible "
    "pour le moment. Réessaie dans un instant."
)

INTERNAL_ERROR_ANSWER = (
    "Une erreur interne m'a empêché de terminer cette demande. Elle a été "
    "enregistrée dans les journaux."
)


def llm_failure_answer(exc: Exception) -> str:
    """What the user reads when the LLM fails: why, and what to do about it."""
    if isinstance(exc, LLMUnavailableError):
        if exc.kind == "rate_limit":
            return (
                "Je n'ai pas pu terminer : le quota de l'API du modèle est atteint "
                f"({exc}). Attends une minute avant de réessayer. Si ça se répète, "
                "règle LLM_REQUESTS_PER_MINUTE dans le .env (5 pour l'offre gratuite "
                "Gemini), passe sur un modèle moins demandé (LLM_MODEL) ou active la "
                "facturation sur ta clé API."
            )
        if exc.kind == "overloaded":
            return (
                "Je n'ai pas pu terminer : le modèle est surchargé côté fournisseur "
                f"({exc}). Réessaie dans un moment, ou change de modèle avec LLM_MODEL."
            )
        return FALLBACK_ANSWER
    return (
        "Je n'ai pas pu traiter cette demande : le modèle de langage a renvoyé une "
        f"erreur ({exc}). Vérifie la configuration (clé API, nom du modèle)."
    )


EventSink = Callable[[AgentEvent], None]


class Agent:
    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        settings: Settings,
        *,
        memory: ConversationMemory | None = None,
        tracer: TraceWriter | None = None,
        prompt_builder: Callable[[], str] | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.settings = settings
        self.memory = memory or ConversationMemory(settings.history_max_messages)
        self.tracer = tracer
        self._prompt = prompt_builder or (lambda: system_prompt(voice=False))
        # One throttle per agent: the quota is per API key, across requests.
        self.throttle = RequestThrottle(settings.llm_requests_per_minute)

    # ── public API ──────────────────────────────────────────────────────
    async def run(
        self,
        request: str,
        *,
        session_id: str = "default",
        confirm: ConfirmationHandler | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run one request, streaming events. Never raises for agent failures."""
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        run_id = uuid.uuid4().hex[:12]

        def emit(event_type: EventType, **data: Any) -> None:
            event = AgentEvent(event_type, run_id, data)
            if self.tracer is not None:
                self.tracer.record(event)
            queue.put_nowait(event)

        async def worker() -> None:
            try:
                await self._loop(request, session_id, confirm, emit)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # last line of defence
                logger.exception("AGENT crashed")
                emit(EventType.ERROR, error=f"{type(exc).__name__}: {exc}", fatal=True)
                emit(EventType.FINAL, answer=INTERNAL_ERROR_ANSWER, failed=True)
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(worker())
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def ask(
        self,
        request: str,
        *,
        session_id: str = "default",
        confirm: ConfirmationHandler | None = None,
        on_event: EventSink | None = None,
    ) -> str:
        """Run a request and return only the final answer."""
        answer = ""
        async for event in self.run(request, session_id=session_id, confirm=confirm):
            if on_event is not None:
                on_event(event)
            if event.type is EventType.FINAL:
                answer = event.data.get("answer", "")
        return answer

    # ── the loop ────────────────────────────────────────────────────────
    async def _loop(
        self,
        request: str,
        session_id: str,
        confirm: ConfirmationHandler | None,
        emit: Callable[..., None],
    ) -> None:
        request = request.strip()
        task = TaskContext(request=request)
        ctx = ToolContext(settings=self.settings, session_id=session_id, task=task)
        emit(EventType.STARTED, request=request, session_id=session_id)
        logger.info("AGENT started (session=%s)", session_id)

        if not request:
            emit(EventType.FINAL, answer="Je n'ai reçu aucune demande.", iterations=0)
            return

        messages: list[Message] = [
            Message.system(self._prompt()),
            *self.memory.history(session_id),
            Message.user(request),
        ]
        specs = self.registry.specs()
        seen_calls: dict[str, int] = {}
        tools_used: list[str] = []
        confirm_with_events = self._confirmation_bridge(confirm, emit)

        for iteration in range(1, self.settings.max_agent_iterations + 1):
            task.iterations = iteration
            emit(EventType.THINKING, iteration=iteration)
            logger.info("AGENT reasoning (step %d)", iteration)

            try:
                response = await self._complete(messages, specs or None, emit)
            except LLMError as exc:
                logger.error("AGENT LLM failure: %s", exc)
                emit(EventType.ERROR, error=str(exc), stage="llm")
                emit(
                    EventType.FINAL,
                    answer=llm_failure_answer(exc),
                    failed=True,
                    iterations=iteration,
                )
                return

            if not response.tool_calls:
                answer = response.text or "Je n'ai pas de réponse à proposer."
                self._finish(session_id, request, answer, tools_used, emit, iteration)
                return

            if response.text:
                emit(EventType.MESSAGE, text=response.text)
            messages.append(response.as_message(self.llm.name))

            # Independent calls from the same turn run concurrently (reading
            # four pages takes as long as the slowest one). Results go back to
            # the model in the order it asked for them.
            contents = await asyncio.gather(
                *(
                    self._run_tool(
                        call, ctx, confirm_with_events, emit, seen_calls, tools_used
                    )
                    for call in response.tool_calls
                )
            )
            for call, content in zip(response.tool_calls, contents, strict=True):
                messages.append(Message.tool_result(call, content))

        # Iteration budget spent: one last turn without tools to get an answer.
        logger.warning(
            "AGENT hit MAX_AGENT_ITERATIONS=%d", self.settings.max_agent_iterations
        )
        emit(
            EventType.ERROR, error="iteration limit reached", stage="loop", fatal=False
        )
        messages.append(Message.user(LIMIT_NOTICE))
        try:
            response = await self._complete(messages, None, emit)
            answer = response.text or FALLBACK_ANSWER
        except LLMError as exc:
            emit(EventType.ERROR, error=str(exc), stage="llm")
            answer = llm_failure_answer(exc)
        self._finish(
            session_id,
            request,
            answer,
            tools_used,
            emit,
            self.settings.max_agent_iterations,
            limit_reached=True,
        )

    async def _complete(
        self, messages: list[Message], specs: list | None, emit: Callable[..., None]
    ) -> LLMResponse:
        """One LLM turn with pacing and retries; waits are streamed as events."""

        def on_retry(error: LLMUnavailableError, delay: float, attempt: int) -> None:
            emit(
                EventType.WAITING,
                reason=error.kind,
                seconds=round(delay, 1),
                attempt=attempt,
                detail=str(error),
            )

        def on_throttle(delay: float) -> None:
            emit(
                EventType.WAITING, reason="throttle", seconds=round(delay, 1), attempt=0
            )

        return await complete_with_retry(
            self.llm,
            messages,
            specs,
            timeout=self.settings.llm_timeout,
            retries=self.settings.llm_max_retries,
            temperature=self.settings.llm_temperature,
            max_wait=self.settings.llm_max_retry_wait,
            throttle=self.throttle,
            on_retry=on_retry,
            on_throttle=on_throttle,
        )

    async def _run_tool(
        self,
        call: ToolCall,
        ctx: ToolContext,
        confirm: ConfirmationHandler,
        emit: Callable[..., None],
        seen_calls: dict[str, int],
        tools_used: list[str],
    ) -> str:
        signature = (
            f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
        )
        emit(
            EventType.TOOL_STARTED,
            tool=call.name,
            arguments=call.arguments,
            call_id=call.id,
        )

        if seen_calls.get(signature, 0) >= MAX_IDENTICAL_CALLS:
            content = (
                "ERROR: you already made this exact call in this task. Use its "
                "previous result, change the arguments, or give your answer."
            )
            emit(
                EventType.TOOL_FAILED,
                tool=call.name,
                call_id=call.id,
                error="duplicate call skipped",
                duration=0.0,
            )
            return content
        seen_calls[signature] = seen_calls.get(signature, 0) + 1

        execution = await self.registry.execute(call.name, call.arguments, ctx, confirm)
        result = execution.result
        tools_used.append(call.name)
        if ctx.task is not None:
            ctx.task.record(call.name, call.arguments, result.ok, result.content)

        if call.name == PLAN_TOOL and result.ok and isinstance(result.data, dict):
            emit(EventType.PLAN, **result.data)

        emit(
            EventType.TOOL_COMPLETED if result.ok else EventType.TOOL_FAILED,
            tool=call.name,
            call_id=call.id,
            duration=round(execution.duration, 3),
            preview=result.content[:300],
            data=result.data,
            error=result.error,
            risk=str(execution.decision.level),
        )
        return result.content

    def _confirmation_bridge(
        self, confirm: ConfirmationHandler | None, emit: Callable[..., None]
    ) -> ConfirmationHandler:
        """Wrap the caller's handler so every confirmation is also an event."""

        # Tools may run concurrently, but the user answers one question at a time.
        lock = asyncio.Lock()

        async def bridge(request: ConfirmationRequest) -> bool:
            async with lock:
                return await ask(request)

        async def ask(request: ConfirmationRequest) -> bool:
            emit(
                EventType.CONFIRMATION_REQUIRED,
                id=request.id,
                tool=request.tool,
                arguments=request.arguments,
                reason=request.reason,
                summary=request.summary,
            )
            approved = False
            if confirm is not None:
                handler: Callable[[ConfirmationRequest], Awaitable[bool]] = confirm
                approved = bool(await handler(request))
            emit(EventType.CONFIRMATION_RESOLVED, id=request.id, approved=approved)
            return approved

        return bridge

    def _finish(
        self,
        session_id: str,
        request: str,
        answer: str,
        tools_used: list[str],
        emit: Callable[..., None],
        iterations: int,
        *,
        limit_reached: bool = False,
    ) -> None:
        self.memory.add_exchange(
            session_id, request, answer, [t for t in tools_used if t != PLAN_TOOL]
        )
        emit(
            EventType.FINAL,
            answer=answer,
            iterations=iterations,
            tools_used=tools_used,
            limit_reached=limit_reached,
        )
        logger.info("AGENT completed in %d step(s)", iterations)
