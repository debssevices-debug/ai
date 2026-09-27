"""NEC REST API (FastAPI).

    GET    /health                     liveness, no auth
    GET    /v1/tools                   the tools the agent can use
    POST   /v1/chat                    run a request, return the final answer
    POST   /v1/chat/stream             same, streamed as Server-Sent Events
    POST   /v1/confirmations/{id}      approve or refuse a risky action
    DELETE /v1/sessions/{session_id}   forget a conversation

Streaming events are the agent's own (``agent.started``, ``agent.thinking``,
``tool.started``, ``tool.completed``, ``confirmation.required``,
``agent.final``, ``agent.error``...), one SSE message each::

    event: tool.started
    data: {"type": "tool.started", "run_id": "...", "data": {...}}

Sessions are scoped to the API key: two keys never see each other's history.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nec_ai import __version__
from nec_ai.agent.core import Agent
from nec_ai.agent.events import AgentEvent, EventType
from nec_ai.api.confirmations import ConfirmationBroker
from nec_ai.api.security import Authenticator, RateLimiter
from nec_ai.config.settings import Settings, get_settings
from nec_ai.llm import LLMError

logger = logging.getLogger("nec.api")

SESSION_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"
KEEPALIVE_SECONDS = 15.0


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="What you ask NEC.")
    session_id: str = Field("default", pattern=SESSION_PATTERN)


class ChatResponse(BaseModel):
    answer: str
    run_id: str
    session_id: str
    iterations: int = 0
    tools_used: list[str] = []
    failed: bool = False
    events: list[dict[str, Any]] = []


class ConfirmationAnswer(BaseModel):
    approve: bool


def create_app(settings: Settings | None = None, agent: Agent | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="NEC AI",
        version=__version__,
        description="API de l'agent personnel NEC.",
        # Interactive docs only for local development (no key configured).
        docs_url="/docs" if not settings.api_keys else None,
        openapi_url="/openapi.json" if not settings.api_keys else None,
        redoc_url=None,
    )
    if settings.api_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api_cors_origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    authenticate = Authenticator(settings)
    limiter = RateLimiter(settings.rate_limit_per_minute)
    broker = ConfirmationBroker(settings.confirmation_timeout)
    runs = asyncio.Semaphore(max(1, settings.api_max_concurrent_runs))
    state: dict[str, Any] = {"agent": agent}

    def get_agent() -> Agent:
        if state["agent"] is None:
            from nec_ai.app import build_agent

            try:
                state["agent"] = build_agent(settings)
            except LLMError as exc:
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"LLM not configured: {exc}",
                ) from None
        return state["agent"]

    def limited_client(client: str = Depends(authenticate)) -> str:
        limiter.check(client)
        return client

    def validate_message(body: ChatRequest) -> str:
        text = body.message.strip()
        if not text:
            raise HTTPException(422, "Empty message.")
        if len(text) > settings.max_request_chars:
            raise HTTPException(
                413,
                f"Message longer than {settings.max_request_chars} characters.",
            )
        return text

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("API error on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": "Internal error."}, status_code=500)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/tools")
    async def tools(client: str = Depends(authenticate)) -> list[dict[str, Any]]:
        return [
            {"name": s.name, "description": s.description, "parameters": s.parameters}
            for s in get_agent().registry.specs()
        ]

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(
        body: ChatRequest, client: str = Depends(limited_client)
    ) -> ChatResponse:
        text = validate_message(body)
        agent = get_agent()
        events: list[AgentEvent] = []
        async with runs:
            # No one can answer a confirmation on this endpoint: risky actions
            # are refused. Use /v1/chat/stream to be asked.
            async for event in agent.run(
                text, session_id=_scoped(client, body.session_id)
            ):
                events.append(event)
        final = next((e for e in reversed(events) if e.type is EventType.FINAL), None)
        data = final.data if final else {}
        return ChatResponse(
            answer=data.get("answer", ""),
            run_id=events[0].run_id if events else "",
            session_id=body.session_id,
            iterations=data.get("iterations", 0),
            tools_used=data.get("tools_used", []),
            failed=bool(data.get("failed")),
            events=[_public(e) for e in events],
        )

    @app.post("/v1/chat/stream")
    async def chat_stream(
        body: ChatRequest, request: Request, client: str = Depends(limited_client)
    ) -> StreamingResponse:
        text = validate_message(body)
        agent = get_agent()
        session = _scoped(client, body.session_id)

        async def stream() -> AsyncIterator[str]:
            queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

            async def produce() -> None:
                try:
                    async with runs:
                        async for event in agent.run(
                            text, session_id=session, confirm=broker.handler_for(client)
                        ):
                            await queue.put(event)
                finally:
                    await queue.put(None)

            task = asyncio.create_task(produce())
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), KEEPALIVE_SECONDS)
                    except TimeoutError:
                        if await request.is_disconnected():
                            break
                        yield ": keep-alive\n\n"
                        continue
                    if event is None:
                        break
                    yield _sse(event)
            finally:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},  # nginx/caddy: do not buffer
        )

    @app.post("/v1/confirmations/{confirmation_id}")
    async def confirm(
        confirmation_id: str,
        answer: ConfirmationAnswer,
        client: str = Depends(limited_client),
    ) -> dict[str, Any]:
        if not broker.resolve(confirmation_id, client, answer.approve):
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, "No pending confirmation with this id."
            )
        return {"id": confirmation_id, "approved": answer.approve}

    @app.delete("/v1/sessions/{session_id}")
    async def forget(
        session_id: str, client: str = Depends(authenticate)
    ) -> dict[str, str]:
        if not _valid_session(session_id):
            raise HTTPException(422, "Invalid session id.")
        get_agent().memory.clear(_scoped(client, session_id))
        return {"session_id": session_id, "status": "cleared"}

    app.state.broker = broker
    return app


def _scoped(client: str, session_id: str) -> str:
    return f"{client}:{session_id}"


def _valid_session(session_id: str) -> bool:
    import re

    return re.fullmatch(SESSION_PATTERN, session_id) is not None


def _public(event: AgentEvent) -> dict[str, Any]:
    payload = event.to_dict()
    data = dict(payload["data"])
    data.pop("session_id", None)  # internal, scoped id: not the client's business
    payload["data"] = data
    return payload


def _sse(event: AgentEvent) -> str:
    body = json.dumps(_public(event), ensure_ascii=False, default=str)
    return f"event: {event.type}\ndata: {body}\n\n"
