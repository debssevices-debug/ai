"""The REST API: auth, rate limits, validation, chat, streaming, confirmations."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn
from pydantic import BaseModel

from nec_ai.agent.core import Agent
from nec_ai.agent.events import EventType
from nec_ai.api.security import auth_problem
from nec_ai.api.server import create_app
from nec_ai.client import RemoteClient, RemoteError
from nec_ai.config.settings import Settings
from nec_ai.llm.base import LLMResponse, ToolCall
from nec_ai.llm.fake import FakeLLM
from nec_ai.tools.base import PermissionDecision, Tool, ToolContext, ToolResult
from nec_ai.tools.registry import ToolRegistry

KEY_A = "a" * 32
KEY_B = "b" * 32


class Delete(Tool):
    name = "delete_file"
    description = "Delete a file."

    class Input(BaseModel):
        path: str

    def check_permission(self, args, ctx) -> PermissionDecision:
        return PermissionDecision.confirm("supprime un fichier")

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        return ToolResult.success(f"deleted {args.path}")


def echo_llm() -> FakeLLM:
    """Answers with what it saw, so tests can check history and results."""

    def respond(messages):
        last = messages[-1]
        if last.role == "user" and last.content.startswith("supprime"):
            return LLMResponse(tool_calls=[ToolCall("delete_file", {"path": "a.txt"})])
        if last.role == "tool":
            return LLMResponse(text=f"résultat: {last.content}")
        history = [m.content for m in messages if m.role == "user"]
        return LLMResponse(text=f"vu: {' | '.join(history)}")

    return FakeLLM([respond] * 1000)


def settings(**overrides) -> Settings:
    base = {
        "api_keys": [KEY_A, KEY_B],
        "rate_limit_per_minute": 100,
        "trace_enabled": False,
    }
    return Settings(_env_file=None, **{**base, **overrides})


def make_app(**overrides):
    s = settings(**overrides)
    agent = Agent(echo_llm(), ToolRegistry([Delete()]), s)
    return create_app(s, agent=agent)


def client_for(app, key: str | None = KEY_A) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers
    )


# ── configuration safety ────────────────────────────────────────────────────


def test_the_server_refuses_to_run_without_a_key() -> None:
    assert "nec new-key" in auth_problem(Settings(_env_file=None))


def test_no_auth_is_only_allowed_on_localhost() -> None:
    assert auth_problem(Settings(_env_file=None, api_allow_no_auth=True)) is None
    exposed = Settings(_env_file=None, api_allow_no_auth=True, api_host="0.0.0.0")
    assert "127.0.0.1" in auth_problem(exposed)


def test_weak_keys_are_refused() -> None:
    assert "24" in auth_problem(Settings(_env_file=None, api_keys=["short"]))


# ── auth and limits ─────────────────────────────────────────────────────────


async def test_health_is_public() -> None:
    async with client_for(make_app(), key=None) as c:
        r = await c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("key", [None, "wrong-key-wrong-key-wrong-key"])
async def test_v1_needs_a_valid_key(key) -> None:
    async with client_for(make_app(), key=key) as c:
        r = await c.get("/v1/tools")
    assert r.status_code == 401


async def test_docs_are_hidden_when_keys_are_configured() -> None:
    async with client_for(make_app(), key=None) as c:
        assert (await c.get("/docs")).status_code == 404
        assert (await c.get("/openapi.json")).status_code == 404


async def test_tools_are_listed() -> None:
    async with client_for(make_app()) as c:
        r = await c.get("/v1/tools")
    assert [t["name"] for t in r.json()] == ["delete_file"]


async def test_rate_limit_answers_429_with_retry_after() -> None:
    async with client_for(make_app(rate_limit_per_minute=2)) as c:
        for _ in range(2):
            assert (await c.post("/v1/chat", json={"message": "x"})).status_code == 200
        r = await c.post("/v1/chat", json={"message": "x"})
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1


# ── validation ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"message": ""}, 422),
        ({"message": "   "}, 422),
        ({"message": "x" * 9000}, 413),
        ({"message": "x", "session_id": "../etc"}, 422),
        ({}, 422),
    ],
)
async def test_bad_input_is_rejected(body, code) -> None:
    async with client_for(make_app()) as c:
        r = await c.post("/v1/chat", json=body)
    assert r.status_code == code


# ── chat ────────────────────────────────────────────────────────────────────


async def test_chat_returns_the_answer_and_events() -> None:
    async with client_for(make_app()) as c:
        r = await c.post("/v1/chat", json={"message": "bonjour"})
    body = r.json()
    assert body["answer"] == "vu: bonjour"
    assert body["events"][0]["type"] == "agent.started"
    assert body["events"][-1]["type"] == "agent.final"
    assert "session_id" not in body["events"][0]["data"]  # internal scoped id hidden


async def test_sessions_are_private_to_each_key() -> None:
    app = make_app()
    async with client_for(app, KEY_A) as a, client_for(app, KEY_B) as b:
        await a.post("/v1/chat", json={"message": "secret A"})
        r = await b.post("/v1/chat", json={"message": "question B"})
        assert "secret A" not in r.json()["answer"]
        r = await a.post("/v1/chat", json={"message": "suite"})
        assert "secret A" in r.json()["answer"]


async def test_a_session_can_be_forgotten() -> None:
    app = make_app()
    async with client_for(app) as c:
        await c.post("/v1/chat", json={"message": "souviens-toi", "session_id": "s1"})
        assert (await c.delete("/v1/sessions/s1")).status_code == 200
        r = await c.post("/v1/chat", json={"message": "et alors", "session_id": "s1"})
    assert "souviens-toi" not in r.json()["answer"]


async def test_risky_actions_are_refused_without_a_stream() -> None:
    async with client_for(make_app()) as c:
        r = await c.post("/v1/chat", json={"message": "supprime a.txt"})
    body = r.json()
    assert "did not approve" in body["answer"]
    assert any(e["type"] == "confirmation.required" for e in body["events"])


async def test_a_missing_llm_key_is_a_503() -> None:
    s = settings(llm_provider="gemini", google_api_key=None)
    async with client_for(create_app(s)) as c:
        r = await c.post("/v1/chat", json={"message": "x"})
    assert r.status_code == 503
    assert "GOOGLE_API_KEY" in r.json()["detail"]


# ── streaming against a real server ─────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def live_server() -> Iterator[str]:
    port = _free_port()
    config = uvicorn.Config(
        make_app(confirmation_timeout=10),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


async def test_events_are_streamed(live_server: str) -> None:
    async with RemoteClient(live_server, KEY_A) as client:
        events = [e async for e in client.stream("bonjour")]
    types = [e.type for e in events]
    assert types[0] is EventType.STARTED
    assert types[-1] is EventType.FINAL
    assert events[-1].data["answer"] == "vu: bonjour"


async def test_a_remote_confirmation_approves_the_action(live_server: str) -> None:
    async with RemoteClient(live_server, KEY_A) as client:
        answer = ""
        async for event in client.stream("supprime a.txt"):
            if event.type is EventType.CONFIRMATION_REQUIRED:
                assert event.data["tool"] == "delete_file"
                await client.confirm(event.data["id"], approve=True)
            if event.type is EventType.FINAL:
                answer = event.data["answer"]
    assert answer == "résultat: deleted a.txt"


async def test_another_key_cannot_answer_my_confirmation(live_server: str) -> None:
    async with (
        RemoteClient(live_server, KEY_A) as mine,
        RemoteClient(live_server, KEY_B) as other,
    ):
        answer = ""
        async for event in mine.stream("supprime a.txt"):
            if event.type is EventType.CONFIRMATION_REQUIRED:
                with pytest.raises(RemoteError, match="404"):
                    await other.confirm(event.data["id"], approve=True)
                await mine.confirm(event.data["id"], approve=False)
            if event.type is EventType.FINAL:
                answer = event.data["answer"]
    assert "did not approve" in answer


async def test_a_bad_key_is_explained_to_the_client(live_server: str) -> None:
    async with RemoteClient(live_server, "nope" * 8) as client:
        with pytest.raises(RemoteError, match="NEC_API_KEY"):
            async for _ in client.stream("x"):
                pass


async def test_an_unreachable_server_is_explained() -> None:
    async with RemoteClient(f"http://127.0.0.1:{_free_port()}", KEY_A) as client:
        with pytest.raises(RemoteError, match="injoignable"):
            await client.health()


async def test_disconnecting_cancels_the_run(live_server: str) -> None:
    """Leaving mid-stream must not leave a run (and its tools) going on the server."""
    async with RemoteClient(live_server, KEY_A) as client:
        async for event in client.stream("supprime a.txt"):
            if event.type is EventType.CONFIRMATION_REQUIRED:
                break  # client goes away without answering
    await asyncio.sleep(0.5)
    async with RemoteClient(live_server, KEY_A) as client:
        events = [e async for e in client.stream("bonjour")]
    assert events[-1].type is EventType.FINAL
