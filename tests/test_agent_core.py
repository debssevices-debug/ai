"""The agent loop, driven by a scripted LLM. No network, no API key."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from nec_ai.agent.core import FALLBACK_ANSWER, Agent
from nec_ai.agent.events import EventType
from nec_ai.agent.planner import UpdatePlanTool
from nec_ai.config.settings import Settings
from nec_ai.llm.base import LLMError, LLMResponse, LLMUnavailableError, ToolCall
from nec_ai.llm.fake import FakeLLM
from nec_ai.observability.trace import TraceWriter
from nec_ai.tools.base import PermissionDecision, Tool, ToolContext, ToolResult
from nec_ai.tools.registry import ConfirmationRequest, ToolRegistry


class FakeSearch(Tool):
    name = "web_search"
    description = "Search."
    untrusted_output = True

    class Input(BaseModel):
        query: str

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def run(self, args: Input, ctx: ToolContext) -> ToolResult:
        self.queries.append(args.query)
        return ToolResult.success(f"1. Résultat pour {args.query}")


class Delete(Tool):
    name = "delete"
    description = "Delete a file."

    class Input(BaseModel):
        path: str

    def check_permission(self, args, ctx) -> PermissionDecision:
        return PermissionDecision.confirm("deletes a file")

    async def run(self, args, ctx) -> ToolResult:
        return ToolResult.success(f"deleted {args.path}")


def call(name: str, **arguments) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name, arguments)])


def make_agent(llm: FakeLLM, *tools: Tool, **overrides) -> Agent:
    settings = Settings(_env_file=None, **{"llm_max_retries": 0, **overrides})
    registry = ToolRegistry(tools or (FakeSearch(),))
    return Agent(llm, registry, settings)


async def collect(agent: Agent, request: str, **kwargs):
    return [event async for event in agent.run(request, **kwargs)]


async def test_a_direct_answer_needs_no_tool() -> None:
    agent = make_agent(FakeLLM([LLMResponse(text="Bonjour !")]))
    events = await collect(agent, "Salut")
    types = [e.type for e in events]
    assert types == [EventType.STARTED, EventType.THINKING, EventType.FINAL]
    assert events[-1].data["answer"] == "Bonjour !"


async def test_think_act_observe_then_answer() -> None:
    search = FakeSearch()
    llm = FakeLLM(
        [
            call("web_search", query="Odigo prix"),
            call("web_search", query="Genesys prix"),
            LLMResponse(text="Synthèse finale"),
        ]
    )
    agent = make_agent(llm, search)
    events = await collect(agent, "Compare Odigo et Genesys")

    assert search.queries == ["Odigo prix", "Genesys prix"]
    assert [e.type for e in events].count(EventType.TOOL_COMPLETED) == 2
    final = events[-1]
    assert final.type is EventType.FINAL
    assert final.data["answer"] == "Synthèse finale"
    assert final.data["iterations"] == 3

    # The LLM saw the tool result, wrapped as untrusted data.
    last_messages = llm.calls[-1].messages
    tool_messages = [m for m in last_messages if m.role == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0].content.startswith("<page_data>")
    assert "Résultat pour Odigo prix" in tool_messages[0].content


async def test_tools_are_offered_to_the_llm() -> None:
    llm = FakeLLM([LLMResponse(text="ok")])
    await collect(make_agent(llm), "x")
    assert [t.name for t in llm.calls[0].tools] == ["web_search"]


async def test_the_iteration_limit_forces_a_final_answer() -> None:
    queries = iter(range(100))

    def always_search(messages):
        return call("web_search", query=f"q{next(queries)}")

    def final(messages):
        assert "limite" in messages[-1].content
        return LLMResponse(text="Réponse partielle")

    llm = FakeLLM([always_search] * 3 + [final])
    agent = make_agent(llm, max_agent_iterations=3)
    events = await collect(agent, "boucle")
    assert events[-1].data["answer"] == "Réponse partielle"
    assert events[-1].data["limit_reached"] is True
    assert len(llm.calls) == 4
    assert llm.calls[-1].tools is None  # no tool may be called after the limit


async def test_an_identical_call_is_not_executed_twice() -> None:
    search = FakeSearch()
    llm = FakeLLM(
        [
            call("web_search", query="même"),
            call("web_search", query="même"),
            LLMResponse(text="fin"),
        ]
    )
    events = await collect(make_agent(llm, search), "x")
    assert search.queries == ["même"]
    failed = [e for e in events if e.type is EventType.TOOL_FAILED]
    assert failed and failed[0].data["error"] == "duplicate call skipped"


async def test_an_llm_outage_is_an_error_event_not_a_crash() -> None:
    llm = FakeLLM([LLMUnavailableError("503")])
    events = await collect(make_agent(llm), "x")
    assert EventType.ERROR in [e.type for e in events]
    assert events[-1].type is EventType.FINAL
    assert events[-1].data["answer"] == FALLBACK_ANSWER
    assert events[-1].data["failed"] is True


async def test_an_unknown_tool_is_reported_back_to_the_llm() -> None:
    llm = FakeLLM([call("teleport", where="mars"), LLMResponse(text="désolé")])
    events = await collect(make_agent(llm), "x")
    failed = next(e for e in events if e.type is EventType.TOOL_FAILED)
    assert "Unknown tool" in failed.data["preview"]
    assert events[-1].data["answer"] == "désolé"


async def test_a_confirmation_is_streamed_and_honoured() -> None:
    asked: list[ConfirmationRequest] = []

    async def yes(request: ConfirmationRequest) -> bool:
        asked.append(request)
        return True

    llm = FakeLLM([call("delete", path="a.txt"), LLMResponse(text="Supprimé")])
    events = await collect(make_agent(llm, Delete()), "supprime a.txt", confirm=yes)
    types = [e.type for e in events]
    assert EventType.CONFIRMATION_REQUIRED in types
    assert EventType.CONFIRMATION_RESOLVED in types
    assert asked[0].arguments == {"path": "a.txt"}
    completed = next(e for e in events if e.type is EventType.TOOL_COMPLETED)
    assert "deleted a.txt" in completed.data["preview"]


async def test_without_a_confirmation_handler_risky_tools_are_refused() -> None:
    llm = FakeLLM([call("delete", path="a.txt"), LLMResponse(text="Refusé")])
    events = await collect(make_agent(llm, Delete()), "supprime a.txt")
    failed = next(e for e in events if e.type is EventType.TOOL_FAILED)
    assert "did not approve" in failed.data["preview"]


async def test_the_plan_is_streamed_and_kept_in_working_memory() -> None:
    llm = FakeLLM(
        [
            call("update_plan", steps=["Chercher", "Comparer"]),
            call("update_plan", steps=["Chercher", "Vérifier les prix", "Comparer"]),
            LLMResponse(text="ok"),
        ]
    )
    events = await collect(make_agent(llm, FakeSearch(), UpdatePlanTool()), "x")
    plans = [e for e in events if e.type is EventType.PLAN]
    assert plans[0].data["steps"] == ["Chercher", "Comparer"]
    assert plans[1].data["revised"] is True


async def test_history_gives_continuity_between_requests() -> None:
    llm = FakeLLM(
        [
            call("web_search", query="Odigo"),
            LLMResponse(text="Odigo est une solution de centre de contact."),
            LLMResponse(text="Je continuais sur Odigo."),
        ]
    )
    agent = make_agent(llm)
    await collect(agent, "Cherche Odigo", session_id="s1")
    await collect(agent, "Continue", session_id="s1")
    second = llm.calls[-1].messages
    contents = [m.content for m in second]
    assert "Cherche Odigo" in contents
    assert any("Odigo est une solution" in c for c in contents)
    # Other sessions do not see it.
    await collect(
        make_agent(FakeLLM([LLMResponse(text="x")])), "Continue", session_id="s2"
    )


async def test_every_event_is_traced(tmp_path) -> None:
    llm = FakeLLM([call("web_search", query="q"), LLMResponse(text="fin")])
    settings = Settings(_env_file=None)
    tracer = TraceWriter(tmp_path)
    agent = Agent(llm, ToolRegistry([FakeSearch()]), settings, tracer=tracer)
    events = await collect(agent, "trace-moi")

    files = list(tmp_path.rglob("*.jsonl"))
    assert len(files) == 1
    lines = [json.loads(line) for line in files[0].read_text().splitlines()]
    names = [line["trace"] for line in lines]
    assert names[0] == "USER_REQUEST"
    assert "TOOL_SELECTED" in names and "TOOL_RESULT" in names
    assert names[-1] == "FINAL_RESPONSE"
    assert len(lines) == len(events)


async def test_ask_returns_the_final_answer() -> None:
    agent = make_agent(FakeLLM([LLMResponse(text="42")]))
    assert await agent.ask("?") == "42"


async def test_a_permanent_llm_error_is_not_retried() -> None:
    llm = FakeLLM([LLMError("bad key")])
    events = await collect(make_agent(llm, llm_max_retries=3), "x")
    assert len(llm.calls) == 1
    assert events[-1].data["failed"] is True
    assert "configuration" in events[-1].data["answer"]


@pytest.mark.parametrize("request_text", ["", "   "])
async def test_an_empty_request_is_handled(request_text: str) -> None:
    llm = FakeLLM([])
    events = await collect(make_agent(llm), request_text)
    assert events[-1].type is EventType.FINAL
    assert llm.calls == []
