"""Tool wiring: what the agent actually hands to the model.

Conversations live in ``scenarios.yaml`` and run against the live model. These
are the checks that need no model at all, covering the seam those scenarios sit
on: a tool that exists in a toolset but never reaches the agent fails in
conversation and nowhere else. The commented eval at the bottom is the
in-process framework example (https://docs.livekit.io/agents/start/testing/).
"""

from __future__ import annotations

import pytest
from livekit.agents.llm.tool_context import ToolContext
from livekit.agents.llm.utils import function_arguments_to_pydantic_model

import agent as agent_module
from browser import BrowserConfig, StubBrowserToolset

TOOLS = frozenset(ToolContext(agent_module.build_session_tools()).function_tools)


def test_the_agent_can_search_the_web_without_a_browser() -> None:
    """`search_web` is the fast path, so it must not depend on the toolset.

    If it were registered inside the browser toolset, a disabled browser would
    take web search down with it, and the agent would have no way to look
    anything up.
    """
    assert "search_web" in TOOLS
    assert agent_module.search_web in agent_module.build_session_tools()


def test_the_agent_can_show_a_results_page() -> None:
    assert "open_search" in TOOLS
    assert "open_page" in TOOLS


def test_each_session_gets_its_own_browser_context() -> None:
    """Two live callers must not share cookies, tabs or element refs.

    The toolset used to be built once at import, so every `Assistant` handed to
    LiveKit carried the *same* `BrowserSession` and therefore the same
    `BrowserContext` for the life of the worker. Concurrent users then shared
    cookies -- one user's logged-in session readable by another -- and a `[3]`
    from one conversation could resolve in a different one.

    `BrowserSession.start` already calls `browser.new_context()` per instance, so
    the fix is to stop sharing the toolset rather than to change any browser
    code.
    """
    first = [
        t
        for t in agent_module.build_session_tools()
        if t is not agent_module.search_web
    ]
    second = [
        t
        for t in agent_module.build_session_tools()
        if t is not agent_module.search_web
    ]
    assert first and second
    assert first[0] is not second[0], "the toolset is still shared between sessions"
    assert first[0].session is not second[0].session


def test_every_registered_tool_is_callable_by_the_model() -> None:
    """A schema the model cannot fill is a tool that fails on first use.

    This is the shape of the bug that made the whole browser toolset unusable:
    the tool existed, the toolset built, and every call failed argument
    validation. Asserting on the schemas, not on a call, is what catches it.
    """
    leaky = {"ctx", "context", "self"}
    offenders = []
    for name, tool in ToolContext(
        agent_module.build_session_tools()
    ).function_tools.items():
        schema = function_arguments_to_pydantic_model(tool).model_json_schema()
        bad = leaky & set(schema.get("properties", {}))
        if bad:
            offenders.append(f"{name}: {sorted(bad)}")
    assert not offenders, f"tools exposing their context to the model: {offenders}"


@pytest.mark.parametrize(
    "expected",
    [
        "open_browser",
        "open_search",
        "open_page",
        "read_page",
        "click",
        "browser_status",
    ],
)
def test_the_agent_exposes_the_same_browser_surface_the_stub_simulates(
    expected: str,
) -> None:
    """CI simulates against the stub, so a name the stub lacks is untested surface.

    The stub exists so scenarios can run without a browser. If the real toolset
    grows a tool the stub does not have, every scenario passes while the tool
    ships unexercised.
    """
    real = ToolContext(
        [
            t
            for t in agent_module.build_session_tools()
            if t is not agent_module.search_web
        ]
    ).function_tools
    stub = ToolContext([StubBrowserToolset(BrowserConfig())]).function_tools
    assert expected in real, f"{expected} is missing from the real toolset"
    assert expected in stub, (
        f"{expected} is missing from the stub, so no scenario covers it"
    )


# Agent behavior is covered by the simulations in scenarios.yaml, which run full
# conversations against the agent on LiveKit Cloud (see README.md). The eval
# below is kept as an example of the in-process testing framework
# (https://docs.livekit.io/agents/start/testing/) for turn-level checks that
# don't need a live session. Uncomment it and run `uv run pytest` to use it.
#
# import textwrap
#
# import pytest
# from livekit.agents import AgentSession, inference, llm
#
# from agent import Assistant
#
#
# def _judge_llm() -> llm.LLM:
#     return inference.LLM(model="openai/gpt-4.1-mini")
#
#
# @pytest.mark.asyncio
# async def test_offers_assistance() -> None:
#     """Evaluation of the agent's friendly nature."""
#     async with (
#         _judge_llm() as judge_llm,
#         AgentSession() as session,
#     ):
#         await session.start(Assistant())
#
#         # Run an agent turn following the user's greeting
#         result = await session.run(user_input="Hello")
#
#         # Evaluate the agent's response for friendliness
#         await (
#             result.expect.next_event()
#             .is_message(role="assistant")
#             .judge(
#                 judge_llm,
#                 intent=textwrap.dedent(
#                     """\
#                     Greets the user in a friendly manner.
#
#                     Optional context that may or may not be included:
#                     - Offer of assistance with any request the user may have
#                     - Other small talk or chit chat is acceptable, so long as it is friendly and not too intrusive
#                     """
#                 ),
#             )
#         )
#
#         # Ensures there are no function calls or other unexpected events
#         result.expect.no_more_events()
