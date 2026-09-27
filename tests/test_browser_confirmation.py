"""A consequential action must be confirmed by the user before it happens.

The requirement is that NEC_AI never auto-submits a form, takes a payment, or
changes an account without the user saying yes. Written only in the tool
descriptions and the instructions, that is a suggestion to the model. The gate
below makes it a property of the code: a commit raises unless the caller passes
``confirmed=True``.

The tests are split by what could break:

* **Classification** - whether a call is treated as a commit at all. This is pure
  and carries the most weight, because a tool wrongly classed as a read is a
  silently ungated path.
* **The refusal** - what the model is told. A refusal the model cannot act on
  produces a tool that fails once and is never retried, which is the same
  failure as having no gate at all.
* **The stub** - the same gate behind the same wording, because the stub is what
  CI actually trains the model against.

Reads and unfilled form state are deliberately never gated; a test asserts that
too, since a gate that is too broad is its own failure mode.
"""

from __future__ import annotations

import pytest
from livekit.agents.llm import ToolError

from nec_ai.tools.browser.toolset import (
    ActionRisk,
    BrowserConfig,
    StubBrowserToolset,
    classify_action,
    confirmation_refusal,
    plan_commits,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════
#
# A tool misclassified as READ is a gate that silently does not apply, and it is
# the kind of bug no behavioural test catches: the happy path is identical either
# way. `press_key` was exactly this before the Enter rule was added — it had no
# `submit` argument to key off, so it rated as a plain read while being the most
# direct way to send a form.


def test_enter_counts_as_a_commit() -> None:
    """Enter submits the focused form, so pressing it is not a read."""
    assert classify_action("press_key", {"key": "Enter"}) is ActionRisk.COMMIT


@pytest.mark.parametrize("spelling", ["Enter", "enter", "  ENTER  ", "return"])
def test_enter_is_matched_whatever_the_spelling(spelling: str) -> None:
    assert classify_action("press_key", {"key": spelling}) is ActionRisk.COMMIT


@pytest.mark.parametrize("key", ["Tab", "Escape", "ArrowDown", "PageUp", "F5"])
def test_other_keys_are_not_commits(key: str) -> None:
    """A gate that fires on Tab trains the user to approve harmless keys."""
    assert classify_action("press_key", {"key": key}) is ActionRisk.READ


def test_a_submit_argument_is_a_commit() -> None:
    assert classify_action("type_text", {"submit": True}) is ActionRisk.COMMIT
    assert classify_action("fill_form", {"submit": True}) is ActionRisk.COMMIT


def test_filling_without_submitting_is_not_a_commit() -> None:
    """Local form state is not an external effect."""
    assert classify_action("type_text", {"submit": False}) is ActionRisk.FILL
    assert classify_action("fill_form", {"submit": False}) is ActionRisk.FILL
    assert classify_action("select_option", {}) is ActionRisk.FILL
    assert classify_action("upload_file", {}) is ActionRisk.FILL


def test_a_download_is_a_commit() -> None:
    """A download puts something on the machine, often a statement or invoice."""
    assert classify_action("download_file", {}) is ActionRisk.COMMIT


@pytest.mark.parametrize(
    "name",
    ["read_page", "click", "scroll", "open_page", "open_search", "list_links"],
)
def test_reads_are_never_commits(name: str) -> None:
    assert classify_action(name, {}) is ActionRisk.READ


# ═══════════════════════════════════════════════════════════════════════════
# 2. RUN_STEPS
# ═══════════════════════════════════════════════════════════════════════════
#
# run_steps is classed a commit wholesale, because it is the general escape
# hatch. Gating every plan would be wrong in the other direction: most plans are
# clicks and scrolls, and an agent that interrupts to ask permission before
# ordinary navigation is an agent whose user stops listening to it.


def test_a_plan_that_presses_enter_commits() -> None:
    assert plan_commits([{"action": "press", "key": "Enter"}])


def test_a_plan_of_clicks_and_typing_does_not_commit() -> None:
    plan = [
        {"action": "click", "text": "Panier"},
        {"action": "type", "into": "Recherche", "text": "pizza"},
        {"action": "scroll", "direction": "down"},
    ]
    assert not plan_commits(plan)


def test_a_plan_that_presses_a_harmless_key_does_not_commit() -> None:
    assert not plan_commits([{"action": "press", "key": "Tab"}])


def test_an_empty_or_malformed_plan_does_not_commit() -> None:
    assert not plan_commits([])
    assert not plan_commits(None)
    assert not plan_commits(["not a dict", 3, None])


# ═══════════════════════════════════════════════════════════════════════════
# 3. THE REFUSAL
# ═══════════════════════════════════════════════════════════════════════════
#
# This text is the load-bearing part. A realtime model that is handed a bare
# failure treats the tool as broken and moves on; it does not infer that the
# remedy is to ask a human a question. So the refusal has to say what to do next.


def test_the_refusal_names_the_action() -> None:
    assert "Downloading 'Facture'" in confirmation_refusal("Downloading 'Facture'")


def test_the_refusal_tells_the_model_to_ask_and_retry() -> None:
    text = confirmation_refusal("Sending the form")
    assert "Ask the user" in text
    assert "confirmed=True" in text


def test_the_refusal_forbids_claiming_success() -> None:
    """Otherwise the agent reports a purchase it never made."""
    assert "Do not report it as done" in confirmation_refusal("Sending the form")


def test_the_refusal_forbids_a_silent_retry() -> None:
    """Otherwise the agent asks, gets ignored, and submits anyway."""
    assert "do not retry without asking" in confirmation_refusal("Sending the form")


# ═══════════════════════════════════════════════════════════════════════════
# 4. THE STUB
# ═══════════════════════════════════════════════════════════════════════════
#
# CI runs the scenarios against the stub. If the stub accepted an unconfirmed
# submit, every scenario would pass while the real toolset refused in
# production, and the model would be trained to retry without asking.


@pytest.fixture
def stub() -> StubBrowserToolset:
    return StubBrowserToolset(BrowserConfig())


async def _on_login(stub: StubBrowserToolset) -> None:
    await stub._open_page(_ctx(), url="https://login.example")


class _Ctx:
    """The stub's tools ignore the context entirely, so an empty stand-in is
    enough and the tests stay off a real session."""

    session = None


def _ctx() -> _Ctx:
    return _Ctx()


async def test_the_stub_refuses_an_unconfirmed_submit(stub: StubBrowserToolset) -> None:
    await _on_login(stub)
    with pytest.raises(ToolError) as caught:
        await stub._type_text(_ctx(), text="test@example.test", ref=1, submit=True)
    assert "confirmed=True" in str(caught.value)


async def test_the_stub_allows_a_confirmed_submit(stub: StubBrowserToolset) -> None:
    await _on_login(stub)
    result = await stub._type_text(
        _ctx(), text="test@example.test", ref=1, submit=True, confirmed=True
    )
    assert "Pressed Enter to send" in result


async def test_the_stub_never_gates_a_plain_fill(stub: StubBrowserToolset) -> None:
    """Filling without sending must keep working with no confirmation."""
    await _on_login(stub)
    result = await stub._type_text(_ctx(), text="test@example.test", ref=1)
    assert "Typed into" in result


async def test_the_stub_never_gates_a_plain_fill_form(
    stub: StubBrowserToolset,
) -> None:
    await _on_login(stub)
    result = await stub._fill_form(_ctx(), fields={"Adresse e-mail": "a@b.test"})
    assert "Filled 1 field" in result


async def test_the_stub_refuses_an_unconfirmed_fill_form_submit(
    stub: StubBrowserToolset,
) -> None:
    await _on_login(stub)
    with pytest.raises(ToolError) as caught:
        await stub._fill_form(
            _ctx(), fields={"Adresse e-mail": "a@b.test"}, submit=True
        )
    assert "confirmed=True" in str(caught.value)


async def test_the_stub_allows_a_confirmed_fill_form_submit(
    stub: StubBrowserToolset,
) -> None:
    await _on_login(stub)
    result = await stub._fill_form(
        _ctx(), fields={"Adresse e-mail": "a@b.test"}, submit=True, confirmed=True
    )
    assert "send the form" in result


async def test_the_stub_records_the_confirmation_for_assertions(
    stub: StubBrowserToolset,
) -> None:
    """Scenarios assert on recorded calls, so the flag has to land there."""
    await _on_login(stub)
    await stub._type_text(_ctx(), text="x", ref=1, submit=True, confirmed=True)
    name, arguments = stub.calls[-1]
    assert name == "type_text"
    assert arguments["confirmed"] is True


# ═══════════════════════════════════════════════════════════════════════════
# 5. SCHEMA
# ═══════════════════════════════════════════════════════════════════════════
#
# The flag is only a gate if the model can actually see and set it. A parameter
# missing from the schema is a gate that is always closed, and one the model
# cannot fill is a tool that fails on every call.


@pytest.mark.parametrize(
    "tool_name", ["type_text", "fill_form", "press_key", "run_steps", "download_file"]
)
def test_confirmed_is_offered_to_the_model(tool_name: str) -> None:
    from livekit.agents.llm.utils import function_arguments_to_pydantic_model

    from nec_ai.tools.browser.toolset import BrowserToolset

    # The real toolset, not the stub: it is the surface that ships, and the stub
    # deliberately implements only a subset. Constructing it launches nothing —
    # the browser starts on the first open_page.
    toolset = BrowserToolset(BrowserConfig())
    tool = next(t for t in toolset.tools if t.info.name == tool_name)
    schema = function_arguments_to_pydantic_model(tool).model_json_schema()

    assert "confirmed" in schema.get("properties", {}), (
        f"{tool_name} does not expose `confirmed`, so the model cannot ever "
        "unblock the gate"
    )
    # Optional, so an ordinary call does not have to invent it.
    assert "confirmed" not in schema.get("required", [])
    assert schema["properties"]["confirmed"].get("default") is False


def test_every_gated_tool_mentions_the_gate_in_its_description() -> None:
    """The description is the only thing the model reads before deciding.

    Asserted across the whole gated set, because a tool that takes the flag
    without explaining it is a flag the model never thinks to send.
    """
    from nec_ai.tools.browser.toolset import BrowserToolset

    toolset = BrowserToolset(BrowserConfig())
    gated = {"type_text", "fill_form", "press_key", "run_steps", "download_file"}
    seen = set()
    for tool in toolset.tools:
        if tool.info.name not in gated:
            continue
        seen.add(tool.info.name)
        description = (tool.info.description or "").lower()
        assert "confirmed" in description, f"{tool.info.name} hides the gate"
    assert seen == gated, f"not found: {sorted(gated - seen)}"
