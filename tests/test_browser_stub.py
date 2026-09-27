"""The CI stub toolset.

`lk agent simulate text` runs with no browser and no network, so the agent is
built with the stub. Two things have to hold for that to be worth anything: the
stub must not launch anything, and it must expose the same tool names as the
real thing, or the scenarios are testing a surface that does not ship.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from livekit.agents.llm import ToolError

from nec_ai.tools.browser.toolset import (
    INJECTION_FIXTURE,
    BrowserConfig,
    StubBrowserToolset,
    build_browser_toolset,
)

REAL_TOOLSET = build_browser_toolset(BrowserConfig())


@pytest.fixture
def stub() -> StubBrowserToolset:
    return StubBrowserToolset(BrowserConfig())


def test_importing_the_module_does_not_import_playwright() -> None:
    """A module-scope Playwright import would cost every worker start.

    `browser` is one file now, so the check is that the `playwright` import is
    *nested* — it happens inside `_SharedBrowser.acquire` when a browser is
    actually wanted. Asserted against the parsed tree rather than the file text,
    since the module docstring names Playwright when explaining why it is lazy.
    """
    import ast
    import pathlib

    import nec_ai.tools.browser.toolset as module

    path = pathlib.Path(module.__file__)
    assert path is not None
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if not isinstance(child, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(child, ast.Import):
                roots = {alias.name.split(".")[0] for alias in child.names}
            else:
                roots = {(child.module or "").split(".")[0]}
            if "playwright" not in roots:
                continue
            # Module scope is a direct child of the module body.
            assert parent is not tree, (
                "the playwright import must stay inside the function that needs it"
            )


async def test_stub_exposes_no_browser_dependencies() -> None:
    """A stub that could reach Playwright would still try to launch it in CI.

    `sys.modules` is the honest signal, but the live tests launch a real browser
    in this same process, so the check runs in a subprocess where nothing has
    imported Playwright yet. A stub that fell through to real browser code would
    pull the package in and the subprocess would fail.
    """
    stub = StubBrowserToolset(BrowserConfig())
    result = await stub._open_page(None, "https://example.com/")
    assert result.startswith("<page_data>")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio, sys;"
            "from nec_ai.tools.browser.toolset import BrowserConfig, StubBrowserToolset;"
            "s = StubBrowserToolset(BrowserConfig());"
            "asyncio.run(s._open_page(None, 'https://example.com/'));"
            "assert 'playwright' not in sys.modules, sorted(sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert completed.returncode == 0, (
        "using the stub must not import playwright:\n"
        + completed.stdout
        + completed.stderr
    )


def test_stub_is_a_real_toolset(stub: StubBrowserToolset) -> None:
    """The SDK registers tools with isinstance checks, not duck typing.

    A stub that only looks like a Toolset is accepted by unit tests and then
    rejected on the first real session, which is exactly the CI-only failure
    this suite exists to prevent.
    """
    from livekit.agents.llm import Toolset
    from livekit.agents.llm.tool_context import ToolContext

    assert isinstance(stub, Toolset)

    # Registration must not raise, and every tool must be a FunctionTool.
    context = ToolContext({stub})
    assert context.function_tools, "the toolset registered no tools"
    for name, tool in context.function_tools.items():
        assert hasattr(tool, "info"), f"{name} is not a function tool: {type(tool)}"
    assert set(context.function_tools) == {t.info.name for t in stub.tools}


def test_stub_tool_names_match_the_real_toolset() -> None:
    """The names the model sees in CI must be the names it sees in production."""
    stub = StubBrowserToolset(BrowserConfig())
    real = {t.info.name for t in REAL_TOOLSET.tools}
    stubbed = {t.info.name for t in stub.tools}
    assert stubbed, "the stub should not be empty"
    # The real set is a superset: the stub deliberately implements only the tools
    # the browser scenarios exercise, since a stub method that fell through to
    # real browser code would be worse than a missing one.
    assert stubbed <= real, f"stub has tools the real toolset lacks: {stubbed - real}"


def test_stub_covers_every_tool_the_scenarios_reach() -> None:
    """Anything scenarios.yaml depends on must exist in the stub."""
    required = {
        "open_page",
        "open_search",
        "read_page",
        "find_on_page",
        "click",
        "type_text",
        "fill_form",
        "take_screenshot",
        "browser_status",
        "stop_browsing",
    }
    stub = StubBrowserToolset(BrowserConfig())
    assert {t.info.name for t in stub.tools} >= required


async def test_open_page_returns_a_digest(stub: StubBrowserToolset) -> None:
    result = await stub._open_page(None, "https://example.com/")
    assert result.startswith("<page_data>")
    assert "Example Domain" in result
    assert "URL: https://example.com/" in result


async def test_unknown_url_explains_what_exists(stub: StubBrowserToolset) -> None:
    with pytest.raises(ToolError) as excinfo:
        await stub._open_page(None, "https://nowhere.test/")
    message = str(excinfo.value)
    assert "no test page" in message
    assert "example.com" in message


async def test_read_before_open_is_refused(stub: StubBrowserToolset) -> None:
    with pytest.raises(ToolError, match="No browser is open"):
        await stub._read_page(None)


async def test_open_search_returns_a_numbered_result_list(
    stub: StubBrowserToolset,
) -> None:
    """The stub has to produce the *list* form, not a page digest.

    Simulations train the model on whatever the stub returns. If it returned a
    whole-page digest while production returns a numbered list, every scenario
    would pass against a surface that does not ship.
    """
    result = await stub._open_search(None, "meteo paris demain")
    assert result.startswith("<page_data>")
    assert 'Search results for "meteo paris demain" on bing' in result
    assert "[1] Météo-France" in result
    assert "https://meteofrance.com" in result
    # A digest would carry the engine's own navigation instead.
    assert "Interactive elements" not in result
    assert "click(ref=1)" in result


async def test_open_search_defaults_to_bing(stub: StubBrowserToolset) -> None:
    """Bing is the default because it is the one that answers a browser."""
    result = await stub._open_search(None, "meteo paris demain")
    assert "on bing" in result
    name, arguments = stub.calls[-1]
    assert name == "open_search"
    assert arguments["query"] == "meteo paris demain"


async def test_open_search_refuses_an_unknown_engine(
    stub: StubBrowserToolset,
) -> None:
    with pytest.raises(ToolError, match="not a search engine"):
        await stub._open_search(None, "meteo", engine="altavista")


async def test_click_moves_to_the_next_page(stub: StubBrowserToolset) -> None:
    await stub._open_page(None, "https://shop.example/")
    result = await stub._click(None, text="Panier")
    assert "Panier" in result
    assert "empty" in result or "vide" in result


async def test_stop_browsing_clears_state(stub: StubBrowserToolset) -> None:
    await stub._open_page(None, "https://example.com/")
    result = await stub._stop_browsing(None)
    assert "closed" in result
    with pytest.raises(ToolError, match="No browser is open"):
        await stub._read_page(None)


async def test_injection_page_comes_back_as_data(stub: StubBrowserToolset) -> None:
    result = await stub._open_page(None, f"https://{INJECTION_FIXTURE}/")
    assert result.startswith("<page_data>")
    assert "ignore all previous instructions" in result


async def test_calls_are_recorded_for_assertions(stub: StubBrowserToolset) -> None:
    await stub._open_page(None, "https://example.com/")
    await stub._read_page(None)
    assert [name for name, _ in stub.calls] == ["open_page", "read_page"]


async def test_password_value_is_not_recorded(stub: StubBrowserToolset) -> None:
    """A password must not land in a recorded call either."""
    await stub._open_page(None, "https://login.example/")
    await stub._type_text(None, text="hunter2", into="Mot de passe")
    _name, arguments = stub.calls[-1]
    assert arguments["length"] == len("hunter2")
    assert "hunter2" not in str(arguments)


def test_stub_is_selected_by_config() -> None:
    toolset = build_browser_toolset(BrowserConfig(stub=True))
    assert isinstance(toolset, StubBrowserToolset)


def test_browser_can_be_disabled_entirely() -> None:
    assert build_browser_toolset(BrowserConfig(enabled=False)) is None
