"""Tool schemas must be fillable by the model.

This is the regression guard for a bug that made the whole browser toolset
unusable while looking perfectly healthy in review.

Every `@function_tool` method took its first argument as `ctx: Any`. The SDK
strips a parameter from the JSON schema it shows the model *only* when the
resolved type hint is a `RunContext` subclass. With `Any` the parameter stayed
in the schema as a required, untyped field, so every call from the model failed
argument validation with `ctx Field required` before the body ever ran:

    open_page  ->  {"properties": {"ctx": {...}, "url": {...}}, "required": ["ctx", "url"]}
    call       ->  ValidationError: ctx Field required

Nothing else in the suite caught it. The unit tests called the Python methods
directly, and the CI stub skipped Playwright, so the only thing that ever
exercised the real schema path was a live session — and it failed there.

The assertions below are deliberately about the *schema* rather than the call
result, because a schema that exposes the context is broken even when a test
happens to pass it one by hand.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import typing

import pytest
from livekit.agents import RunContext
from livekit.agents.llm.utils import function_arguments_to_pydantic_model

import browser
from browser import BrowserConfig, StubBrowserToolset

#: Parameter names that would mean the context leaked into the model's call.
LEAKY = {"ctx", "context", "self", "args", "kwargs"}


def _is_function_tool_decorator(dec: ast.expr) -> bool:
    """True for both `@function_tool` and `@function_tool(...)`."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Name):
        return target.id == "function_tool"
    if isinstance(target, ast.Attribute):
        return target.attr == "function_tool"
    return False


def _toolsets() -> list[object]:
    return [
        browser.BrowserToolset(BrowserConfig()),
        StubBrowserToolset(BrowserConfig()),
    ]


TOOLSETS = _toolsets()


def test_there_are_toolsets_to_check() -> None:
    """A silent rename that empties the suite would make the rest vacuous."""
    for toolset in TOOLSETS:
        assert toolset.tools, f"{type(toolset).__name__} exposed no tools"


@pytest.mark.parametrize("toolset", TOOLSETS, ids=lambda t: type(t).__name__)
def test_no_tool_schema_exposes_the_context(toolset) -> None:
    for tool in toolset.tools:
        schema = function_arguments_to_pydantic_model(tool).model_json_schema()
        properties = set(schema.get("properties", {}))
        leaked = properties & LEAKY
        assert not leaked, (
            f"{tool.info.name} exposes {sorted(leaked)} to the model; the SDK only "
            "hides a context parameter when it is annotated RunContext"
        )


@pytest.mark.parametrize("toolset", TOOLSETS, ids=lambda t: type(t).__name__)
def test_no_tool_requires_the_context(toolset) -> None:
    for tool in toolset.tools:
        schema = function_arguments_to_pydantic_model(tool).model_json_schema()
        required = set(schema.get("required", []))
        assert not (required & LEAKY), (
            f"{tool.info.name} requires {sorted(required & LEAKY)}; every call "
            "will fail validation before the tool body runs"
        )


def test_every_decorated_method_annotates_run_context() -> None:
    """The source-level check, so the mistake cannot hide behind a decorator.

    Reading the annotation off the AST is what makes this a real guard: the
    runtime assertions above only see the schema, and a toolset that happens to
    be disabled in CI would otherwise never be checked.
    """
    path = pathlib.Path(browser.__file__)
    assert path is not None
    tree = ast.parse(path.read_text(encoding="utf-8"))

    offenders: list[str] = []
    seen = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        if not any(_is_function_tool_decorator(dec) for dec in node.decorator_list):
            continue
        seen += 1
        args = [a for a in (*node.args.posonlyargs, *node.args.args) if a.arg != "self"]
        contexts = [a for a in args if a.arg in LEAKY]
        if not contexts:
            offenders.append(f"{node.name}: no context parameter at all")
            continue
        for arg in contexts:
            rendered = ast.unparse(arg.annotation) if arg.annotation else ""
            if rendered not in {"RunContext", "RunContext[typing.Any]"}:
                offenders.append(
                    f"{node.name}: {arg.arg} is annotated {rendered or 'nothing'}, "
                    "which leaves it in the model's schema as a required field"
                )

    assert seen, "no @function_tool methods found; the check is not looking at anything"
    assert not offenders, "\n".join(offenders)


def test_context_is_actually_injected_at_call_time() -> None:
    """End to end: the SDK supplies the context, so the model does not.

    This is the half that was broken in production. A clean schema is only half
    the fix; the value also has to arrive.
    """
    toolset = StubBrowserToolset(BrowserConfig())
    open_page = next(t for t in toolset.tools if t.info.name == "open_page")

    model = function_arguments_to_pydantic_model(open_page)
    assert "url" in model.model_fields
    assert "ctx" not in model.model_fields

    # The tool's first parameter is the context, whatever the model supplied.
    parameters = [
        p for p in inspect.signature(open_page._func).parameters if p != "self"
    ]
    assert parameters[0] == "ctx"


def test_run_context_is_importable_for_get_type_hints() -> None:
    """`from __future__ import annotations` makes hints strings.

    `get_type_hints` resolves them against the module globals, so `RunContext`
    has to exist there or resolution fails and the tool silently reverts to
    leaking `ctx`.
    """
    hints = typing.get_type_hints(browser.StubBrowserToolset._open_page)
    assert hints["ctx"] is RunContext
