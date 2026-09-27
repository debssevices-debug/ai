"""LLM_PROVIDER=claude_code: the local Claude Code CLI, driven by a fake binary."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from nec_ai.agent.events import EventType
from nec_ai.app import build_agent
from nec_ai.config.settings import Settings
from nec_ai.llm import LLMError, LLMUnavailableError, Message, create_llm
from nec_ai.llm.claude_code import (
    ClaudeCodeProvider,
    parse_cli_output,
    render_prompt,
)
from nec_ai.tools.base import ToolSpec

SPEC = ToolSpec("web_search", "Search.", {"type": "object", "properties": {}})


def fake_cli(tmp_path: Path, script: str) -> str:
    """A stand-in for `claude`: Python code that reads stdin and prints JSON."""
    path = tmp_path / "claude"
    path.write_text(
        f"#!{sys.executable}\nimport json, sys\nargs = sys.argv[1:]\n{script}\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def envelope(result: str, **extra) -> str:
    return json.dumps({"type": "result", "is_error": False, "result": result, **extra})


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake binary")


async def test_tool_calls_and_cli_flags(tmp_path: Path) -> None:
    decision = {
        "tool_calls": [{"name": "web_search", "arguments": {"query": "q"}}],
        "answer": "Je cherche.",
    }
    exe = fake_cli(
        tmp_path,
        f"""
prompt = sys.stdin.read()
assert "# OUTILS DISPONIBLES" in prompt and "web_search" in prompt, prompt
assert "[UTILISATEUR]\\nBonjour" in prompt
i = args.index("--tools"); assert args[i + 1] == "", args
assert "-p" in args and "--no-session-persistence" in args
assert args[args.index("--model") + 1] == "sonnet"
print({envelope(json.dumps(decision))!r})
""",
    )
    llm = ClaudeCodeProvider("sonnet", exe)
    result = await llm.complete(
        [Message.system("règles"), Message.user("Bonjour")], [SPEC]
    )
    assert result.text == "Je cherche."
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "q"}


async def test_not_logged_in_is_explained(tmp_path: Path) -> None:
    exe = fake_cli(
        tmp_path,
        f"print({json.dumps({'type': 'result', 'is_error': True, 'result': 'Invalid API key · Please run /login'})!r}); sys.exit(1)",
    )
    with pytest.raises(LLMError, match="/login"):
        await ClaudeCodeProvider("", exe).complete([Message.user("hi")])


async def test_a_missing_cli_is_explained() -> None:
    with pytest.raises(LLMError, match="CLAUDE_CODE_PATH"):
        await ClaudeCodeProvider("", "/nonexistent/claude").complete(
            [Message.user("hi")]
        )


def test_usage_limit_fails_fast_instead_of_waiting() -> None:
    out = envelope("Claude AI usage limit reached|1760000000")
    out = out.replace('"is_error": false', '"is_error": true')
    with pytest.raises(LLMUnavailableError) as info:
        parse_cli_output(1, out, "", True)
    assert info.value.kind == "rate_limit"
    assert info.value.retry_after == 3600  # beyond LLM_MAX_RETRY_WAIT: no retry


def test_fenced_json_is_accepted() -> None:
    text = '```json\n{"tool_calls": [], "answer": "Voilà"}\n```'
    assert parse_cli_output(0, envelope(text), "", True).text == "Voilà"


def test_plain_text_becomes_the_final_answer() -> None:
    result = parse_cli_output(0, envelope("Simple réponse."), "", True)
    assert result.text == "Simple réponse."
    assert result.tool_calls == []


def test_tool_calls_are_ignored_when_no_tool_is_allowed() -> None:
    text = json.dumps(
        {"tool_calls": [{"name": "web_search", "arguments": {}}], "answer": "fin"}
    )
    result = parse_cli_output(0, envelope(text), "", False)
    assert result.tool_calls == [] and result.text == "fin"


def test_the_prompt_carries_the_whole_step() -> None:
    from nec_ai.llm import ToolCall

    call = ToolCall("web_search", {"query": "odigo"})
    prompt = render_prompt(
        [
            Message.system("Tu es NEC."),
            Message.user("Cherche Odigo"),
            Message("assistant", "", tool_calls=[call]),
            Message.tool_result(call, "1. Odigo"),
        ],
        [SPEC],
    )
    assert "Tu es NEC." in prompt
    assert '(appel d\'outil : web_search {"query": "odigo"})' in prompt
    assert "[RÉSULTAT DE web_search]\n1. Odigo" in prompt


def test_the_provider_needs_no_key() -> None:
    llm = create_llm(Settings(_env_file=None, llm_provider="claude_code"))
    assert llm.name == "claude_code"
    assert Settings(_env_file=None, llm_provider="claude_code").llm_timeout == 180


async def test_the_agent_runs_end_to_end_on_the_cli(
    tmp_path: Path, monkeypatch
) -> None:
    """Step 1 the CLI asks for update_plan; step 2 it answers with the plan result seen."""
    exe = fake_cli(
        tmp_path,
        """
prompt = sys.stdin.read()
if "[RÉSULTAT DE update_plan]" in prompt:
    decision = {"tool_calls": [], "answer": "Terminé : plan suivi."}
else:
    decision = {"tool_calls": [{"name": "update_plan", "arguments": {"steps": ["Chercher"]}}], "answer": ""}
print(json.dumps({"type": "result", "is_error": False, "result": json.dumps(decision)}))
""",
    )
    settings = Settings(
        _env_file=None,
        llm_provider="claude_code",
        claude_code_path=exe,
        data_dir=tmp_path,
    )
    agent = build_agent(settings)
    events = [e async for e in agent.run("Fais un plan")]
    assert EventType.PLAN in [e.type for e in events]
    assert events[-1].data["answer"] == "Terminé : plan suivi."
