"""LLM layer: message translation, error mapping, retries. No network."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from google.genai import errors, types

from nec_ai.config.settings import Settings
from nec_ai.llm import LLMError, LLMUnavailableError, Message, ToolCall, create_llm
from nec_ai.llm.base import LLMResponse, complete_with_retry
from nec_ai.llm.fake import FakeLLM
from nec_ai.llm.gemini import GeminiProvider, from_gemini_response, to_gemini_contents
from nec_ai.llm.openai import to_openai_messages
from nec_ai.tools.base import ToolSpec

SPEC = ToolSpec("web_search", "Search.", {"type": "object", "properties": {}})


def _conversation() -> list[Message]:
    call = ToolCall("web_search", {"query": "odigo"}, id="c1")
    call2 = ToolCall("fetch_url", {"url": "https://x.fr"}, id="c2")
    return [
        Message.system("Tu es NEC."),
        Message.user("Cherche Odigo"),
        Message("assistant", "", tool_calls=[call, call2]),
        Message.tool_result(call, "résultats"),
        Message.tool_result(call2, "page"),
    ]


def test_gemini_translation_groups_tool_results() -> None:
    system, contents = to_gemini_contents(_conversation())
    assert system == "Tu es NEC."
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].function_call.name == "web_search"
    responses = [p.function_response for p in contents[2].parts]
    assert [r.name for r in responses] == ["web_search", "fetch_url"]
    assert responses[0].response == {"result": "résultats"}


def test_gemini_reuses_its_own_content_for_thought_signatures() -> None:
    original = types.Content(
        role="model", parts=[types.Part(text="x", thought_signature=b"sig")]
    )
    msg = Message("assistant", "x", provider_data=original, provider="gemini")
    _, contents = to_gemini_contents([Message.user("hi"), msg])
    assert contents[1] is original


def test_gemini_response_extracts_calls_and_skips_thoughts() -> None:
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="je réfléchis", thought=True),
                        types.Part(text="Je cherche."),
                        types.Part(
                            function_call=types.FunctionCall(
                                id="abc", name="web_search", args={"query": "q"}
                            )
                        ),
                    ],
                )
            )
        ]
    )
    result = from_gemini_response(response)
    assert result.text == "Je cherche."
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].id == "abc"
    assert result.tool_calls[0].arguments == {"query": "q"}


class _FailingModels:
    def __init__(self, code: int) -> None:
        self.code = code

    async def generate_content(self, **kwargs):
        raise errors.APIError(self.code, {"error": {"message": "nope", "status": "X"}})


def _gemini_with_error(code: int) -> GeminiProvider:
    client = SimpleNamespace(aio=SimpleNamespace(models=_FailingModels(code)))
    return GeminiProvider("key", client=client)


async def test_gemini_rate_limit_is_retryable() -> None:
    with pytest.raises(LLMUnavailableError):
        await _gemini_with_error(429).complete([Message.user("hi")], [SPEC])


async def test_gemini_bad_key_is_not_retryable() -> None:
    with pytest.raises(LLMError) as info:
        await _gemini_with_error(403).complete([Message.user("hi")])
    assert not isinstance(info.value, LLMUnavailableError)


def test_openai_translation_round_trips_tool_calls() -> None:
    out = to_openai_messages(_conversation())
    assert out[0] == {"role": "system", "content": "Tu es NEC."}
    assert out[2]["tool_calls"][0]["id"] == "c1"
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {
        "query": "odigo"
    }
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "résultats"}


async def test_retry_recovers_from_a_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nec_ai.llm.base.asyncio.sleep", _no_sleep)
    llm = FakeLLM([LLMUnavailableError("503"), LLMResponse(text="ok")])
    result = await complete_with_retry(llm, [Message.user("hi")], retries=2)
    assert result.text == "ok"
    assert len(llm.calls) == 2


async def test_retry_gives_up_after_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nec_ai.llm.base.asyncio.sleep", _no_sleep)
    llm = FakeLLM(
        [LLMUnavailableError("a"), LLMUnavailableError("b"), LLMUnavailableError("c")]
    )
    with pytest.raises(LLMUnavailableError):
        await complete_with_retry(llm, [Message.user("hi")], retries=2)
    assert len(llm.calls) == 3


async def test_permanent_errors_are_not_retried() -> None:
    llm = FakeLLM([LLMError("bad key"), LLMResponse(text="never")])
    with pytest.raises(LLMError):
        await complete_with_retry(llm, [Message.user("hi")], retries=3)
    assert len(llm.calls) == 1


def test_missing_key_is_a_clear_error() -> None:
    with pytest.raises(LLMError, match="GOOGLE_API_KEY"):
        create_llm(Settings(_env_file=None, google_api_key=None))
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        create_llm(Settings(_env_file=None, llm_provider="openai", openai_api_key=None))


def test_providers_are_built_from_settings() -> None:
    gemini = create_llm(Settings(_env_file=None, google_api_key="k", llm_model="m1"))
    assert gemini.name == "gemini" and gemini.model == "m1"
    oa = create_llm(Settings(_env_file=None, llm_provider="openai", openai_api_key="k"))
    assert oa.name == "openai"


async def _no_sleep(_: float) -> None:
    return None
