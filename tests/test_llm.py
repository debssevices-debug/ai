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


# ── quota / overload handling (the free-tier scenario) ─────────────────────

REAL_429 = (
    "You exceeded your current quota, please check your plan and billing details.\n"
    "* Quota exceeded for metric: generativelanguage.googleapis.com/"
    "generate_content_free_tier_requests, limit: 5, model: gemini-3.8-flash\n"
    "Please retry in 6.667964504s."
)


def test_a_gemini_quota_error_carries_its_retry_delay() -> None:
    from nec_ai.llm.gemini import translate_error

    error = translate_error(
        errors.APIError(429, {"error": {"code": 429, "message": REAL_429}}), "m"
    )
    assert isinstance(error, LLMUnavailableError)
    assert error.kind == "rate_limit"
    assert error.retry_after == pytest.approx(6.667964504)
    assert "limite 5 requêtes par minute" in str(error)
    assert "\n" not in str(error)


def test_retry_info_details_are_preferred() -> None:
    from nec_ai.llm.gemini import retry_delay

    exc = errors.APIError(
        429,
        {
            "error": {
                "message": "quota",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "12s",
                    }
                ],
            }
        },
    )
    assert retry_delay(exc) == 12.0


def test_an_overloaded_model_is_retryable() -> None:
    from nec_ai.llm.gemini import translate_error

    error = translate_error(
        errors.APIError(503, {"error": {"message": "high demand"}}), "m"
    )
    assert isinstance(error, LLMUnavailableError)
    assert error.kind == "overloaded"


async def test_retry_waits_as_long_as_the_provider_asks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("nec_ai.llm.base.asyncio.sleep", record)
    llm = FakeLLM(
        [
            LLMUnavailableError("quota", retry_after=6.7, kind="rate_limit"),
            LLMResponse(text="ok"),
        ]
    )
    seen = []
    result = await complete_with_retry(
        llm, [Message.user("hi")], on_retry=lambda e, d, a: seen.append((e.kind, d, a))
    )
    assert result.text == "ok"
    assert waits == [pytest.approx(7.2)]
    assert seen == [("rate_limit", pytest.approx(7.2), 1)]


async def test_backoff_grows_when_no_delay_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("nec_ai.llm.base.asyncio.sleep", record)
    llm = FakeLLM([LLMUnavailableError("503")] * 3 + [LLMResponse(text="ok")])
    await complete_with_retry(llm, [Message.user("hi")], retries=3)
    assert waits == [2.0, 4.0, 8.0]


async def test_a_long_quota_wait_fails_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nec_ai.llm.base.asyncio.sleep", _no_sleep)
    llm = FakeLLM(
        [LLMUnavailableError("daily quota", retry_after=3600, kind="rate_limit")]
    )
    with pytest.raises(LLMUnavailableError):
        await complete_with_retry(llm, [Message.user("hi")], max_wait=60)
    assert len(llm.calls) == 1


async def test_the_throttle_spaces_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    from nec_ai.llm.base import RequestThrottle

    clock = [1000.0]
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("nec_ai.llm.base.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("nec_ai.llm.base.asyncio.sleep", fake_sleep)
    throttle = RequestThrottle(per_minute=2)
    notified: list[float] = []
    for _ in range(3):
        await throttle.acquire(notified.append)
        clock[0] += 1
    assert len(waits) == 1
    assert waits[0] == pytest.approx(58.2)
    assert notified == waits


async def test_a_disabled_throttle_never_waits() -> None:
    from nec_ai.llm.base import RequestThrottle

    throttle = RequestThrottle(per_minute=0)
    for _ in range(100):
        await throttle.acquire()


# ── Ollama (local model, no key) ────────────────────────────────────────────


def _ollama(handler):
    import httpx

    from nec_ai.llm.ollama import OllamaProvider

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaProvider("qwen3:8b", client=client)


async def test_ollama_needs_no_key_and_parses_tool_calls() -> None:
    import httpx

    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "<think>hmm</think>Je cherche.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "web_search",
                                "arguments": {"query": "q"},
                            }
                        }
                    ],
                },
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    result = await _ollama(handler).complete(_conversation(), [SPEC], temperature=0.2)
    assert result.text == "Je cherche."
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "q"}
    assert sent["options"]["num_ctx"] == 16384
    assert sent["stream"] is False
    assert sent["tools"][0]["function"]["name"] == "web_search"
    assert sent["messages"][3] == {
        "role": "tool",
        "content": "résultats",
        "tool_name": "web_search",
    }


async def test_ollama_not_running_is_explained() -> None:
    import httpx

    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match=r"ollama\.com"):
        await _ollama(handler).complete([Message.user("hi")])


async def test_ollama_missing_model_says_how_to_install_it() -> None:
    import httpx

    def handler(request):
        return httpx.Response(404, json={"error": "model 'qwen3:8b' not found"})

    with pytest.raises(LLMError, match="ollama pull qwen3:8b"):
        await _ollama(handler).complete([Message.user("hi")])


async def test_ollama_model_without_tools_is_explained() -> None:
    import httpx

    def handler(request):
        return httpx.Response(400, json={"error": "gemma does not support tools"})

    with pytest.raises(LLMError, match="outils"):
        await _ollama(handler).complete([Message.user("hi")], [SPEC])


def test_ollama_is_built_without_any_key() -> None:
    llm = create_llm(Settings(_env_file=None, llm_provider="ollama"))
    assert llm.name == "ollama"
    assert llm.model == "qwen3:8b"


# ── Claude (Anthropic) ───────────────────────────────────────────────────────


class _Recorder:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.kwargs = response, error, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def _claude(response=None, error=None):
    from nec_ai.llm.claude import ClaudeProvider

    recorder = _Recorder(response, error)
    client = SimpleNamespace(beta=SimpleNamespace(messages=recorder))
    return ClaudeProvider("key", client=client), recorder


def _claude_message(content, stop_reason="end_turn"):
    from anthropic.types.beta import BetaMessage

    return BetaMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )


async def test_claude_request_shape_and_tool_call_parsing() -> None:
    response = _claude_message(
        [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Je cherche."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "web_search",
                "input": {"query": "q"},
            },
        ],
        stop_reason="tool_use",
    )
    llm, recorder = _claude(response)
    result = await llm.complete(_conversation(), [SPEC], temperature=0.3)

    sent = recorder.kwargs
    assert sent["model"] == "claude-opus-5"
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["fallbacks"] == "default"
    assert sent["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in sent
    assert sent["system"] == "Tu es NEC."
    assert sent["tools"][0]["input_schema"] == SPEC.parameters
    # Both tool results of the turn travel in one user message.
    last = sent["messages"][-1]
    assert last["role"] == "user"
    assert [b["tool_use_id"] for b in last["content"]] == ["c1", "c2"]

    assert result.text == "Je cherche."
    assert result.tool_calls[0].id == "toolu_1"
    assert result.tool_calls[0].arguments == {"query": "q"}
    # Thinking is kept so it can be echoed back unchanged next turn.
    assert result.provider_data[0]["type"] == "thinking"
    assert result.provider_data[0]["signature"] == "sig"


def test_claude_echoes_its_own_blocks_and_merges_notes() -> None:
    from nec_ai.llm.claude import to_claude_messages

    blocks = [
        {"type": "thinking", "thinking": "", "signature": "s"},
        {"type": "tool_use", "id": "toolu_1", "name": "web_search", "input": {}},
    ]
    call = ToolCall("web_search", {}, id="toolu_1")
    msgs = [
        Message.user("x"),
        Message(
            "assistant", "", tool_calls=[call], provider_data=blocks, provider="claude"
        ),
        Message.tool_result(call, "ERROR: down"),
        Message.user("(note automatique)"),
    ]
    _, out = to_claude_messages(msgs)
    assert out[1] == {"role": "assistant", "content": blocks}
    assert out[2]["content"][0]["is_error"] is True
    assert out[2]["content"][1] == {"type": "text", "text": "(note automatique)"}
    assert len(out) == 3


async def test_claude_refusal_is_a_clear_error() -> None:
    llm, _ = _claude(_claude_message([], stop_reason="refusal"))
    with pytest.raises(LLMError, match="refusé"):
        await llm.complete([Message.user("hi")])


def _api_error(cls, status, headers=None):
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("boom", response=response, body=None)


async def test_claude_rate_limit_carries_retry_after() -> None:
    import anthropic

    llm, _ = _claude(
        error=_api_error(anthropic.RateLimitError, 429, {"retry-after": "12"})
    )
    with pytest.raises(LLMUnavailableError) as info:
        await llm.complete([Message.user("hi")])
    assert info.value.kind == "rate_limit"
    assert info.value.retry_after == 12


async def test_claude_overloaded_is_retryable() -> None:
    import anthropic

    llm, _ = _claude(error=_api_error(anthropic.InternalServerError, 529))
    with pytest.raises(LLMUnavailableError) as info:
        await llm.complete([Message.user("hi")])
    assert info.value.kind == "overloaded"


async def test_claude_bad_key_is_explained() -> None:
    import anthropic

    llm, _ = _claude(error=_api_error(anthropic.AuthenticationError, 401))
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY") as info:
        await llm.complete([Message.user("hi")])
    assert not isinstance(info.value, LLMUnavailableError)


def test_claude_is_built_from_settings() -> None:
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        create_llm(Settings(_env_file=None, llm_provider="claude"))
    llm = create_llm(
        Settings(_env_file=None, llm_provider="claude", anthropic_api_key="k")
    )
    assert llm.name == "claude"
    assert llm.model == "claude-opus-5"
