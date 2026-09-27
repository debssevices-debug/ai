"""Web search: the tool, and each provider's parsing. No real network."""

from __future__ import annotations

import httpx
import pytest

from nec_ai.config.settings import Settings
from nec_ai.tools.base import ToolContext
from nec_ai.tools.registry import ToolRegistry
from nec_ai.tools.web_search import (
    BraveProvider,
    DuckDuckGoProvider,
    SearchError,
    SearchProvider,
    SearchResult,
    SerperProvider,
    WebSearchTool,
    create_provider,
)


class StaticProvider(SearchProvider):
    name = "static"

    def __init__(self, results=None, error: Exception | None = None) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[tuple[str, int, str]] = []

    async def search(self, query, *, max_results=6, region="fr-fr"):
        self.calls.append((query, max_results, region))
        if self.error:
            raise self.error
        return self.results[:max_results]


RESULTS = [
    SearchResult(
        "Odigo - Tarifs", "https://www.odigo.com/fr/tarifs", "Prix par agent", "static"
    ),
    SearchResult(
        "Genesys Cloud", "https://www.genesys.com", "CCaaS", "static", "2026-09-01"
    ),
]


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(settings=Settings(_env_file=None, search_max_results=4))


async def run(tool: WebSearchTool, ctx: ToolContext, **arguments):
    return (await ToolRegistry([tool]).execute("web_search", arguments, ctx)).result


async def test_results_are_structured_and_readable(ctx: ToolContext) -> None:
    provider = StaticProvider(RESULTS)
    result = await run(
        WebSearchTool(provider), ctx, query="prix Odigo centre de contact 50 agents"
    )
    assert result.ok
    assert "1. Odigo - Tarifs" in result.content
    assert "https://www.odigo.com/fr/tarifs" in result.content
    assert "date : 2026-09-01" in result.content
    assert result.content.startswith("<page_data>")  # snippets are untrusted data
    assert result.data["results"][0]["url"] == "https://www.odigo.com/fr/tarifs"
    assert provider.calls == [("prix Odigo centre de contact 50 agents", 4, "fr-fr")]


async def test_max_results_can_be_asked_for(ctx: ToolContext) -> None:
    provider = StaticProvider(RESULTS)
    await run(WebSearchTool(provider), ctx, query="x", max_results=1)
    assert provider.calls[0][1] == 1


async def test_an_empty_search_says_so_and_suggests_rephrasing(
    ctx: ToolContext,
) -> None:
    result = await run(WebSearchTool(StaticProvider([])), ctx, query="zzzz")
    assert result.ok
    assert "Aucun résultat" in result.content
    assert "Reformule" in result.content


async def test_a_provider_failure_is_a_readable_error(ctx: ToolContext) -> None:
    provider = StaticProvider(error=SearchError("quota reached"))
    result = await run(WebSearchTool(provider), ctx, query="x")
    assert not result.ok
    assert "quota reached" in result.content


async def test_an_empty_query_is_rejected(ctx: ToolContext) -> None:
    result = await run(WebSearchTool(StaticProvider(RESULTS)), ctx, query="")
    assert not result.ok


def test_the_provider_is_chosen_by_configuration() -> None:
    assert isinstance(create_provider(Settings(_env_file=None)), DuckDuckGoProvider)
    brave = create_provider(
        Settings(_env_file=None, search_provider="brave", search_api_key="k")
    )
    assert isinstance(brave, BraveProvider)
    serper = create_provider(
        Settings(_env_file=None, search_provider="serper", search_api_key="k")
    )
    assert isinstance(serper, SerperProvider)


def test_a_paid_provider_without_key_fails_clearly() -> None:
    with pytest.raises(SearchError, match="SEARCH_API_KEY"):
        create_provider(Settings(_env_file=None, search_provider="brave"))


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_brave_results_are_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Subscription-Token"] == "k"
        assert request.url.params["q"] == "odigo"
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Odigo",
                            "url": "https://odigo.com",
                            "description": "CCaaS",
                        }
                    ]
                }
            },
        )

    results = await BraveProvider("k", client=_client(handler)).search("odigo")
    assert results == [SearchResult("Odigo", "https://odigo.com", "CCaaS", "brave")]


async def test_serper_results_are_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-KEY"] == "k"
        return httpx.Response(
            200,
            json={"organic": [{"title": "T", "link": "https://t.fr", "snippet": "S"}]},
        )

    results = await SerperProvider("k", client=_client(handler)).search("q")
    assert results[0].url == "https://t.fr"
    assert results[0].source == "serper"


@pytest.mark.parametrize(
    ("status", "message"), [(401, "API key"), (429, "quota"), (500, "HTTP 500")]
)
async def test_http_errors_become_search_errors(status: int, message: str) -> None:
    provider = BraveProvider("k", client=_client(lambda r: httpx.Response(status)))
    with pytest.raises(SearchError, match=message):
        await provider.search("q")


async def test_a_network_failure_becomes_a_search_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(SearchError, match="unreachable"):
        await BraveProvider("k", client=_client(handler)).search("q")


async def test_duckduckgo_rows_are_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDDGS:
        def __init__(self, timeout):
            pass

        def text(self, query, **kwargs):
            return [
                {
                    "title": " Odigo \n tarifs ",
                    "href": "https://odigo.com",
                    "body": "a  b",
                },
                {"title": "sans lien", "href": "", "body": "x"},
            ]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    results = await DuckDuckGoProvider().search("odigo")
    assert results == [
        SearchResult("Odigo tarifs", "https://odigo.com", "a b", "duckduckgo")
    ]


async def test_duckduckgo_errors_are_shortened(monkeypatch: pytest.MonkeyPatch) -> None:
    from ddgs.exceptions import DDGSException

    class BrokenDDGS:
        def __init__(self, timeout):
            pass

        def text(self, query, **kwargs):
            raise DDGSException("proxy refused\nlots of details " + "x" * 500)

    monkeypatch.setattr("ddgs.DDGS", BrokenDDGS)
    with pytest.raises(SearchError) as info:
        await DuckDuckGoProvider().search("odigo")
    assert "lots of details" not in str(info.value)
