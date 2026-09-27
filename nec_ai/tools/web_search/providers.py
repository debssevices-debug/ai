"""Search providers.

* ``duckduckgo``: free, no key (the ``ddgs`` metasearch library). Default.
* ``brave``: Brave Search API, key in ``SEARCH_API_KEY``.
* ``serper``: Google results through serper.dev, key in ``SEARCH_API_KEY``.

Adding one: subclass :class:`SearchProvider`, then add it to :func:`create_provider`.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from nec_ai.config.settings import Settings
from nec_ai.tools.web_search.base import SearchError, SearchProvider, SearchResult

logger = logging.getLogger("nec.search")


def _clean(text: object) -> str:
    return " ".join(str(text or "").split())


class DuckDuckGoProvider(SearchProvider):
    name = "duckduckgo"

    def __init__(self, timeout: float = 15.0, safesearch: str = "moderate") -> None:
        self.timeout = timeout
        self.safesearch = safesearch

    async def search(
        self, query: str, *, max_results: int = 6, region: str = "fr-fr"
    ) -> list[SearchResult]:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException

        def run() -> list[dict]:
            # DDGS is synchronous and network-bound: keep it off the event loop.
            return DDGS(timeout=self.timeout).text(
                query,
                region=region,
                safesearch=self.safesearch,
                max_results=max_results,
            )

        try:
            rows = await asyncio.wait_for(
                asyncio.to_thread(run), timeout=self.timeout + 5
            )
        except TimeoutError as exc:
            raise SearchError(
                f"DuckDuckGo did not answer within {self.timeout:g}s"
            ) from exc
        except DDGSException as exc:
            message = str(exc)
            if "no results" in message.lower():
                return []
            short = message.splitlines()[0][:160] if message else type(exc).__name__
            raise SearchError(f"DuckDuckGo search failed: {short}") from exc

        return [
            SearchResult(
                title=_clean(row.get("title")) or "(sans titre)",
                url=str(row.get("href", "")).strip(),
                snippet=_clean(row.get("body")),
                source=self.name,
            )
            for row in rows or []
            if row.get("href")
        ]


class _HttpProvider(SearchProvider):
    def __init__(
        self,
        api_key: str,
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ):
        if not api_key:
            raise SearchError(f"{self.name} needs SEARCH_API_KEY")
        self.api_key = api_key
        self.timeout = timeout
        self._client = client

    async def _request(self, method: str, url: str, **kwargs) -> dict:
        try:
            if self._client is not None:
                response = await self._client.request(
                    method, url, timeout=self.timeout, **kwargs
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise SearchError(
                f"{self.name} did not answer within {self.timeout:g}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise SearchError(
                f"{self.name} is unreachable: {type(exc).__name__}"
            ) from exc
        if response.status_code in (401, 403):
            raise SearchError(
                f"{self.name} rejected the API key (HTTP {response.status_code})"
            )
        if response.status_code == 429:
            raise SearchError(f"{self.name} quota or rate limit reached")
        if response.status_code >= 400:
            raise SearchError(f"{self.name} returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise SearchError(f"{self.name} returned an invalid response") from exc


class BraveProvider(_HttpProvider):
    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    async def search(
        self, query: str, *, max_results: int = 6, region: str = "fr-fr"
    ) -> list[SearchResult]:
        country = region.split("-")[-1] if "-" in region else region
        data = await self._request(
            "GET",
            self.endpoint,
            params={"q": query, "count": min(max_results, 20), "country": country},
            headers={
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json",
            },
        )
        rows = (data.get("web") or {}).get("results") or []
        return [
            SearchResult(
                title=_clean(r.get("title")),
                url=str(r.get("url", "")),
                snippet=_clean(r.get("description")),
                source=self.name,
                published=r.get("age"),
            )
            for r in rows[:max_results]
            if r.get("url")
        ]


class SerperProvider(_HttpProvider):
    name = "serper"
    endpoint = "https://google.serper.dev/search"

    async def search(
        self, query: str, *, max_results: int = 6, region: str = "fr-fr"
    ) -> list[SearchResult]:
        language, _, country = region.partition("-")
        data = await self._request(
            "POST",
            self.endpoint,
            json={
                "q": query,
                "num": max_results,
                "gl": country or language,
                "hl": language,
            },
            headers={"X-API-KEY": self.api_key},
        )
        rows = data.get("organic") or []
        return [
            SearchResult(
                title=_clean(r.get("title")),
                url=str(r.get("link", "")),
                snippet=_clean(r.get("snippet")),
                source=self.name,
                published=r.get("date"),
            )
            for r in rows[:max_results]
            if r.get("link")
        ]


def create_provider(settings: Settings) -> SearchProvider:
    key = settings.search_api_key.get_secret_value() if settings.search_api_key else ""
    if settings.search_provider == "brave":
        return BraveProvider(key, settings.search_timeout)
    if settings.search_provider == "serper":
        return SerperProvider(key, settings.search_timeout)
    return DuckDuckGoProvider(settings.search_timeout)
