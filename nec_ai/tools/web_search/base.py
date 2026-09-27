"""Search provider abstraction.

A provider turns a query into a list of :class:`SearchResult`. Nothing else in
the project knows which engine is behind it, so moving from DuckDuckGo to a
paid API (Brave, Serper/Google...) or to the browser is a ``SEARCH_PROVIDER``
change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    """Which provider produced it."""

    published: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


class SearchError(Exception):
    """The provider failed (network, quota, bad key). The message is user-safe."""


class SearchProvider(ABC):
    name: str = "search"

    @abstractmethod
    async def search(
        self, query: str, *, max_results: int = 6, region: str = "fr-fr"
    ) -> list[SearchResult]:
        """Run one query. Raises :class:`SearchError` on failure, [] when empty."""
