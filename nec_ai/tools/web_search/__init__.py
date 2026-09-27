"""Web search: provider abstraction, providers, and the ``web_search`` tool."""

from nec_ai.tools.web_search.base import SearchError, SearchProvider, SearchResult
from nec_ai.tools.web_search.providers import (
    BraveProvider,
    DuckDuckGoProvider,
    SerperProvider,
    create_provider,
)
from nec_ai.tools.web_search.tool import WebSearchTool

__all__ = [
    "BraveProvider",
    "DuckDuckGoProvider",
    "SearchError",
    "SearchProvider",
    "SearchResult",
    "SerperProvider",
    "WebSearchTool",
    "create_provider",
]
