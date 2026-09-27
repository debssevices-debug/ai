"""Fast web search.

`search_web` is the agent's default way to answer a question that needs the
internet: one query, a short numbered list of results, no browser process. It
costs about two seconds, against roughly ten for driving a headless Chromium
through the same search.

It is a search, not a browser. When the user wants to *see* a results page, or
to look at a specific site, that is `open_search` and `open_page` in the browser
toolset. Keeping the two apart is what stops the agent from launching a browser
to answer "what is the weather in Paris", which is both slow and fragile.

Results are wrapped as untrusted page data, exactly like a browser digest: a
search snippet is text scraped off the open web, and a page can put instructions
in its own meta description. The import below is safe at module scope because
`browser` keeps Playwright out of its import path; only `wrap_untrusted` is
needed here.
"""

import asyncio
import logging

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from livekit.agents import RunContext
from livekit.agents.llm import ToolError, function_tool

from nec_ai.tools.browser.toolset import wrap_untrusted

logger = logging.getLogger("nec.tools")

#: Results handed back to the model. Five is enough to pick from and few enough
#: to read without another page fetch.
MAX_RESULTS = 5

#: Characters of snippet per result. Long enough to judge relevance, short
#: enough that the whole answer stays inside one breath.
SNIPPET_CHARS = 160

#: Wall-clock budget for one query. A search that outlives this has already
#: failed the user, so cancelling it and saying so beats silent audio.
SEARCH_TIMEOUT = 12.0

#: DDGS picks a backend per call and rotates through them. Pinning the region
#: keeps French queries from being answered with region-locked results.
REGION = "fr-fr"
SAFESEARCH = "moderate"


def _format(query: str, results: list[dict]) -> str:
    """Render results as a numbered list the model can read out or pick from."""
    lines = [f'Search results for "{query}" (DuckDuckGo):', ""]
    for position, row in enumerate(results, start=1):
        title = " ".join(str(row.get("title", "")).split()) or "(untitled)"
        href = str(row.get("href", "")).strip()
        snippet = " ".join(str(row.get("body", "")).split())[:SNIPPET_CHARS]
        lines.append(f"{position}. {title}")
        if href:
            lines.append(f"   {href}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    lines.append(
        "Open a result with open_page, or show the whole results page with "
        "open_search. Answer from the snippets above when they are enough, and "
        "say the search was inconclusive when they are not."
    )
    return wrap_untrusted("\n".join(lines))


async def _run_search(query: str) -> list[dict]:
    """One DDGS query, off the event loop and under a wall-clock budget.

    DDGS is synchronous and network-bound. Calling it inline would block the
    loop for the length of the request, which stalls audio and breaks barge-in
    for every other caller on this worker, so it runs in a thread.
    """
    return await asyncio.wait_for(
        asyncio.to_thread(
            lambda: DDGS(timeout=SEARCH_TIMEOUT).text(
                query,
                region=REGION,
                safesearch=SAFESEARCH,
                max_results=MAX_RESULTS,
            )
        ),
        timeout=SEARCH_TIMEOUT,
    )


@function_tool
async def search_web(ctx: RunContext, query: str) -> str:
    """Rechercher sur Internet et obtenir une liste courte de résultats.

    Utilise cet outil pour répondre à une question qui demande des informations
    actuelles, par exemple la météo, une actualité, un prix ou une adresse. C'est
    la façon la plus rapide d'obtenir une réponse : pas de navigateur, environ
    deux secondes.

    Pour un site précis que l'utilisateur nomme, utilise plutôt open_page. Pour
    lui montrer la page de résultats, utilise open_search.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        raise ToolError("No search terms were given.")

    try:
        results = await _run_search(cleaned)
    except TimeoutError:
        logger.warning("web search timed out for %r", cleaned)
        raise ToolError(
            f"The search for {cleaned!r} took too long and was stopped. "
            "Try again, or open the site directly with open_page if the user "
            "named one."
        ) from None
    except DDGSException as exc:
        logger.warning("web search failed for %r: %s", cleaned, exc)
        raise ToolError(
            f"The web search for {cleaned!r} failed ({exc}). "
            "I can try again, or open a site directly with open_page."
        ) from None
    except Exception as exc:
        logger.error("unexpected web search failure for %r", cleaned, exc_info=True)
        raise ToolError(
            f"The web search for {cleaned!r} could not be completed "
            f"({type(exc).__name__})."
        ) from None

    results = [row for row in results or [] if row.get("href")]
    if not results:
        return wrap_untrusted(
            f'No search results for "{cleaned}". The search worked but matched '
            "nothing. Say so plainly rather than inventing an answer, or try "
            "open_page if the user named a site."
        )

    logger.info("web search for %r returned %d result(s)", cleaned, len(results))
    return _format(cleaned, results)
