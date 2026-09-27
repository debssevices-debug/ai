"""``fetch_url``: read a web page as text, without a browser.

Fast (one HTTP request) and enough for most pages: articles, documentation,
pricing pages rendered server-side. Pages that need JavaScript or interaction
are the browser's job.

Safety: URL policy and DNS check on the first request and on every redirect,
bounded download size, text only, output marked as untrusted data.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import BaseModel, Field

from nec_ai.security.net import check_resolved, check_url
from nec_ai.tools.base import Tool, ToolContext, ToolResult

MAX_REDIRECTS = 5
USER_AGENT = "Mozilla/5.0 (compatible; NEC-AI/0.2; +personal assistant)"

_SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "head"}
_BLOCK = {
    "p", "div", "section", "article", "main", "header", "footer", "nav", "aside",
    "li", "ul", "ol", "table", "tr", "br", "hr", "h1", "h2", "h3", "h4", "h5",
    "h6", "pre", "blockquote", "dd", "dt", "form", "figure", "figcaption",
}  # fmt: skip


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._in_title = True
        if tag in _SKIP:
            self._skip_depth += 1
        elif tag in _BLOCK:
            self._chunks.append("\n")
        if tag in {"h1", "h2", "h3"}:
            self._chunks.append("# ")
        if tag == "li":
            self._chunks.append("- ")
        if tag in {"td", "th"}:
            self._chunks.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in _SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if not self._skip_depth:
            self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n"))
        out: list[str] = []
        for line in lines:
            if line or (out and out[-1]):
                out.append(line)
        return "\n".join(out).strip()


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, readable text) for an HTML document."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup: keep whatever was parsed
        pass
    return " ".join(parser.title.split()), parser.text()


class FetchUrlTool(Tool):
    name = "fetch_url"
    description = (
        "Lire le contenu textuel d'une page web (article, documentation, page de "
        "prix...). Utilise-le sur les URL trouvées avec web_search pour vérifier "
        "une information. Pour une page longue, rappelle-le avec `start` pour lire "
        "la suite."
    )
    untrusted_output = True

    class Input(BaseModel):
        url: str = Field(..., description="Adresse complète, en http:// ou https://.")
        start: int = Field(0, ge=0, description="Position de départ dans le texte.")

    def describe_call(self, args: Input) -> str:
        return f"Lecture de {urlsplit(args.url).hostname or args.url}"

    async def run(self, args: Input, ctx: ToolContext) -> ToolResult:
        settings = ctx.settings
        url = args.url.strip()
        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=settings.fetch_timeout,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "fr,en;q=0.8"},
            ) as client:
                response, body, url = await self._get(
                    client, url, settings.fetch_max_bytes, settings.fetch_allow_private
                )
        except _RefusedError as exc:
            return ToolResult.failure(str(exc))
        except httpx.TimeoutException:
            return ToolResult.failure(f"{url} did not answer in time (site too slow).")
        except httpx.HTTPError as exc:
            return ToolResult.failure(f"{url} is unreachable ({type(exc).__name__}).")

        if response.status_code >= 400:
            return ToolResult.failure(f"{url} returned HTTP {response.status_code}.")

        content_type = response.headers.get("content-type", "").lower()
        charset = response.charset_encoding or "utf-8"
        decoded = body.decode(charset, errors="replace")
        if "html" in content_type or decoded.lstrip()[:15].lower().startswith(
            ("<!doctype", "<html")
        ):
            title, text = html_to_text(decoded)
        elif (
            content_type.startswith("text/")
            or "json" in content_type
            or "xml" in content_type
        ):
            title, text = "", decoded
        else:
            return ToolResult.failure(
                f"{url} is not a text page ({content_type or 'unknown type'}); "
                "it cannot be read with fetch_url."
            )

        total = len(text)
        window = text[args.start : args.start + settings.fetch_max_chars]
        end = args.start + len(window)
        header = [f"URL : {url}"]
        if title:
            header.append(f"Titre : {title}")
        if total > settings.fetch_max_chars or args.start:
            header.append(f"Caractères {args.start}-{end} sur {total}.")
        footer = ""
        if end < total:
            footer = f"\n\n[Suite disponible : fetch_url avec start={end}]"
        if not window.strip():
            window = (
                "(page vide ou contenu généré en JavaScript : essaie le navigateur)"
            )
        return ToolResult.success(
            "\n".join(header) + "\n\n" + window + footer,
            data={
                "url": url,
                "title": title,
                "length": total,
                "start": args.start,
                "end": end,
            },
        )

    async def _get(
        self, client: httpx.AsyncClient, url: str, max_bytes: int, allow_private: bool
    ) -> tuple[httpx.Response, bytes, str]:
        for _ in range(MAX_REDIRECTS + 1):
            reason = check_url(url, allow_private=allow_private)
            if reason is None:
                reason = await check_resolved(
                    urlsplit(url).hostname or "", allow_private=allow_private
                )
            if reason:
                raise _RefusedError(f"Refused: {reason}")

            async with client.stream("GET", url) as response:
                if response.is_redirect and "location" in response.headers:
                    url = urljoin(url, response.headers["location"])
                    continue
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= max_bytes:
                        break
                return response, b"".join(chunks)[:max_bytes], url
        raise _RefusedError(f"Refused: more than {MAX_REDIRECTS} redirects.")


class _RefusedError(Exception):
    pass
