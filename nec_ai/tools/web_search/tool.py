"""The ``web_search`` tool the LLM sees. The engine behind it is a provider."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nec_ai.tools.base import Tool, ToolContext, ToolResult
from nec_ai.tools.web_search.base import SearchError, SearchProvider, SearchResult


def format_results(query: str, provider: str, results: list[SearchResult]) -> str:
    lines = [f'Résultats de recherche pour "{query}" ({provider}) :', ""]
    for position, row in enumerate(results, start=1):
        lines.append(f"{position}. {row.title}")
        lines.append(f"   {row.url}")
        if row.published:
            lines.append(f"   date : {row.published}")
        if row.snippet:
            lines.append(f"   {row.snippet[:300]}")
        lines.append("")
    lines.append(
        "Les extraits sont courts : ouvre les pages importantes avec fetch_url "
        "avant d'affirmer un prix, un chiffre ou une fonctionnalité."
    )
    return "\n".join(lines)


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Rechercher sur Internet. Renvoie une liste de résultats (titre, URL, "
        "extrait). À utiliser pour toute information récente ou vérifiable : "
        "actualités, prix, entreprises, produits, documentation, comparatifs. "
        "Lance plusieurs recherches ciblées plutôt qu'une seule très large."
    )
    untrusted_output = True

    class Input(BaseModel):
        query: str = Field(
            ..., min_length=1, max_length=400, description="La requête de recherche."
        )
        max_results: int | None = Field(
            None, ge=1, le=20, description="Nombre de résultats (défaut : 6)."
        )

    def __init__(self, provider: SearchProvider) -> None:
        self.provider = provider

    def describe_call(self, args: Input) -> str:
        return f'Recherche web : "{args.query}"'

    async def run(self, args: Input, ctx: ToolContext) -> ToolResult:
        query = args.query.strip()
        limit = args.max_results or ctx.settings.search_max_results
        try:
            results = await self.provider.search(
                query, max_results=limit, region=ctx.settings.search_region
            )
        except SearchError as exc:
            return ToolResult.failure(
                f"La recherche a échoué ({exc}). Réessaie avec une autre formulation "
                "ou signale que la recherche est indisponible."
            )

        if not results:
            return ToolResult.success(
                f'Aucun résultat pour "{query}". Reformule la requête (termes plus '
                "simples, autre langue) avant de conclure que l'information n'existe pas.",
                data={"query": query, "results": []},
            )
        return ToolResult.success(
            format_results(query, self.provider.name, results),
            data={
                "query": query,
                "provider": self.provider.name,
                "results": [r.to_dict() for r in results],
            },
        )
