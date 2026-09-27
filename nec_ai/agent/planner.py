"""Planning as a tool.

Rather than a separate planning model call on every request (slow, and wasted
on "quelle heure est-il ?"), the LLM gets an ``update_plan`` tool. For a
multi-step task it writes a short plan first, then rewrites it whenever a step
fails or the approach changes. The plan lives in the task's working memory and
is streamed to the client as an ``agent.plan`` event.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nec_ai.tools.base import Tool, ToolContext, ToolResult

PLAN_TOOL = "update_plan"


class UpdatePlanTool(Tool):
    name = PLAN_TOOL
    description = (
        "Écrire ou réviser le plan d'une tâche en plusieurs étapes (recherche, "
        "comparaison, synthèse, rapport). Donne la liste complète des étapes à "
        "chaque appel. Inutile pour une question simple."
    )

    class Input(BaseModel):
        steps: list[str] = Field(
            ..., min_length=1, max_length=12, description="Étapes, dans l'ordre."
        )
        reason: str = Field(
            "", description="Pourquoi ce plan, ou pourquoi il change (optionnel)."
        )

    async def run(self, args: Input, ctx: ToolContext) -> ToolResult:
        steps = [s.strip() for s in args.steps if s.strip()]
        if not steps:
            return ToolResult.failure("The plan has no step.")
        revised = bool(ctx.task and ctx.task.plan)
        if ctx.task is not None:
            ctx.task.plan = steps
        listing = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        verb = "Plan revised" if revised else "Plan recorded"
        return ToolResult.success(
            f"{verb}:\n{listing}\nNow carry it out.",
            data={"steps": steps, "revised": revised, "reason": args.reason},
        )
