"""Report the prompt cost of the tool surface and the instructions.

The main agent carries every browser tool, so both numbers land in every LLM
request. Worth knowing rather than guessing at.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from browser import build_browser_toolset

CHARS_PER_TOKEN = 3.6


def tool_schema(tool) -> dict:
    info = tool.info
    parameters = (
        info.parameters.model_dump(exclude_none=True)
        if hasattr(info, "parameters")
        else {}
    )
    return {
        "name": info.name,
        "description": info.description,
        "parameters": parameters,
    }


def report(label: str, text: str) -> None:
    print(
        f"{label:>28}: {len(text):>7} chars  ~{int(len(text) / CHARS_PER_TOKEN):>5} tokens"
    )


def main() -> None:
    toolset = build_browser_toolset()
    schemas = [tool_schema(t) for t in toolset.tools]

    print(f"{'tool count':>28}: {len(schemas):>7}")
    report("descriptions", "".join(s["description"] or "" for s in schemas))
    report("full tool schemas", json.dumps(schemas, ensure_ascii=False))

    from agent import Assistant, build_session_tools

    # One session's worth of tools. Built per session now, so the constant is
    # gone; this reports what a single caller actually sends.
    session_tools = build_session_tools()

    agent = Assistant()
    report("instructions", agent.instructions or "")

    search = [t for t in session_tools if getattr(t, "info", None)]
    report(
        "search_web schema",
        json.dumps([tool_schema(t) for t in search], ensure_ascii=False),
    )

    print()
    print("per-tool schema size, largest first:")
    ranked = sorted(
        ((len(json.dumps(s, ensure_ascii=False)), s["name"]) for s in schemas),
        reverse=True,
    )
    for size, name in ranked[:6]:
        print(f"  {name:<18} {size:>5} chars  ~{int(size / CHARS_PER_TOKEN):>4} tokens")


if __name__ == "__main__":
    main()
