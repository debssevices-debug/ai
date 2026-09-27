"""Marking of untrusted text.

Anything that comes from the open web (search snippets, page text) or from a
file the agent did not write is *data*, never instructions. It is wrapped in
``<page_data>`` tags before reaching the LLM, and the system prompt tells the
model that nothing inside those tags can give it orders.
"""

UNTRUSTED_PREFIX = "<page_data>"
UNTRUSTED_SUFFIX = "</page_data>"


def wrap_untrusted(text: str) -> str:
    """Mark text that came from outside as data, not instructions."""
    return f"{UNTRUSTED_PREFIX}\n{text}\n{UNTRUSTED_SUFFIX}"
