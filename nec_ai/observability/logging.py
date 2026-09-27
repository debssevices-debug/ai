"""Logging setup: short, timestamped, one line per event.

[20:15:02] INFO    nec.agent      AGENT started
[20:15:03] INFO    nec.tools      TOOL web_search started
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)-14s %(message)s"
_DATEFMT = "%H:%M:%S"

#: Chatty third-party loggers kept at WARNING unless we run at DEBUG.
_NOISY = ("httpx", "httpcore", "primp", "ddgs", "google_genai", "urllib3", "asyncio")


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once. Safe to call again (it replaces handlers)."""
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        numeric = logging.INFO

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(numeric)

    for name in _NOISY:
        logging.getLogger(name).setLevel(
            logging.DEBUG if numeric <= logging.DEBUG else logging.WARNING
        )
