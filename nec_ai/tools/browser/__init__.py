"""Browser control over Playwright.

The implementation lives in :mod:`nec_ai.tools.browser.toolset` for now; it is
split into policy / digest / session / tools modules in a later step.
"""

from nec_ai.tools.browser.toolset import (
    BrowserConfig,
    build_browser_toolset,
    build_playwright_toolset,
    check_url,
    wrap_untrusted,
)

__all__ = [
    "BrowserConfig",
    "build_browser_toolset",
    "build_playwright_toolset",
    "check_url",
    "wrap_untrusted",
]
