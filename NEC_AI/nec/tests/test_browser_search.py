"""Search URL building, redirect unwrapping and block detection.

Pure functions, so they are tested without a browser. The engine behaviour they
encode was measured against real headless-browser requests, not assumed, and the
comments record what was measured so a future change is not made on a guess.
"""

from __future__ import annotations

import pytest

from browser import (
    DEFAULT_SEARCH_ENGINE,
    SEARCH_ENGINES,
    build_search_url,
    looks_blocked,
    looks_empty,
    unwrap_redirect,
)


def test_bing_is_the_default() -> None:
    """The default is the one engine measured to answer an automated browser."""
    assert DEFAULT_SEARCH_ENGINE == "bing"
    assert build_search_url("meteo paris").startswith("https://www.bing.com/search")


@pytest.mark.parametrize("engine", sorted(SEARCH_ENGINES))
def test_every_engine_builds_a_url(engine: str) -> None:
    url = build_search_url("meteo paris demain", engine)
    assert url.startswith("https://")
    assert "meteo+paris+demain" in url
    # The template is the only thing that varies, so a placeholder left in it
    # would surface as a literal "{query}" in the URL.
    assert "{query}" not in url


def test_the_query_is_percent_encoded() -> None:
    """A query must never be able to inject a second URL parameter."""
    url = build_search_url("a&b=c d/e?f#g", "bing")
    assert "&b=c" not in url
    assert " " not in url
    assert url.count("?") == 1
    assert "a%26b%3Dc+d%2Fe%3Ff%23g" in url


def test_an_unknown_engine_falls_back_rather_than_failing() -> None:
    """A hallucinated engine should cost a different page, not a failed call."""
    assert build_search_url("x", "altavista") == build_search_url("x", "bing")
    assert build_search_url("x", "") == build_search_url("x", "bing")


def test_bing_tracking_links_are_unwrapped() -> None:
    """The destination is base64url in the `u` parameter."""
    href = (
        "https://www.bing.com/ck/a?!&&p=abc123&ptn=3&ver=2&hsh=4"
        "&u=a1aHR0cHM6Ly9tZXRlb2ZyYW5jZS5jb20vcGFyaXM"
    )
    assert unwrap_redirect(href) == "https://meteofrance.com/paris"


def test_a_plain_result_link_is_untouched() -> None:
    href = "https://meteofrance.com/paris"
    assert unwrap_redirect(href) == href


def test_a_bing_link_without_a_destination_is_untouched() -> None:
    """Truncated or malformed links must not become garbage."""
    assert unwrap_redirect("https://www.bing.com/ck/a?!&&p=abc") == (
        "https://www.bing.com/ck/a?!&&p=abc"
    )
    assert unwrap_redirect("https://www.bing.com/ck/a?u=a1!!!not-base64!!!") == (
        "https://www.bing.com/ck/a?u=a1!!!not-base64!!!"
    )


def test_googles_encrypted_redirect_is_left_alone() -> None:
    """Google's payload is encrypted, not encoded, so it must not be mangled."""
    href = "https://www.google.com/goto?url=CAESfQHrOzAVrpEYG5jZH0_5kFy_DVmw0m"
    assert unwrap_redirect(href) == href


# The strings below are the actual copy these engines served a headless browser.
@pytest.mark.parametrize(
    ("title", "body"),
    [
        (
            "meteo paris demain - Recherche DuckDuckGo",
            "les bots utilisent aussi DuckDuckGo",
        ),
        (
            "https://www.google.com/sorry/index",
            "Nos systèmes ont détecté un trafic exceptionnel sur votre réseau "
            "informatique. Cette page permet de vérifier que c'est bien vous qui "
            "envoyez des requêtes, et non un robot.",
        ),
        ("Captcha - Brave Search", "Verification that you're not a robot"),
        ("Un instant…", "Confirm you're not a robot"),
        ("Access Denied - Startpage", "PRODUITS Startpage Recherche"),
        ("403 - Forbidden", "automated queries"),
    ],
)
def test_every_measured_challenge_page_is_recognised(title: str, body: str) -> None:
    assert looks_blocked(title, body) is True


def test_a_real_results_page_is_not_mistaken_for_a_block() -> None:
    """False positives here would blank out working searches."""
    assert (
        looks_blocked(
            "meteo paris demain - Recherche",
            "Météo-France Demain à Paris : 19 degrés, pluie attendue. "
            "Images Vidéos Actualités Carte Confidentialité",
        )
        is False
    )


def test_a_page_mentioning_a_link_is_not_a_block() -> None:
    """`Access denied` and `Forbidden` are ordinary words on a real page."""
    assert looks_blocked("Documentation", "Forbidden fruit: a recipe") is False
    assert looks_blocked("Guide", "If access is denied, contact the owner") is False
    # The same words in the *title* are the real thing.
    assert looks_blocked("403 - Forbidden", "Nothing here") is True
    assert looks_blocked("Access Denied - Startpage", "PRODUITS") is True


# An empty result set has to be recognised without waiting out the settle
# budget, or every fruitless search costs the user the full wait.
@pytest.mark.parametrize(
    "body",
    [
        "Aucun résultat pour cette requête",
        "No results found for your query",
        "Votre recherche ne correspond à aucun document",
    ],
)
def test_an_explicit_empty_result_is_recognised(body: str) -> None:
    assert looks_empty("Recherche", body) is True


def test_a_page_with_results_is_not_empty() -> None:
    """A wrong match here would report "no results" on a page that has them."""
    assert (
        looks_empty(
            "meteo paris demain - Recherche",
            "Météo-France Demain à Paris : 19 degrés, pluie attendue.",
        )
        is False
    )
    assert looks_empty("", "") is False
