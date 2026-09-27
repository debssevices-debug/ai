"""Digest shaping and ref bookkeeping. No browser required."""

from __future__ import annotations

import pytest

from nec_ai.tools.browser.toolset import (
    BrowserConfig,
    ElementRef,
    RefRegistry,
    StaleRefError,
    build_registry,
    format_digest,
    role_for,
)

CONFIG = BrowserConfig()

SNAPSHOT = {
    "url": "https://example.com/form",
    "title": "Formulaire",
    "headings": ["Formulaire de contact"],
    "text": "Nom Email Message  " * 40,
    "frameCount": 0,
    "elementCount": 3,
    "elements": [
        {
            "ordinal": 0,
            "tag": "a",
            "inputType": "",
            "role": "",
            "label": "Conditions générales",
            "href": "/cg",
            "value": "",
        },
        {
            "ordinal": 1,
            "tag": "input",
            "inputType": "email",
            "role": "",
            "label": "Adresse e-mail",
            "href": "",
            "value": "",
        },
        {
            "ordinal": 2,
            "tag": "button",
            "inputType": "submit",
            "role": "",
            "label": "Se connecter",
            "href": "",
            "value": "",
        },
    ],
}


def test_registry_indexes_from_one() -> None:
    registry = build_registry(SNAPSHOT["elements"], SNAPSHOT["url"], CONFIG)
    assert [r.index for r in registry.refs] == [1, 2, 3]
    assert registry.get(1).label == "Conditions générales"


def test_digest_contains_the_essentials() -> None:
    registry = build_registry(SNAPSHOT["elements"], SNAPSHOT["url"], CONFIG)
    digest = format_digest(SNAPSHOT, registry, CONFIG)
    assert "URL: https://example.com/form" in digest
    assert "Title: Formulaire" in digest
    assert "Headings: Formulaire de contact" in digest
    assert "[1] link 'Conditions générales' -> /cg" in digest
    assert "[3] button 'Se connecter'" in digest


def test_digest_respects_the_character_cap() -> None:
    tight = BrowserConfig(max_digest_chars=200, max_text_chars=50)
    registry = build_registry(SNAPSHOT["elements"], SNAPSHOT["url"], tight)
    digest = format_digest(SNAPSHOT, registry, tight)
    assert len(digest) <= 200
    assert "[...]" in digest


def test_digest_respects_the_element_cap_and_says_so() -> None:
    tight = BrowserConfig(max_elements=2)
    registry = build_registry(SNAPSHOT["elements"], SNAPSHOT["url"], tight)
    digest = format_digest(SNAPSHOT, registry, tight)
    assert "[1]" in digest
    assert "[2]" in digest
    assert "[3]" not in digest
    assert "1 more element(s) not shown" in digest


def test_digest_reports_embedded_frames() -> None:
    data = {**SNAPSHOT, "frameCount": 2}
    registry = build_registry(SNAPSHOT["elements"], SNAPSHOT["url"], CONFIG)
    digest = format_digest(data, registry, CONFIG)
    assert "2 embedded frame(s)" in digest


def test_password_value_never_reaches_a_ref() -> None:
    items = [
        {
            "ordinal": 0,
            "tag": "input",
            "inputType": "password",
            "role": "",
            "label": "Mot de passe",
            "href": "",
            "value": "",
        }
    ]
    registry = build_registry(items, "https://x.test", CONFIG)
    ref = registry.get(1)
    assert ref.is_password
    assert "hunter2" not in ref.describe()


def test_stale_ref_is_actionable() -> None:
    registry = RefRegistry()
    registry.replace([ElementRef(1, 0, "button", "button", "OK")], "https://x.test")
    with pytest.raises(StaleRefError) as excinfo:
        registry.get(9)
    message = str(excinfo.value)
    assert "valid numbers are 1 to 1" in message
    assert "read the page again" in message


def test_ref_registry_is_replaced_on_navigation() -> None:
    registry = RefRegistry()
    first = registry.generation
    registry.replace([ElementRef(1, 0, "button", "button", "OK")], "https://a.test")
    registry.replace([ElementRef(1, 0, "a", "link", "Next")], "https://b.test")
    assert registry.generation > first
    assert registry.get(1).label == "Next"


def test_empty_registry_explains_itself() -> None:
    with pytest.raises(StaleRefError) as excinfo:
        RefRegistry().get(1)
    assert "Read the page" in str(excinfo.value)


def test_label_matching_is_accent_and_case_insensitive() -> None:
    registry = RefRegistry()
    registry.replace(
        [
            ElementRef(1, 0, "a", "link", "Généralités"),
            ElementRef(2, 1, "button", "button", "Se connecter"),
        ],
        "https://x.test",
    )
    assert registry.match("generalites").index == 1
    assert registry.match("se connecter").index == 2
    assert registry.match("Se Connecter") is not None


def test_ambiguous_label_does_not_silently_pick() -> None:
    registry = RefRegistry()
    registry.replace(
        [
            ElementRef(1, 0, "button", "button", "Supprimer le compte"),
            ElementRef(2, 1, "button", "button", "Supprimer le projet"),
        ],
        "https://x.test",
    )
    assert registry.match("Supprimer le") is None
    assert len(registry.find_by_label("Supprimer")) == 2


@pytest.mark.parametrize(
    "tag,input_type,expected",
    [
        ("a", "", "link"),
        ("button", "", "button"),
        ("input", "checkbox", "checkbox"),
        ("input", "radio", "radio"),
        ("input", "password", "textbox"),
        ("input", "email", "textbox"),
        ("input", "submit", "button"),
        ("select", "", "dropdown"),
        ("textarea", "", "textbox"),
        ("div", "", "div"),
    ],
)
def test_role_mapping(tag: str, input_type: str, expected: str) -> None:
    assert role_for(tag, input_type, "") == expected


def test_explicit_role_wins() -> None:
    assert role_for("div", "", "button") == "button"
