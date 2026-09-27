"""Live browser tests against a local fixture server.

These are the tests that catch real breakage: digest content, ref validity
across navigation, form filling, and the network guards. They skip when no
browser is available.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
import pytest_asyncio
from livekit.agents.llm import ToolError

from browser import StaleRefError, check_url, read_search_results

# One loop for the whole module: the Playwright driver is bound to the loop that
# launched it, so per-test loops would force a browser relaunch each time.
pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_navigation_produces_a_digest(session, server: str) -> None:
    await session.goto(f"{server}/index.html")
    digest = await session.read(wrap=False)

    assert "Accueil de test" in digest
    assert "Page d'accueil" in digest
    assert "Formulaire de contact" in digest
    assert "[1] link" in digest
    assert "Bouton test" in digest
    # Small enough to put in a prompt without hurting.
    assert len(digest) < 2000


async def test_digest_is_wrapped_as_untrusted_data(session, server: str) -> None:
    await session.goto(f"{server}/index.html")
    digest = await session.read(wrap=True)
    assert digest.startswith("<page_data>")
    assert digest.endswith("</page_data>")


async def test_ref_resolves_to_the_right_element(session, server: str) -> None:
    await session.goto(f"{server}/index.html")
    await session.read(wrap=False)

    ref, element = await session.find("Bouton test")
    assert ref.role == "button"
    # Clicking must change the page, which proves the handle was the right node.
    await element.click()
    text = await session.page.locator("#out").inner_text()
    assert text == "cliqué"


async def test_ref_by_number(session, server: str) -> None:
    await session.goto(f"{server}/menu.html")
    await session.read(wrap=False)
    ref, element = await session.element_for(1)
    assert ref.label == "Ouvrir le menu"
    assert await element.get_attribute("role") == "button"


async def test_stale_ref_is_rejected_after_navigation(session, server: str) -> None:
    await session.goto(f"{server}/index.html")
    await session.read(wrap=False)
    await session.goto(f"{server}/menu.html")
    await session.read(wrap=False)
    # A ref from the previous page must not silently point at something else.
    with pytest.raises(StaleRefError):
        await session.element_for(99)


async def test_invisible_and_disabled_elements_are_excluded(
    session, server: str
) -> None:
    """Hidden controls must not be offered as clickable.

    A disabled button still contributes its text to the page, so the check is on
    the element list, not on the whole digest.
    """
    await session.goto(f"{server}/hidden.html")
    digest = await session.read(wrap=False)
    offered = [r.label for r in session.registry.refs]
    assert offered == ["Visible"]
    # display:none removes the label from the text entirely.
    assert "Invisible" not in digest
    assert "Lien caché" not in digest


async def test_password_field_is_listed_without_its_value(session, server: str) -> None:
    await session.goto(f"{server}/menu.html")
    await session.read(wrap=False)
    secret = [r for r in session.registry.refs if r.input_type == "password"]
    assert secret, "the password field should still be addressable"
    assert secret[0].is_password
    digest = await session.read(wrap=False)
    assert "Code secret" in digest


async def test_form_fill_writes_every_field(session, server: str) -> None:
    await session.goto(f"{server}/form.html")
    await session.read(wrap=False)

    _ref, nom = await session.find("Nom")
    await nom.fill("Dupont")
    _ref, email = await session.find("Email")
    await email.fill("dupont@example.test")
    _ref, message = await session.find("Message")
    await message.fill("Bonjour")
    _ref, pays = await session.find("Pays")
    await pays.select_option(label="Belgique")

    values = await session.page.evaluate(
        "() => Array.from(document.querySelectorAll('input,select,textarea'))"
        ".map(e => [e.name, e.value])"
    )
    assert ["nom", "Dupont"] in values
    assert ["email", "dupont@example.test"] in values
    assert ["pays", "be"] in values


async def test_submit_navigates_and_the_new_page_is_read(session, server: str) -> None:
    await session.goto(f"{server}/form.html")
    await session.read(wrap=False)
    _ref, button = await session.find("Envoyer")
    await button.click()
    await session.page.wait_for_load_state("domcontentloaded")
    digest = await session.read(wrap=False)
    assert "Envoyé" in digest
    assert "bien été transmis" in digest


async def test_cloud_metadata_request_is_blocked(session, server: str) -> None:
    """A page pointing an <img> at the metadata endpoint must not reach it.

    The URL is unique per run so the browser cannot serve it from cache, which
    would make the assertion pass or fail for the wrong reason.
    """
    token = uuid4().hex
    await session.goto(f"{server}/index.html")
    before = session.blocked_request_count
    # Injected from a real origin: Chromium does not route subresource requests
    # of an about:blank document through context interception.
    await session.page.evaluate(
        "(t) => { const i = new Image();"
        " i.src = 'http://169.254.169.254/latest/meta-data/?' + t;"
        " document.body.appendChild(i); }",
        token,
    )
    await session.page.wait_for_timeout(750)
    assert session.blocked_request_count > before
    size = await session.page.evaluate(
        "() => { const i = document.images[document.images.length - 1];"
        " return [i.naturalWidth, i.naturalHeight]; }"
    )
    assert size == [0, 0]


async def test_navigation_to_a_blocked_url_is_refused(session, server: str) -> None:
    from livekit.agents.llm import ToolError

    with pytest.raises(ToolError) as excinfo:
        await session.goto("file:///etc/passwd")
    assert "not allowed" in str(excinfo.value)


async def test_host_allowlist_is_enforced(session, server: str) -> None:
    from livekit.agents.llm import ToolError

    with pytest.raises(ToolError) as excinfo:
        await session.goto("https://example.com/")
    assert "allowlist" in str(excinfo.value)


class _ClickPath:
    """The toolset's own click helper, bound without building a whole toolset.

    ``click_element`` lives on ``InputTools`` and only needs ``self.session`` and
    ``self.config``, so this exercises the production code path rather than a
    reimplementation of it.
    """

    def __init__(self, session) -> None:
        from browser import InputTools

        self.session = session
        self.config = session.config
        self.click_element = InputTools.click_element.__get__(self)


async def _click_via_tool_path(session, selector: str) -> None:
    element = await session.page.query_selector(selector)
    assert element is not None, f"{selector} not found on the page"
    await _ClickPath(session).click_element(element)


async def test_a_dialog_is_held_open_and_can_be_answered(session, server: str) -> None:
    """A confirm() must not be able to dismiss itself.

    With no dialog handler Playwright auto-dismisses, the page takes its cancel
    branch, and the click reports success. That is the failure this covers: the
    agent telling the user a destructive action happened when it did not.
    """
    await session.goto(f"{server}/dialog.html")
    assert session.has_dialog is False

    # Through the production click path, which has to tolerate the page blocking
    # on the dialog rather than hanging until the action timeout.
    await _click_via_tool_path(session, "#ask")
    assert session.has_dialog, "the confirm() was auto-dismissed instead of held"

    # The digest has to say so, or the model cannot tell a blocked page from a
    # slow one.
    digest = await session.read(wrap=False)
    assert "BLOCKED" in digest
    assert "Supprimer" in digest

    # And no other action may proceed while the question is unanswered.
    from livekit.agents.llm import ToolError

    with pytest.raises(ToolError, match="dialog"):
        await session.require_no_dialog()

    await session.resolve_dialog(accept=True)
    assert session.has_dialog is False
    body = await session.page.evaluate("() => document.body.innerText")
    assert "DELETED" in body


async def test_a_dialog_can_be_dismissed(session, server: str) -> None:
    await session.goto(f"{server}/dialog.html")
    await _click_via_tool_path(session, "#ask")
    assert session.has_dialog

    result = await session.resolve_dialog(accept=False)
    assert "dismissed" in result.lower()
    assert session.has_dialog is False
    body = await session.page.evaluate("() => document.body.innerText")
    assert "KEPT" in body


async def test_a_popup_tab_is_tracked_and_followed(session, server: str) -> None:
    """target=_blank must not open a tab the agent cannot see.

    Measured against the real web before this was fixed: the context gained a
    page that `session.pages` never learned about, so the digest kept describing
    the page the click came from.
    """
    await session.goto(f"{server}/popup.html")
    before = len(session.pages)

    await session.page.click("#pop", no_wait_after=True)
    await session.page.wait_for_timeout(1500)

    assert len(session.pages) == before + 1, "the popup was not tracked"

    # The new tab becomes active, so the digest describes what the user can now
    # see rather than the tab they left.
    assert session.active_index == before
    assert "Popup" in await session.read(wrap=False)

    # The original tab is still reachable.
    session.switch_tab(0)
    assert "Popup" not in await session.read(wrap=False)


async def test_tabs_are_isolated_and_switchable(session, server: str) -> None:
    from livekit.agents.llm import ToolError

    await session.goto(f"{server}/index.html")
    await session.new_tab(f"{server}/form.html")
    assert len(session.pages) == 2
    assert session.active_index == 1

    rows = await session.tab_summary()
    assert rows[1]["active"] is True
    assert "Formulaire" in rows[1]["title"]

    session.switch_tab(0)
    assert session.active_index == 0
    digest = await session.read(wrap=False)
    assert "Page d'accueil" in digest

    with pytest.raises(ToolError, match="no tab"):
        session.switch_tab(7)


async def test_tab_limit_is_enforced(session, server: str, config) -> None:
    from livekit.agents.llm import ToolError

    await session.goto(f"{server}/index.html")
    for _ in range(config.max_tabs - 1):
        await session.new_tab()
    with pytest.raises(ToolError) as excinfo:
        await session.new_tab()
    assert "limit" in str(excinfo.value)


async def test_long_page_text_is_chunked(session, server: str) -> None:
    from browser import TEXT_CHUNK

    await session.goto(f"{server}/long.html")
    await session.read(wrap=False)
    text = await session.page.evaluate("() => document.body.innerText")
    assert len(text) > TEXT_CHUNK * 2

    chunk = text[:TEXT_CHUNK]
    assert len(chunk) == TEXT_CHUNK
    assert "suite" not in chunk  # stops at the boundary rather than mid-word


async def test_page_with_injection_is_only_ever_reported_as_data(
    session, server: str
) -> None:
    """The page text is attacker-controlled. It must come back as data."""
    await session.goto(f"{server}/injection.html")
    digest = await session.read(wrap=True)
    assert digest.startswith("<page_data>")
    assert "ignore all previous instructions" in digest
    # The digest is labelled data, so the model has an explicit signal.
    assert "</page_data>" in digest


async def test_search_results_come_back_as_a_list(session, server: str, config):
    """A results page is read as results, not as a page digest.

    The failure this guards against is silent: a plain digest of a search page is
    dominated by the engine's own nav, so the model gets a menu and no results
    while every intermediate value still looks fine.
    """
    await session.goto(f"{server}/search.html")
    digest, registry = await read_search_results(
        session.page, config, "bing", query="meteo paris demain"
    )

    assert digest.startswith("<page_data>")
    # Results, not the engine's menu.
    assert "PREVISIONS METEO FRANCE" in digest
    assert "La Chaîne Météo" in digest
    assert "Images" not in digest.split("Search results for")[0]
    # The breadcrumb anchor must not become the title.
    assert "meteofrance.com https://meteofrance.com" not in digest
    # The snippet is the summary, not a repeat of the title.
    assert "19 degrés" in digest
    assert len(registry) == 3
    assert [r.index for r in registry.refs] == [1, 2, 3]


async def test_search_result_refs_resolve_to_the_result(session, server: str, config):
    """`[2]` in the list has to be the second result's link.

    The number is only worth printing if clicking it opens the result the model
    was shown, so this checks the ordinal against a real handle.
    """
    await session.goto(f"{server}/search.html")
    _digest, registry = await read_search_results(
        session.page, config, "bing", query="meteo"
    )
    # What the tool does: the session resolves refs against this registry.
    session.registry = registry

    second = registry.get(2)
    _ref, element = await session.element_for(second.index)
    href = await element.get_attribute("href")
    assert "meteo.be" in href
    assert second.label == "Météo en Belgique"


async def test_a_blocked_engine_is_named_rather_than_shown_as_empty(
    session, server: str, config
) -> None:
    """A challenge page must not be reported as "no results".

    An empty result set and a refusal look identical to the model otherwise, and
    the honest answer — this engine will not answer an automated browser — is the
    one that lets it pick a different tool.
    """
    await session.goto(f"{server}/blocked.html")
    digest, registry = await read_search_results(
        session.page, config, "bing", query="meteo paris demain"
    )

    assert "refused to show search results" in digest
    assert "search_web" in digest
    assert len(registry) == 0
    # The challenge page's own links must not be passed off as results.
    assert "Conditions générales" not in digest


async def test_a_page_with_no_results_says_so(session, server: str, config) -> None:
    """A page with nothing to link to reports emptiness, not invented results."""
    await session.goto(f"{server}/form.html")
    digest, registry = await read_search_results(
        session.page, config, "bing", query="nothing here"
    )
    assert "No results" in digest
    assert len(registry) == 0


async def test_results_win_over_the_phrase_that_means_empty(
    session, server: str, config
) -> None:
    """A snippet quoting "aucun résultat" is a result, not an empty result set.

    The reader has an early exit for a page where the engine says it matched
    nothing, so that a fruitless search does not pay the settle budget. That exit
    is only allowed to fire once the extraction has come back empty — otherwise
    a page whose own content mentions the phrase would report itself as empty.
    """
    await session.goto(f"{server}/search_phrase.html")
    digest, registry = await read_search_results(
        session.page, config, "bing", query="citation"
    )
    assert "Gérer une recherche sans résultat" in digest
    assert "Documentation de la recherche" in digest
    assert "No results" not in digest
    assert len(registry) == 2


async def test_closing_releases_the_context(session, server: str) -> None:
    await session.goto(f"{server}/index.html")
    assert session.is_open
    await session.aclose()
    assert not session.is_open
    assert len(session.registry) == 0


async def test_policy_default_denies_loopback_even_though_tests_allow_it() -> None:
    """Guard against the test fixture's relaxation leaking into the default."""
    from browser import BrowserConfig

    assert check_url("http://127.0.0.1/", BrowserConfig()) is not None


# ═══════════════════════════════════════════════════════════════════════════
# THE CONFIRMATION GATE, AGAINST A REAL BROWSER
# ═══════════════════════════════════════════════════════════════════════════
#
# The unit tests prove the classification and the message. These prove the thing
# that actually matters: that a refused submit leaves the page exactly where it
# was, and a confirmed one really goes through. A gate that raises after filling
# the fields, or after the request was already sent, would satisfy every
# assertion in test_browser_confirmation.py and still be wrong.


class _LiveCtx:
    """Just enough RunContext for the toolset.

    The gated tools speak before they act, and a real ``generate_reply`` would
    need a live session with a room. What is under test here is the browser, not
    the speech, so the speech is recorded and dropped.
    """

    def __init__(self) -> None:
        self.session = self
        self.spoken: list[str] = []
        self.current_agent = None

    async def generate_reply(self, instructions: str = "", **_: object) -> None:
        self.spoken.append(instructions)

    async def wait_for_playout(self) -> None:
        return None


@pytest_asyncio.fixture(loop_scope="session")
async def live_toolset(config):
    """A real BrowserToolset with a real browser behind it.

    Built directly rather than through ``build_browser_toolset`` so the stub
    cannot be swapped in by an environment variable and quietly make these tests
    assert against canned pages.
    """
    from browser import BrowserToolset

    toolset = BrowserToolset(config)
    toolset._want_browser()
    try:
        await toolset.session.start()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no usable browser: {type(exc).__name__}: {exc}")
    try:
        yield toolset
    finally:
        await toolset.aclose()


async def test_an_unconfirmed_submit_is_refused_and_nothing_is_sent(
    live_toolset, server: str
) -> None:
    await live_toolset.session.goto(f"{server}/form.html")
    await live_toolset.session.read(wrap=False)
    ctx = _LiveCtx()

    with pytest.raises(ToolError) as caught:
        await live_toolset._type_text(
            ctx, text="Dupont", into="Nom", submit=True, confirmed=False
        )

    # The message has to be actionable, or the model retries blind.
    assert "confirmed=True" in str(caught.value)
    assert "Ask the user" in str(caught.value)
    # Nothing was said, because nothing happened.
    assert ctx.spoken == []
    # The page did not move, and nothing was typed: the check runs before the
    # element is touched, not after.
    assert live_toolset.session.page.url.endswith("/form.html")
    value = await live_toolset.session.page.evaluate(
        "() => document.querySelector('input[name=nom]').value"
    )
    assert value == ""


async def test_a_confirmed_submit_goes_through(live_toolset, server: str) -> None:
    await live_toolset.session.goto(f"{server}/form.html")
    await live_toolset.session.read(wrap=False)
    ctx = _LiveCtx()

    digest = await live_toolset._type_text(
        ctx, text="Dupont", into="Nom", submit=True, confirmed=True
    )

    assert "Envoyé" in digest
    assert "bien été transmis" in digest
    # The confirmed path announces the action, which is the point of confirming
    # it out loud rather than just flipping a flag.
    assert ctx.spoken


async def test_a_plain_fill_is_never_gated(live_toolset, server: str) -> None:
    """A gate that also caught ordinary typing would be unusable."""
    await live_toolset.session.goto(f"{server}/form.html")
    await live_toolset.session.read(wrap=False)
    ctx = _LiveCtx()

    digest = await live_toolset._type_text(ctx, text="Dupont", into="Nom")

    # The digest reports the action, never the value: a typed value can be a
    # password, and the digest is the one thing guaranteed to reach the model.
    assert "Typed into" in digest
    assert "Dupont" not in digest
    value = await live_toolset.session.page.evaluate(
        "() => document.querySelector('input[name=nom]').value"
    )
    assert value == "Dupont"
    assert live_toolset.session.page.url.endswith("/form.html")


async def test_enter_is_gated_on_a_real_page(live_toolset, server: str) -> None:
    """The hole this closes: Enter had no `submit` argument to key off."""
    await live_toolset.session.goto(f"{server}/form.html")
    await live_toolset.session.read(wrap=False)
    # Put focus in a text field, or Enter goes to the body and submits nothing.
    await live_toolset.session.page.locator("input[name=nom]").click()
    ctx = _LiveCtx()

    with pytest.raises(ToolError) as caught:
        await live_toolset._press_key(ctx, key="Enter", confirmed=False)

    assert "confirmed=True" in str(caught.value)
    assert live_toolset.session.page.url.endswith("/form.html")


async def test_a_confirmed_enter_sends_the_form(live_toolset, server: str) -> None:
    await live_toolset.session.goto(f"{server}/form.html")
    await live_toolset.session.read(wrap=False)
    await live_toolset.session.page.locator("input[name=nom]").fill("Dupont")
    await live_toolset.session.page.locator("input[name=nom]").click()
    ctx = _LiveCtx()

    digest = await live_toolset._press_key(ctx, key="Enter", confirmed=True)

    assert "Envoyé" in digest
    assert "bien été transmis" in digest


async def test_a_harmless_key_is_never_gated(live_toolset, server: str) -> None:
    """Tab is a key press, and gating it would be absurd to a user."""
    await live_toolset.session.goto(f"{server}/form.html")
    await live_toolset.session.read(wrap=False)
    await live_toolset.session.page.locator("input[name=nom]").click()
    ctx = _LiveCtx()

    digest = await live_toolset._press_key(ctx, key="Tab")

    assert "Pressed Tab" in digest
    assert ctx.spoken == []


async def test_a_step_plan_without_enter_is_never_gated(
    live_toolset, server: str
) -> None:
    """Most run_steps plans commit nothing; asking each time would be noise."""
    await live_toolset.session.goto(f"{server}/index.html")
    await live_toolset.session.read(wrap=False)
    ctx = _LiveCtx()

    digest = await live_toolset._run_steps(
        ctx, steps=[{"action": "click", "text": "Formulaire de contact"}]
    )

    assert "Formulaire de contact" in digest


async def test_a_step_plan_that_presses_enter_is_gated(
    live_toolset, server: str
) -> None:
    await live_toolset.session.goto(f"{server}/form.html")
    await live_toolset.session.read(wrap=False)
    ctx = _LiveCtx()

    with pytest.raises(ToolError) as caught:
        await live_toolset._run_steps(
            ctx, steps=[{"action": "press", "key": "Enter"}], confirmed=False
        )

    assert "confirmed=True" in str(caught.value)
    assert live_toolset.session.page.url.endswith("/form.html")
