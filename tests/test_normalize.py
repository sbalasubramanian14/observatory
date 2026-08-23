import pytest
from feed.models import Item, Source, Stage
from feed.stages import normalize as normalize_module
from feed.stages.normalize import content_hash, normalize, normalize_item


def _seed(session, **kw):
    # Reuse the "s" source across calls in the same test instead of
    # re-adding it -- Source.id is a primary key, so adding a second
    # Source(id="s", ...) after the first has been committed raises an
    # IntegrityError on commit.
    if session.get(Source, "s") is None:
        session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
    defaults = dict(source_id="s", url="https://arxiv.org/abs/2607.09510",
                    url_hash="h1", title="A paper", summary="An abstract with real words.")
    defaults.update(kw)
    item = Item(**defaults)
    session.add(item)
    session.commit()
    return item


# Network-blocking safety net now lives suite-wide in tests/conftest.py
# (_block_real_network, autouse) so every test file is covered, not just
# this one. See that fixture's docstring for the rationale.


def test_content_hash_ignores_whitespace_differences():
    assert content_hash("hello   world\n") == content_hash("hello world")


def test_content_hash_differs_for_different_text():
    assert content_hash("a") != content_hash("b")


def test_content_hash_ignores_leading_trailing_whitespace_and_case():
    assert content_hash("  Hello World  ") == content_hash("hello world")
    assert content_hash("HELLO\tWORLD\n\n") == content_hash("hello world")


def test_normalize_populates_text_and_hash_from_summary(session):
    item = _seed(session)
    res = normalize(session)
    assert res.processed == 1
    session.refresh(item)
    assert item.stage is Stage.NORMALIZED
    assert item.text == "An abstract with real words."
    assert item.content_hash is not None


def test_near_duplicate_content_is_marked_failed_not_advanced(session):
    _seed(session)
    normalize(session)
    dup = _seed(session, url_hash="h2", url="https://other.example/x",
                summary="An abstract with real words.")
    res = normalize(session)
    session.refresh(dup)
    assert res.processed == 0 and res.failed == 1
    assert dup.stage is Stage.FAILED
    assert "duplicate" in dup.error.lower()


def test_item_with_no_usable_text_fails_cleanly(session):
    item = _seed(session, url_hash="h3", summary="", title="")
    res = normalize(session)
    session.refresh(item)
    assert item.stage is Stage.FAILED
    assert "no text" in item.error.lower()


def test_dedup_lookup_excludes_the_item_itself(session):
    # A freshly-seeded item has content_hash == NULL, so re-normalizing it
    # once can never self-match on content regardless of whether the
    # Item.id != item.id filter is present -- that setup can't actually
    # exercise the filter. To reach the condition the filter guards
    # against, seed an item whose content_hash is ALREADY populated in the
    # DB (as if it were previously normalized, e.g. re-processed after a
    # partial failure), then normalize it again. Without
    # `Item.id != item.id` in the lookup, the query finds the item's own
    # row (same content_hash, same id) and wrongly raises
    # DuplicateContent; with the filter, it excludes its own row and
    # proceeds normally.
    item = _seed(session)
    digest = content_hash("An abstract with real words.")
    item.content_hash = digest
    item.text = "An abstract with real words."
    session.commit()

    normalize_item(session, item)  # must NOT raise DuplicateContent
    session.commit()
    assert item.content_hash == digest


def test_arxiv_item_with_empty_summary_fails_without_reaching_fetch_seam(session, monkeypatch):
    """Risk 4: an arXiv abstract URL short-circuits to '' before the
    fallback-fetch path is even considered, so an empty-summary arXiv item
    must fail with "no text" and must NEVER call the fetch seam."""
    def _fail_if_called(url):
        raise AssertionError(f"fetch seam should not be called for arXiv item, got url={url!r}")

    monkeypatch.setattr(normalize_module, "_fetch_remote_text", _fail_if_called)

    item = _seed(session, url_hash="h4", summary="", title="",
                 url="https://arxiv.org/abs/2607.09999")
    with pytest.raises(ValueError, match="no text"):
        normalize_item(session, item)


def test_html_summary_is_stripped_to_plain_text(session):
    """I4: _extract only whitespace-collapsed item.summary; trafilatura ran
    only on the network-fallback path. Real RSS descriptions are HTML, so
    markup was stored verbatim in item.text, polluting content_hash, the
    embedding text, and entity extraction. This proves the summary path now
    goes through trafilatura's HTML-to-text extraction, not a regex.
    """
    html = ('<p>OpenAI <a href="https://openai.com/index/gpt-6">announced</a> '
            'GPT-6 today.</p>')
    item = _seed(session, url_hash="h6", url="https://example.com/gpt6",
                title="OpenAI announces GPT-6", summary=html)
    normalize_item(session, item)
    session.commit()

    assert "<p>" not in item.text
    assert "<a " not in item.text
    assert "href" not in item.text
    assert item.text == "OpenAI announced GPT-6 today."


def test_stripped_html_improves_entity_extraction(session):
    """I4: confirm the fix actually improves the downstream signal it was
    supposed to protect, not just "looks cleaner". Raw HTML smuggles
    spurious "entities" out of tag attributes -- a hostname and a CSS class
    value that never appeared in the article's visible text -- into
    extract_entities()'s cheap capitalised-token heuristic. The stripped
    text must not contain them, while still containing the genuine ones.
    """
    from feed.clustering.entities import extract_entities

    html = ('<p>OpenAI <a href="https://Example.com/GPT-6-Launch" '
            'class="TrackingLink">announced</a> GPT-6 today.</p>')
    item = _seed(session, url_hash="h7", url="https://example.com/gpt6-2",
                title="", summary=html)
    normalize_item(session, item)
    session.commit()

    raw_entities = extract_entities(html)
    clean_entities = extract_entities(item.text)

    assert "example.com" in raw_entities
    assert "trackinglink" in raw_entities
    assert "example.com" not in clean_entities
    assert "trackinglink" not in clean_entities
    assert "openai" in clean_entities
    assert "gpt-6" in clean_entities


def test_plain_text_summary_with_no_markup_still_normalizes(session):
    """Non-regression: a summary that is already plain text (no HTML at
    all) -- the common case for e.g. arXiv abstracts and many RSS feeds --
    must still normalize correctly through the new HTML-to-text path, not
    just when there happens to be markup to strip.
    """
    item = _seed(session, url_hash="h8", url="https://example.com/plain",
                title="Plain summary item",
                summary="A perfectly ordinary plain-text summary, no markup at all.")
    normalize_item(session, item)
    session.commit()
    assert item.text == "A perfectly ordinary plain-text summary, no markup at all."


# --- D0: og:image / twitter:image fallback -------------------------------
#
# Priority chain (spec D0): a feed-supplied item.image_url (set at collect
# time from media:content/thumbnail/enclosure) always wins; only when that
# is None does normalize fall back to scraping the article page's
# og:image, then twitter:image. Any fetch failure degrades to None rather
# than failing the item -- a missing lead image is cosmetic.

def test_feed_supplied_image_is_kept_and_og_image_fetch_is_never_attempted(session, monkeypatch):
    def _fail_if_called(url):
        raise AssertionError("og:image fallback must not be attempted when "
                              "the feed already supplied an image")

    monkeypatch.setattr(normalize_module, "_fetch_og_image", _fail_if_called)
    item = _seed(session, url_hash="img1", url="https://example.com/has-image",
                image_url="https://img.example.com/from-feed.jpg")
    normalize_item(session, item)
    session.commit()
    assert item.image_url == "https://img.example.com/from-feed.jpg"


def test_og_image_fallback_is_used_when_feed_supplied_no_image(session, monkeypatch):
    monkeypatch.setattr(normalize_module, "_fetch_og_image",
                        lambda url: "https://img.example.com/og.jpg")
    item = _seed(session, url_hash="img2", url="https://example.com/no-feed-image")
    assert item.image_url is None
    normalize_item(session, item)
    session.commit()
    assert item.image_url == "https://img.example.com/og.jpg"


def test_og_image_fallback_failure_degrades_to_none_not_a_raised_error(session, monkeypatch):
    def _boom(url):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(normalize_module, "_fetch_og_image", _boom)
    item = _seed(session, url_hash="img3", url="https://example.com/broken-fetch")
    normalize_item(session, item)  # must NOT raise
    session.commit()
    assert item.image_url is None
    assert item.stage != Stage.FAILED


def test_arxiv_item_never_attempts_og_image_fetch(session, monkeypatch):
    def _fail_if_called(url):
        raise AssertionError("arXiv items must never attempt the og:image fetch seam")

    monkeypatch.setattr(normalize_module, "_fetch_og_image", _fail_if_called)
    item = _seed(session, url_hash="img4", url="https://arxiv.org/abs/2607.09511")
    normalize_item(session, item)
    session.commit()
    assert item.image_url is None


def test_extract_og_image_prefers_og_image_over_twitter_image():
    html = (
        '<html><head>'
        '<meta property="og:image" content="https://img.example.com/og.jpg"/>'
        '<meta name="twitter:image" content="https://img.example.com/tw.jpg"/>'
        '</head><body></body></html>'
    )
    assert normalize_module._extract_og_image(html, "https://example.com/page") == \
        "https://img.example.com/og.jpg"


def test_extract_og_image_falls_back_to_twitter_image():
    html = (
        '<html><head>'
        '<meta name="twitter:image" content="https://img.example.com/tw.jpg"/>'
        '</head><body></body></html>'
    )
    assert normalize_module._extract_og_image(html, "https://example.com/page") == \
        "https://img.example.com/tw.jpg"


def test_extract_og_image_resolves_relative_urls_against_the_page():
    html = '<html><head><meta property="og:image" content="/assets/hero.jpg"/></head></html>'
    assert normalize_module._extract_og_image(html, "https://example.com/articles/x") == \
        "https://example.com/assets/hero.jpg"


def test_extract_og_image_skips_known_unreliable_host_and_falls_through():
    """research.facebook.com/file/... was measured live (2/2 sampled
    articles) to return HTTP 400 for its own og:image URLs regardless of
    Referer -- not hotlink protection a real browser request would pass,
    a structurally broken URL shape. Confirm the extractor skips it and
    still finds a usable twitter:image rather than returning the broken
    URL."""
    html = (
        '<html><head>'
        '<meta property="og:image" content="https://research.facebook.com/file/123/x.jpg"/>'
        '<meta name="twitter:image" content="https://img.example.com/ok.jpg"/>'
        '</head></html>'
    )
    assert normalize_module._extract_og_image(html, "https://research.facebook.com/blog/x") == \
        "https://img.example.com/ok.jpg"


def test_extract_og_image_returns_none_when_only_unreliable_host_available():
    html = '<html><head><meta property="og:image" content="https://research.facebook.com/file/9/y.jpg"/></head></html>'
    assert normalize_module._extract_og_image(html, "https://research.facebook.com/blog/y") is None


def test_feed_supplied_image_on_unreliable_host_is_discarded(session, monkeypatch):
    monkeypatch.setattr(normalize_module, "_fetch_og_image", lambda url: None)
    item = _seed(session, url_hash="img6", url="https://research.facebook.com/blog/z",
                image_url="https://research.facebook.com/file/7/z.jpg")
    normalize_item(session, item)
    session.commit()
    assert item.image_url is None


def test_extract_og_image_returns_none_when_no_meta_tags_present():
    html = "<html><head><title>No image here</title></head><body></body></html>"
    assert normalize_module._extract_og_image(html, "https://example.com/page") is None


def test_non_arxiv_empty_summary_uses_fetch_seam_not_real_network(session, monkeypatch):
    """Risk 3 + the critical network-seam risk: a non-arXiv URL with an
    empty summary DOES take the fallback-fetch path. Prove that path goes
    through the monkeypatchable module-level seam (`_fetch_remote_text`)
    rather than calling trafilatura directly, by monkeypatching only the
    seam and confirming its return value flows into item.text. The
    autouse `_block_real_network` fixture above independently guarantees
    that even if this seam were bypassed, the real trafilatura.fetch_url
    call would raise instead of hitting the network.
    """
    calls = []

    def _fake_fetch(url):
        calls.append(url)
        return "Real extracted article text goes here, well past the minimum length."

    monkeypatch.setattr(normalize_module, "_fetch_remote_text", _fake_fetch)

    item = _seed(session, url_hash="h5", summary="", title="",
                 url="https://example.com/some-article")
    normalize_item(session, item)
    session.commit()

    assert calls == ["https://example.com/some-article"]
    assert item.stage != Stage.FAILED
    assert "real extracted article text" in item.text.lower()


def test_no_summary_and_no_feed_image_uses_both_fallback_seams(session, monkeypatch):
    """The two fallbacks (full-text fetch, og:image fetch) are independent
    seams -- an item needing both must consult both, and the result of each
    must land in the right field."""
    monkeypatch.setattr(
        normalize_module, "_fetch_remote_text",
        lambda url: "Real extracted article text goes here, well past the minimum length.",
    )
    monkeypatch.setattr(
        normalize_module, "_fetch_og_image",
        lambda url: "https://img.example.com/fallback.jpg",
    )
    item = _seed(session, url_hash="img5", summary="", title="",
                url="https://example.com/needs-both")
    normalize_item(session, item)
    session.commit()
    assert "real extracted article text" in item.text.lower()
    assert item.image_url == "https://img.example.com/fallback.jpg"
