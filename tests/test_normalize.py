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
