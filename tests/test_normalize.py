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


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch):
    """Safety net: fail loudly if anything in this file reaches the real
    trafilatura.fetch_url, regardless of which code path got there.

    This is in addition to (not instead of) the module-level seam
    (`_fetch_remote_text`) that production code and targeted tests use to
    avoid the network path entirely.
    """
    def _boom(*args, **kwargs):
        raise AssertionError("real network fetch (trafilatura.fetch_url) attempted in a test")

    monkeypatch.setattr("trafilatura.fetch_url", _boom)


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
    # A freshly-normalized item must not be treated as a duplicate of
    # itself when Item.content_hash == Item.content_hash trivially matches
    # its own row -- the lookup must exclude Item.id == item.id.
    item = _seed(session)
    normalize_item(session, item)
    session.commit()
    assert item.stage != Stage.FAILED
    assert item.content_hash is not None


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
