"""Unit tests for feed.imaging: the concurrent, rate-limited og:image
fallback shared by the normalize stage's post-step and `feed
backfill-images` (Phase D-images).

No test here ever lets a real HTTP request out -- the suite-wide autouse
`_block_real_imaging_network` fixture in tests/conftest.py makes
feed.imaging._get raise if reached without being monkeypatched first.
"""
from __future__ import annotations
import threading
import time

import pytest

from feed import imaging
from feed.imaging import (
    HostThrottle,
    ImageFetchResult,
    fetch_og_image,
    needs_image_fetch,
    resolve_images,
)
from feed.models import Item, Source


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "", url: str = ""):
        self.status_code = status_code
        self.text = text
        self.url = url


def _og_html(url: str) -> str:
    return f'<html><head><meta property="og:image" content="{url}"/></head></html>'


# --- fetch_og_image: status classification --------------------------------

def test_fetch_og_image_returns_ok_with_url_on_a_clean_200(monkeypatch):
    monkeypatch.setattr(imaging, "_get",
                        lambda url, *, timeout: _FakeResponse(
                            200, _og_html("https://img.example.com/x.jpg"), url))
    r = fetch_og_image("https://example.com/a")
    assert r == ImageFetchResult("https://img.example.com/x.jpg", "ok")


def test_fetch_og_image_no_og_tag_on_200_without_meta(monkeypatch):
    monkeypatch.setattr(imaging, "_get",
                        lambda url, *, timeout: _FakeResponse(200, "<html></html>", url))
    r = fetch_og_image("https://example.com/a")
    assert r.image_url is None
    assert r.status == "no_og_tag"


@pytest.mark.parametrize("code", [403, 202, 429])
def test_fetch_og_image_blocked_statuses_are_not_transient(monkeypatch, code):
    monkeypatch.setattr(imaging, "_get", lambda url, *, timeout: _FakeResponse(code, "", url))
    r = fetch_og_image("https://example.com/a")
    assert r.image_url is None
    assert r.status == f"blocked_{code}"
    assert r.is_transient is False


def test_fetch_og_image_other_http_error_is_not_transient(monkeypatch):
    monkeypatch.setattr(imaging, "_get", lambda url, *, timeout: _FakeResponse(500, "", url))
    r = fetch_og_image("https://example.com/a")
    assert r.image_url is None
    assert r.status == "http_500"
    assert r.is_transient is False


def test_fetch_og_image_network_exception_is_transient(monkeypatch):
    def _boom(url, *, timeout):
        raise TimeoutError("connect timed out")

    monkeypatch.setattr(imaging, "_get", _boom)
    r = fetch_og_image("https://example.com/a")
    assert r.image_url is None
    assert r.status == "network_error:TimeoutError"
    assert r.is_transient is True


def test_fetch_og_image_never_returns_a_denylisted_url(monkeypatch):
    """The denylist is enforced by extract_meta_image (see
    tests/test_normalize.py's test_extract_og_image_* tests for the
    fall-through-to-twitter:image behaviour); this proves fetch_og_image
    as a whole never surfaces a denylisted URL as "ok" even when it is the
    only candidate on the page."""
    monkeypatch.setattr(imaging, "_get",
                        lambda url, *, timeout: _FakeResponse(
                            200, _og_html("https://research.facebook.com/file/1/x.jpg"), url))
    r = fetch_og_image("https://research.facebook.com/blog/a")
    assert r.image_url is None
    assert r.status == "no_og_tag"


def test_fetch_og_image_never_raises_on_parse_failure(monkeypatch):
    """extract_meta_image is a well-tested pure function elsewhere; this
    just proves fetch_og_image's own try/except around it actually works,
    by handing it something BeautifulSoup/lxml cannot parse as expected."""
    monkeypatch.setattr(imaging, "_get", lambda url, *, timeout: _FakeResponse(200, None, url))
    r = fetch_og_image("https://example.com/a")
    assert r.image_url is None
    assert r.status.startswith("parse_error")


# --- needs_image_fetch: the cache/skip rule -------------------------------

def _item(**kw) -> Item:
    defaults = dict(source_id="s", url="https://example.com/x", url_hash="h",
                    title="T")
    defaults.update(kw)
    return Item(**defaults)


def test_needs_image_fetch_true_for_a_fresh_item_with_no_image():
    assert needs_image_fetch(_item()) is True


def test_needs_image_fetch_false_when_image_already_present():
    assert needs_image_fetch(_item(image_url="https://img.example.com/x.jpg")) is False


def test_needs_image_fetch_false_when_already_checked():
    import datetime
    assert needs_image_fetch(
        _item(image_checked_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc))
    ) is False


def test_needs_image_fetch_false_for_arxiv_abstract_url():
    assert needs_image_fetch(_item(url="https://arxiv.org/abs/2607.00001")) is False


# --- HostThrottle: per-host politeness delay -------------------------------
#
# These three tests specifically prove the real timing behaviour, so they
# restore the genuine time.sleep seam that the suite-wide autouse
# `_no_real_imaging_sleep` fixture (tests/conftest.py) otherwise no-ops for
# every other test in this file/suite.

def test_host_throttle_spaces_out_requests_to_the_same_host(monkeypatch):
    monkeypatch.setattr(imaging, "_sleep", time.sleep)
    throttle = HostThrottle(delay=0.05)
    start = time.monotonic()
    throttle.wait("a.example")
    throttle.wait("a.example")
    elapsed = time.monotonic() - start
    assert elapsed >= 0.045  # small slack for scheduler jitter


def test_host_throttle_does_not_delay_different_hosts(monkeypatch):
    monkeypatch.setattr(imaging, "_sleep", time.sleep)
    throttle = HostThrottle(delay=1.0)
    start = time.monotonic()
    throttle.wait("a.example")
    throttle.wait("b.example")
    elapsed = time.monotonic() - start
    assert elapsed < 0.5  # would be >= 1.0s if hosts wrongly shared a slot


def test_host_throttle_zero_delay_never_sleeps(monkeypatch):
    monkeypatch.setattr(imaging, "_sleep", time.sleep)
    throttle = HostThrottle(delay=0.0)
    start = time.monotonic()
    for _ in range(5):
        throttle.wait("a.example")
    assert time.monotonic() - start < 0.1


def test_host_throttle_goes_through_the_sleep_seam_not_a_bare_time_sleep(monkeypatch):
    """Regression guard for the exact bug this seam was introduced to
    fix: if HostThrottle.wait() ever called time.sleep() directly again
    instead of the module-level _sleep() seam, every test suite run
    seeding same-host items (a common fixture pattern -- see
    tests/test_cli.py::test_run_drains_more_than_one_batch_per_stage's 130
    same-host items) would silently start taking real wall-clock minutes
    again, exactly as it did before this fix. Proves the seam is actually
    on the call path by monkeypatching only _sleep and confirming it is
    invoked with a plausible, positive duration.
    """
    calls = []
    monkeypatch.setattr(imaging, "_sleep", lambda seconds: calls.append(seconds))
    throttle = HostThrottle(delay=0.2)
    throttle.wait("a.example")
    throttle.wait("a.example")
    assert len(calls) == 1
    assert 0 < calls[0] <= 0.2


# --- resolve_images: the concurrent batch driver ---------------------------

def _seed_source(session):
    if session.get(Source, "s") is None:
        session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
        session.commit()


def test_resolve_images_skips_items_that_already_have_an_image(session, monkeypatch):
    calls = []
    monkeypatch.setattr(imaging, "_get",
                        lambda url, *, timeout: calls.append(url) or _FakeResponse(200, "", url))
    _seed_source(session)
    item = Item(source_id="s", url="https://example.com/x", url_hash="h1", title="T",
               image_url="https://img.example.com/already.jpg")
    session.add(item)
    session.commit()

    result = resolve_images(session, [item])
    assert calls == []
    assert result.attempted == 0
    assert item.image_url == "https://img.example.com/already.jpg"


def test_resolve_images_skips_items_already_checked(session, monkeypatch):
    import datetime
    calls = []
    monkeypatch.setattr(imaging, "_get",
                        lambda url, *, timeout: calls.append(url) or _FakeResponse(200, "", url))
    _seed_source(session)
    item = Item(source_id="s", url="https://example.com/y", url_hash="h2", title="T",
               image_checked_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc))
    session.add(item)
    session.commit()

    result = resolve_images(session, [item])
    assert calls == []
    assert result.attempted == 0


def test_resolve_images_gains_an_image_and_records_the_attempt(session, monkeypatch):
    monkeypatch.setattr(imaging, "_get",
                        lambda url, *, timeout: _FakeResponse(
                            200, _og_html("https://img.example.com/found.jpg"), url))
    _seed_source(session)
    item = Item(source_id="s", url="https://example.com/z", url_hash="h3", title="T")
    session.add(item)
    session.commit()

    result = resolve_images(session, [item])
    assert result.attempted == 1
    assert result.gained == 1
    assert item.image_url == "https://img.example.com/found.jpg"
    assert item.image_checked_at is not None
    assert result.by_source["s"]["gained"] == 1


def test_resolve_images_transient_failure_is_not_committed_as_checked(session, monkeypatch):
    def _boom(url, *, timeout):
        raise ConnectionError("reset")

    monkeypatch.setattr(imaging, "_get", _boom)
    _seed_source(session)
    item = Item(source_id="s", url="https://example.com/w", url_hash="h4", title="T")
    session.add(item)
    session.commit()

    result = resolve_images(session, [item])
    assert result.attempted == 0  # transient failures are not "attempted"
    assert item.image_url is None
    assert item.image_checked_at is None  # eligible again next run


def test_resolve_images_reports_blocked_status_per_source(session, monkeypatch):
    monkeypatch.setattr(imaging, "_get", lambda url, *, timeout: _FakeResponse(403, "", url))
    _seed_source(session)
    item = Item(source_id="s", url="https://example.com/blocked", url_hash="h5", title="T")
    session.add(item)
    session.commit()

    result = resolve_images(session, [item])
    assert result.attempted == 1
    assert result.gained == 0
    assert result.by_source["s"]["blocked_403"] == 1
    assert item.image_checked_at is not None


def test_resolve_images_respects_the_max_workers_bound(session, monkeypatch):
    """Prove the pool never runs more than max_workers fetches at once: each
    fake fetch registers itself as active, holds briefly, then releases --
    if the bound were not honoured, `peak` would exceed max_workers.
    """
    _seed_source(session)
    items = [
        Item(source_id="s", url=f"https://example.com/{i}", url_hash=f"hh{i}", title="T")
        for i in range(10)
    ]
    session.add_all(items)
    session.commit()

    active = 0
    peak = 0
    lock = threading.Lock()

    def _fake_get(url, *, timeout):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return _FakeResponse(200, "", url)

    monkeypatch.setattr(imaging, "_get", _fake_get)

    result = resolve_images(session, items, max_workers=3, host_delay=0.0)
    assert result.attempted == 10
    assert peak <= 3
    assert peak > 1  # also prove it actually ran concurrently, not serially


def test_resolve_images_is_faster_concurrent_than_it_would_be_serial(session, monkeypatch):
    """Non-vacuity check for the concurrency itself (distinct from the
    bound test above): 10 items each "taking" 0.05s must complete in well
    under 10 * 0.05s = 0.5s serial time when run with a pool of 5.
    """
    _seed_source(session)
    items = [
        Item(source_id="s", url=f"https://example.com/c{i}", url_hash=f"cc{i}", title="T")
        for i in range(10)
    ]
    session.add_all(items)
    session.commit()

    def _fake_get(url, *, timeout):
        time.sleep(0.05)
        return _FakeResponse(200, "", url)

    monkeypatch.setattr(imaging, "_get", _fake_get)

    start = time.monotonic()
    result = resolve_images(session, items, max_workers=5, host_delay=0.0)
    elapsed = time.monotonic() - start
    assert result.attempted == 10
    assert elapsed < 0.3  # would be ~0.5s+ if serialized


def test_resolve_images_progress_callback_reports_final_total(session, monkeypatch):
    monkeypatch.setattr(imaging, "_get", lambda url, *, timeout: _FakeResponse(200, "", url))
    _seed_source(session)
    items = [
        Item(source_id="s", url=f"https://example.com/p{i}", url_hash=f"pp{i}", title="T")
        for i in range(4)
    ]
    session.add_all(items)
    session.commit()

    calls = []
    resolve_images(session, items, on_progress=lambda done, total: calls.append((done, total)))
    assert len(calls) == 4
    assert calls[-1] == (4, 4)
    assert all(total == 4 for _, total in calls)


def test_resolve_images_returns_immediately_for_empty_candidate_list(session):
    result = resolve_images(session, [])
    assert result.attempted == 0
    assert result.gained == 0


# --- Mutation test: prove needs_image_fetch's skip rule is load-bearing ---
#
# tests above rely on needs_image_fetch actually filtering out
# already-imaged/already-checked items before any network call. A test
# suite that would pass identically whether or not that filter runs at all
# is not proving anything. The mutation below (run via a separate `sed`
# invocation from the report-writing agent, mutate -> targeted tests fail
# -> restore in the same shell command) targets exactly that condition;
# this test itself just documents the contract so a future refactor of
# needs_image_fetch has something concrete to keep green.

def test_needs_image_fetch_mutation_target_is_a_real_gate(session, monkeypatch):
    """If needs_image_fetch always returned True (the mutation), an
    already-imaged item would be re-fetched -- this test's own assertion is
    exactly what a `sed`-based mutation of the `if item.image_url:` guard
    in feed/imaging.py would break, which is what proves the guard is
    load-bearing rather than vacuously satisfied."""
    calls = []
    monkeypatch.setattr(imaging, "_get",
                        lambda url, *, timeout: calls.append(url) or _FakeResponse(200, "", url))
    _seed_source(session)
    item = Item(source_id="s", url="https://example.com/already", url_hash="hm1", title="T",
               image_url="https://img.example.com/keep-me.jpg")
    session.add(item)
    session.commit()
    resolve_images(session, [item])
    assert calls == []
    assert item.image_url == "https://img.example.com/keep-me.jpg"
