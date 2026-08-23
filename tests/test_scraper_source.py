from __future__ import annotations
import logging
from datetime import datetime, timezone
from pathlib import Path
import pytest
from feed.sources.registry import build_source
from feed.sources import scraper as scraper_module

FIX = Path(__file__).parent / "fixtures"

GENERIC_CONFIG = {
    "path": str(FIX / "sample_scraper.html"),
    "url": "https://news.example.com/all",  # base for relative-link resolution
    "item_selector": "article.card",
    "title_selector": ".card-title",
    "link_selector": "a.card-link",
    "date_selector": "time.card-date",
    "summary_selector": ".card-summary",
}

ANTHROPIC_CONFIG = {
    "path": str(FIX / "sample_scraper_anthropic.html"),
    "url": "https://www.anthropic.com/news",
    "item_selector": 'a[class*="PublicationList"][class*="listItem"]',
    "title_selector": '[class*="title"]',
    "date_selector": "time",
}


def _build(config: dict, source_id: str = "scraper:test"):
    return build_source("scraper", source_id, dict(config))


# --- Basic scraping: title, link, date, summary -----------------------------

def test_scraper_extracts_title_link_date_summary():
    src = _build(GENERIC_CONFIG)
    items = list(src.fetch(since=None))
    first = items[0]
    assert first.title == "First post title"
    assert first.url == "https://news.example.com/posts/first-post"
    assert first.published_at == datetime(2026, 8, 20, tzinfo=timezone.utc)
    assert first.summary == "Summary of the first post."


def test_scraper_resolves_relative_urls_against_page_url():
    src = _build(GENERIC_CONFIG)
    items = list(src.fetch(since=None))
    # First item's href is relative ("/posts/first-post") -- must resolve
    # against the configured page url, not stay relative.
    assert items[0].url.startswith("https://news.example.com/")


def test_scraper_leaves_absolute_urls_on_other_hosts_untouched():
    src = _build(GENERIC_CONFIG)
    items = list(src.fetch(since=None))
    second = next(i for i in items if "second-post" in i.url)
    assert second.url == "https://other-host.example.com/posts/second-post"


def test_scraper_undated_item_falls_back_to_none_not_a_crash():
    src = _build(GENERIC_CONFIG)
    items = list(src.fetch(since=None))
    undated = next(i for i in items if "undated-post" in i.url)
    assert undated.published_at is None


def test_scraper_unparseable_date_falls_back_to_none_not_a_crash():
    src = _build(GENERIC_CONFIG)
    items = list(src.fetch(since=None))
    bad_date = next(i for i in items if "unparseable-date-post" in i.url)
    assert bad_date.published_at is None


def test_scraper_flexible_date_formats_all_parse():
    # "August 20, 2026" and "2026-08-19" are deliberately different formats
    # in the fixture -- both must parse.
    src = _build(GENERIC_CONFIG)
    items = list(src.fetch(since=None))
    assert items[0].published_at == datetime(2026, 8, 20, tzinfo=timezone.utc)
    second = next(i for i in items if "second-post" in i.url)
    assert second.published_at == datetime(2026, 8, 19, tzinfo=timezone.utc)


def test_scraper_skips_items_with_no_link_and_logs_warning(caplog):
    src = _build(GENERIC_CONFIG)
    with caplog.at_level(logging.WARNING, logger="feed.sources.scraper"):
        items = list(src.fetch(since=None))
    titles = [i.title for i in items]
    assert "Linkless card, must be skipped" not in titles
    assert any("no link" in r.message for r in caplog.records)


def test_scraper_since_filters_out_older_dated_items():
    src = _build(GENERIC_CONFIG)
    since = datetime(2026, 8, 19, tzinfo=timezone.utc)
    items = list(src.fetch(since=since))
    urls = [i.url for i in items]
    assert any("first-post" in u for u in urls)          # Aug 20 > since
    assert not any(u.endswith("second-post") for u in urls)  # Aug 19 == since, excluded


def test_scraper_since_does_not_drop_undated_items():
    # Undated items can never be judged against `since` -- they must always
    # pass through, same rule as every other source plugin.
    src = _build(GENERIC_CONFIG)
    since = datetime(2026, 12, 31, tzinfo=timezone.utc)  # after everything dated
    items = list(src.fetch(since=since))
    assert any("undated-post" in i.url for i in items)


def test_scraper_requires_url_or_path():
    with pytest.raises(ValueError):
        _build({"item_selector": "article"})


def test_scraper_requires_item_selector():
    with pytest.raises(ValueError):
        _build({"path": str(FIX / "sample_scraper.html")})


# --- Anthropic-shaped fixture: proves the real catalogue selectors work -----

def test_anthropic_shaped_page_extracts_three_items():
    src = _build(ANTHROPIC_CONFIG, source_id="anthropic")
    items = list(src.fetch(since=None))
    assert len(items) == 3
    assert items[0].title == "How Claude's text watermark works"
    # canonical_url() strips the "www." prefix -- see feed/sources/base.py
    assert items[0].url == "https://anthropic.com/news/claude-text-watermark"
    assert items[0].published_at == datetime(2026, 8, 14, tzinfo=timezone.utc)


def test_anthropic_shaped_page_resolves_relative_and_keeps_absolute():
    src = _build(ANTHROPIC_CONFIG, source_id="anthropic")
    items = list(src.fetch(since=None))
    # item 2's href is relative ("/news/tino-cuellar")
    assert items[1].url == "https://anthropic.com/news/tino-cuellar"
    # item 3's href is already absolute
    assert items[2].url == "https://anthropic.com/news/position-open-weights-models"


# --- robots.txt compliance ---------------------------------------------------

ROBOTS_DISALLOW_ALL = "User-agent: *\nDisallow: /\n"
ROBOTS_DISALLOW_NEWS = "User-agent: *\nDisallow: /news\n"
ROBOTS_ALLOW_ALL = "User-agent: *\nAllow: /\n"


def test_robots_disallowed_path_skips_the_source_entirely(monkeypatch, caplog):
    def fake_get_text(url, *, timeout):
        assert url == "https://blocked.example.com/robots.txt"
        return ROBOTS_DISALLOW_ALL
    monkeypatch.setattr(scraper_module, "_get_text", fake_get_text)
    monkeypatch.setattr(scraper_module.time, "sleep", lambda s: None)

    src = _build({
        "url": "https://blocked.example.com/news",
        "item_selector": "article",
    }, source_id="blocked")
    with caplog.at_level(logging.WARNING, logger="feed.sources.scraper"):
        items = list(src.fetch(since=None))

    assert items == []
    assert src.coverage_warning is not None
    assert "robots.txt disallows" in src.coverage_warning
    assert any("robots.txt disallows" in r.message for r in caplog.records)


def test_robots_allowed_path_fetches_normally(monkeypatch):
    calls = []

    def fake_get_text(url, *, timeout):
        calls.append(url)
        if url.endswith("/robots.txt"):
            return ROBOTS_ALLOW_ALL
        return (
            '<article class="card"><a class="card-link" href="/p/1">x</a>'
            '<h3 class="card-title">Live title</h3></article>'
        )
    monkeypatch.setattr(scraper_module, "_get_text", fake_get_text)
    monkeypatch.setattr(scraper_module.time, "sleep", lambda s: None)

    src = _build({
        "url": "https://allowed.example.com/news",
        "item_selector": "article.card",
        "title_selector": ".card-title",
        "link_selector": "a.card-link",
    }, source_id="allowed")
    items = list(src.fetch(since=None))

    assert [i.title for i in items] == ["Live title"]
    assert "https://allowed.example.com/robots.txt" in calls
    assert "https://allowed.example.com/news" in calls


def test_robots_path_disallowed_but_other_paths_allowed():
    # Proves the check is path-aware, not host-wide: a robots.txt that
    # disallows /news specifically must still block a page under /news
    # while a different path on the same host would be unaffected.
    import unittest.mock as mock
    import feed.sources.scraper as sm
    with mock.patch.object(sm, "_get_text", return_value=ROBOTS_DISALLOW_NEWS):
        assert sm._allowed_by_robots("https://x.example.com/news/foo", timeout=5) is False
        assert sm._allowed_by_robots("https://x.example.com/other/foo", timeout=5) is True


def test_robots_fetch_failure_treated_as_allowed(monkeypatch):
    # A robots.txt that cannot be fetched at all (network error, no such
    # file, whatever) must not silently block every source that hits it --
    # the conventional interpretation of "no robots.txt" is "no restrictions".
    def boom(url, *, timeout):
        raise RuntimeError("simulated network failure")
    monkeypatch.setattr(scraper_module, "_get_text", boom)

    assert scraper_module._allowed_by_robots(
        "https://unreachable.example.com/news", timeout=5
    ) is True


def test_respect_robots_false_skips_the_check_entirely(monkeypatch):
    # Escape hatch used only to unit-test the scraping logic itself in
    # isolation; production catalogue entries never set this.
    calls = []

    def fake_get_text(url, *, timeout):
        calls.append(url)
        return '<article class="card"><a class="card-link" href="/p/1">x</a></article>'
    monkeypatch.setattr(scraper_module, "_get_text", fake_get_text)
    monkeypatch.setattr(scraper_module.time, "sleep", lambda s: None)

    src = _build({
        "url": "https://x.example.com/news",
        "item_selector": "article.card",
        "link_selector": "a.card-link",
        "respect_robots": False,
    }, source_id="norobots")
    list(src.fetch(since=None))
    assert all("robots.txt" not in c for c in calls)


# --- Mutation-proof of the robots check (non-vacuity) ------------------------

def test_mutation_proof_disallow_all_actually_blocks(monkeypatch):
    """Restores in the same test that mutates: proves the disallow branch is
    load-bearing, not a check that always returns True regardless of
    robots.txt content. If someone deletes the `.can_fetch(...)` call and
    hardcodes `return True`, this test fails; if it's still wired up
    correctly, allow and disallow content genuinely produce different
    answers for the identical URL.
    """
    import unittest.mock as mock
    import feed.sources.scraper as sm
    url = "https://mutation.example.com/news/x"

    with mock.patch.object(sm, "_get_text", return_value=ROBOTS_DISALLOW_ALL):
        disallowed = sm._allowed_by_robots(url, timeout=5)
    with mock.patch.object(sm, "_get_text", return_value=ROBOTS_ALLOW_ALL):
        allowed = sm._allowed_by_robots(url, timeout=5)

    assert disallowed is False
    assert allowed is True
    assert disallowed != allowed  # the mutation-killing assertion


# --- Politeness delay ---------------------------------------------------------

def test_delay_is_applied_between_robots_and_page_requests(monkeypatch):
    sleeps = []
    monkeypatch.setattr(scraper_module.time, "sleep", lambda s: sleeps.append(s))

    def fake_get_text(url, *, timeout):
        return ROBOTS_ALLOW_ALL if url.endswith("robots.txt") else '<article class="card"></article>'
    monkeypatch.setattr(scraper_module, "_get_text", fake_get_text)

    src = _build({
        "url": "https://timing.example.com/news",
        "item_selector": "article.card",
        "delay": 3.5,
    }, source_id="timing")
    list(src.fetch(since=None))
    assert sleeps == [3.5]


def test_default_delay_constant_is_two_seconds():
    assert scraper_module.DEFAULT_DELAY_SECONDS == 2.0


# --- User-Agent is descriptive -----------------------------------------------

def test_user_agent_identifies_the_project():
    ua = scraper_module.USER_AGENT
    assert "ObservatoryFeedBot" in ua or "Observatory" in ua
    assert len(ua) > 20  # not just "bot" -- must carry actual identifying info


# --- Live verification (marked slow; real network) ---------------------------

@pytest.mark.slow
def test_anthropic_scraper_config_against_the_real_page():
    # Verifies the catalogue's actual anthropic selectors still match the
    # real page -- www.anthropic.com publishes no RSS feed (six URLs
    # checked, all 404; see phaseC-report.md), so this is the only source
    # of truth for whether the ScraperSource config in
    # sources.catalogue.toml still works. Marked slow: real network, and
    # honours robots.txt + the configured delay exactly like production.
    import httpx as real_httpx
    import feed.sources.scraper as sm

    def real_get_text(url, *, timeout):
        resp = real_httpx.get(url, timeout=timeout,
                              headers={"User-Agent": sm.USER_AGENT},
                              follow_redirects=True)
        resp.raise_for_status()
        return resp.text

    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    mp.setattr(sm, "_get_text", real_get_text)
    try:
        src = _build({
            "url": "https://www.anthropic.com/news",
            "item_selector": ANTHROPIC_CONFIG["item_selector"],
            "title_selector": ANTHROPIC_CONFIG["title_selector"],
            "date_selector": ANTHROPIC_CONFIG["date_selector"],
        }, source_id="anthropic")
        items = list(src.fetch(since=None))
    finally:
        mp.undo()

    assert len(items) > 0
    assert all(i.title for i in items)
    assert all(i.url.startswith("https://anthropic.com/") for i in items)
