import json
from datetime import datetime, timezone
from pathlib import Path
import pytest
from feed.sources.registry import build_source

FIX = Path(__file__).parent / "fixtures"


def test_arxiv_builds_abs_urls_and_titles():
    src = build_source("arxiv", "arxiv:cs.AI", {"path": str(FIX / "sample_arxiv.xml")})
    items = list(src.fetch(since=None))
    assert len(items) == 2
    assert items[0].url == "https://arxiv.org/abs/2607.09510"
    assert "Failure as a Process" in items[0].title
    assert items[0].published_at == datetime(2026, 7, 13, 10, 4, tzinfo=timezone.utc)
    assert items[0].published_at.tzinfo is not None


def test_arxiv_strips_version_suffix_so_v1_and_v2_share_a_url():
    src = build_source("arxiv", "arxiv:cs.AI", {"path": str(FIX / "sample_arxiv.xml")})
    urls = [i.url for i in src.fetch(since=None)]
    assert all("v1" not in u and "v2" not in u for u in urls)
    assert urls[1] == "https://arxiv.org/abs/2606.17799"


def test_arxiv_version_regex_handles_single_and_double_digit_versions_and_no_version():
    # Risk item 1: v1, v12 (two-digit), no version suffix, and a malformed id
    # must all be handled correctly by the version-stripping regex.
    from feed.sources.arxiv import _ABS

    m1 = _ABS.search("http://arxiv.org/abs/2607.09510v1")
    assert m1 is not None
    assert m1.group("id") == "2607.09510"

    m2 = _ABS.search("http://arxiv.org/abs/2606.17799v12")
    assert m2 is not None
    assert m2.group("id") == "2606.17799"

    m3 = _ABS.search("http://arxiv.org/abs/2607.09510")
    assert m3 is not None
    assert m3.group("id") == "2607.09510"

    # Malformed / old-style id (letters before the numeric part, e.g. the
    # pre-2007 "hep-th/9901001" scheme) does not match [\d.]+ and must not
    # produce a false match.
    m4 = _ABS.search("http://arxiv.org/abs/hep-th/9901001v3")
    assert m4 is None


def test_arxiv_since_does_not_drop_entries_with_no_parseable_date(tmp_path):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2608.00001v1</id>
    <title>No date fields at all</title>
    <summary>Nothing to parse a date from.</summary>
  </entry>
</feed>
"""
    p = tmp_path / "arxiv_undated.xml"
    p.write_text(xml, encoding="utf-8")
    src = build_source("arxiv", "arxiv:cs.AI", {"path": str(p)})
    cutoff = datetime(2026, 8, 20, tzinfo=timezone.utc)  # far in the future of nothing
    items = list(src.fetch(since=cutoff))
    assert len(items) == 1
    assert items[0].published_at is None


def test_hackernews_applies_min_score_and_skips_urlless():
    src = build_source("hackernews", "hn", {"path": str(FIX / "sample_hn.json"), "min_score": 100})
    items = list(src.fetch(since=None))
    assert [i.title for i in items] == ["Show HN: DeepSeek V4 weights are up"]


def test_hackernews_skips_wrong_type_even_with_url_and_high_score():
    # Fixture item id=4 has type "job", a url, and score 500 (>= min_score).
    # If the type filter were missing or short-circuited by the other two
    # checks, this item would leak through.
    src = build_source("hackernews", "hn", {"path": str(FIX / "sample_hn.json"), "min_score": 100})
    titles = [i.title for i in src.fetch(since=None)]
    assert "A job posting" not in titles


def test_hackernews_published_at_is_timezone_aware():
    src = build_source("hackernews", "hn", {"path": str(FIX / "sample_hn.json"), "min_score": 100})
    items = list(src.fetch(since=None))
    assert items[0].published_at == datetime(2026, 8, 17, 20, 53, 20, tzinfo=timezone.utc)
    assert items[0].published_at.tzinfo is not None


def test_hackernews_since_does_not_drop_items_missing_the_time_field(tmp_path):
    # HN's public API always sets `time`, but the source should not crash or
    # silently drop an item if it were ever absent/malformed -- same rule as
    # the undated-item handling in arxiv/rss/github_releases.
    data = [
        {"id": 9, "title": "No time field", "url": "https://example.com/x", "score": 999, "type": "story"},
    ]
    p = tmp_path / "hn_no_time.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    src = build_source("hackernews", "hn", {"path": str(p), "min_score": 100})
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    items = list(src.fetch(since=cutoff))
    assert [i.title for i in items] == ["No time field"]
    assert items[0].published_at is None


def test_github_releases_url_is_built_from_repo():
    src = build_source("github_releases", "gh:vllm", {"repo": "vllm-project/vllm"})
    assert src.feed_url == "https://github.com/vllm-project/vllm/releases.atom"


def test_github_releases_since_does_not_drop_entries_with_no_parseable_date(tmp_path):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <link href="https://github.com/vllm-project/vllm/releases/tag/v9.9.9"/>
    <title>v9.9.9</title>
  </entry>
</feed>
"""
    p = tmp_path / "gh_undated.xml"
    p.write_text(xml, encoding="utf-8")
    src = build_source("github_releases", "gh:vllm", {"repo": "vllm-project/vllm", "path": str(p)})
    cutoff = datetime(2026, 8, 20, tzinfo=timezone.utc)
    items = list(src.fetch(since=cutoff))
    assert len(items) == 1
    assert items[0].published_at is None


def test_importing_feed_sources_twice_does_not_raise():
    import importlib
    import feed.sources  # noqa: F401
    import feed.sources  # second import must be a no-op (sys.modules cache)

    importlib.import_module("feed.sources")


# --- Coverage-loss logging (code review follow-up) -------------------------
#
# Category A (unexpected shape: unparseable arxiv id, linkless rss/github
# entry) must log at WARNING with enough detail to identify the offender.
# Category B (hackernews' intentional score/type/url filtering) must NOT
# log -- it's routine, and logging it would drown Category A in noise.

import logging


def test_arxiv_logs_warning_for_unparseable_id(tmp_path, caplog):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/hep-th/9901001v3</id>
    <title>Old-style id, unparseable by the version regex</title>
  </entry>
</feed>
"""
    p = tmp_path / "arxiv_malformed.xml"
    p.write_text(xml, encoding="utf-8")
    src = build_source("arxiv", "arxiv:cs.AI", {"path": str(p)})
    with caplog.at_level(logging.WARNING, logger="feed.sources.arxiv"):
        items = list(src.fetch(since=None))
    assert items == []
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "hep-th/9901001v3" in caplog.records[0].message


def test_rss_logs_warning_for_linkless_entry(tmp_path, caplog):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item><title>No link here</title></item>
</channel></rss>
"""
    p = tmp_path / "rss_linkless.xml"
    p.write_text(xml, encoding="utf-8")
    src = build_source("rss", "rss:example", {"path": str(p)})
    with caplog.at_level(logging.WARNING, logger="feed.sources.rss"):
        items = list(src.fetch(since=None))
    assert items == []
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "No link here" in caplog.records[0].message


def test_github_releases_logs_warning_for_linkless_entry(tmp_path, caplog):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Release with no link element</title>
  </entry>
</feed>
"""
    p = tmp_path / "gh_linkless.xml"
    p.write_text(xml, encoding="utf-8")
    src = build_source("github_releases", "gh:vllm", {"repo": "vllm-project/vllm", "path": str(p)})
    with caplog.at_level(logging.WARNING, logger="feed.sources.github_releases"):
        items = list(src.fetch(since=None))
    assert items == []
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "Release with no link element" in caplog.records[0].message


def test_hackernews_does_not_log_for_intentional_filtering(caplog):
    # Score/type/url filtering is HN doing its job, not a coverage gap.
    # Logging it would emit noise on every run and drown category-A signals.
    src = build_source("hackernews", "hn", {"path": str(FIX / "sample_hn.json"), "min_score": 100})
    with caplog.at_level(logging.DEBUG, logger="feed.sources.hackernews"):
        items = list(src.fetch(since=None))
    assert len(items) == 1  # sanity: filtering is still happening
    assert caplog.records == []


# --- A1: arXiv pagination ------------------------------------------------
#
# A single request capped at max_results (e.g. 60) loses the vast majority
# of a multi-day backlog. fetch() now pages via start= until an entry is
# reached that is not newer than `since`, or the max_pages safety cap is
# hit, or the API runs out of results.

def _arxiv_atom_page(entries: list[tuple[str, str, str]]) -> str:
    """entries: list of (numeric_id, iso_date, title)."""
    items = "".join(
        f"""  <entry>
    <id>http://arxiv.org/abs/{num_id}v1</id>
    <updated>{iso_date}</updated>
    <published>{iso_date}</published>
    <title>{title}</title>
    <summary>summary for {title}</summary>
  </entry>
"""
        for num_id, iso_date, title in entries
    )
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom">\n{items}</feed>\n'


def test_arxiv_pagination_fetches_beyond_a_single_page_and_stops_at_since(tmp_path):
    page1 = _arxiv_atom_page([
        ("2608.00020", "2026-08-20T10:00:00Z", "Newest paper"),
        ("2608.00019", "2026-08-19T10:00:00Z", "Second newest paper"),
    ])
    page2 = _arxiv_atom_page([
        ("2608.00018", "2026-08-18T10:00:00Z", "Third page-two paper (kept)"),
        ("2608.00017", "2026-08-17T10:00:00Z", "At the since cutoff (dropped)"),
    ])
    p1 = tmp_path / "page1.xml"
    p2 = tmp_path / "page2.xml"
    p1.write_text(page1, encoding="utf-8")
    p2.write_text(page2, encoding="utf-8")

    src = build_source("arxiv", "arxiv:cs.AI", {
        "paths": [str(p1), str(p2)], "max_results": 2,
    })
    since = datetime(2026, 8, 17, 10, 0, 0, tzinfo=timezone.utc)
    items = list(src.fetch(since=since))
    titles = [i.title for i in items]

    # Proves pagination genuinely crossed a page boundary: titles from both
    # page1 AND page2 are present.
    assert "Newest paper" in titles
    assert "Second newest paper" in titles
    assert "Third page-two paper (kept)" in titles
    # Proves it genuinely stopped AT since, not just ran out of pages: the
    # entry exactly at the cutoff (and anything that would follow on a
    # hypothetical page 3) is excluded.
    assert "At the since cutoff (dropped)" not in titles
    assert len(items) == 3


def test_arxiv_pagination_stops_at_max_pages_cap(tmp_path, caplog):
    import logging
    pages = [
        _arxiv_atom_page([
            (f"2608.{n:05d}", "2026-08-2" + str(9 - n) + "T10:00:00Z", f"Paper {n}"),
            (f"2608.{n + 1:05d}", "2026-08-2" + str(9 - n) + "T09:00:00Z", f"Paper {n + 1}"),
        ])
        for n in (1, 3, 5)
    ]
    paths = []
    for i, page in enumerate(pages):
        p = tmp_path / f"page{i}.xml"
        p.write_text(page, encoding="utf-8")
        paths.append(str(p))

    src = build_source("arxiv", "arxiv:cs.AI", {
        "paths": paths, "max_results": 2, "max_pages": 2,
    })
    with caplog.at_level(logging.WARNING, logger="feed.sources.arxiv"):
        items = list(src.fetch(since=None))

    # Only the first two pages (max_pages=2) were consumed -- the third
    # fixture's papers never appear, proving the cap actually stopped
    # pagination rather than exhausting all supplied pages.
    titles = [i.title for i in items]
    assert len(items) == 4
    assert "Paper 5" not in titles and "Paper 6" not in titles
    assert any("max_pages" in r.message for r in caplog.records)


def test_arxiv_single_path_fixture_is_still_one_page_only(tmp_path):
    # Backward compatibility: the old singular `path=` mechanism must
    # still behave as exactly one page, never paginating further, even
    # when max_results is set small enough that a live fetch would.
    page = _arxiv_atom_page([
        ("2608.00001", "2026-08-20T10:00:00Z", "Only entry"),
    ])
    p = tmp_path / "only.xml"
    p.write_text(page, encoding="utf-8")
    src = build_source("arxiv", "arxiv:cs.AI", {"path": str(p), "max_results": 1})
    items = list(src.fetch(since=None))
    assert [i.title for i in items] == ["Only entry"]


@pytest.mark.slow
def test_arxiv_pagination_against_the_real_api(monkeypatch):
    # Verifies the real arXiv API genuinely returns MORE than max_results
    # once paginated -- the actual bug being fixed. Marked slow: real
    # network, and honours the ~3s rate limit between the two page
    # requests it makes.
    import httpx as real_httpx
    import feed.sources.arxiv as arxiv_module

    def real_get(url, *, timeout):
        resp = real_httpx.get(url, timeout=timeout,
                               headers={"User-Agent": "feed/0.1 (personal reader)"})
        resp.raise_for_status()
        return resp.content

    monkeypatch.setattr(arxiv_module, "_get", real_get)
    src = build_source("arxiv", "arxiv:cs.AI", {
        "categories": ["cs.AI"], "max_results": 25, "max_pages": 3,
    })
    items = list(src.fetch(since=None))
    assert len(items) > 25  # proves pagination happened, not just one request


# --- A2: Hacker News via the Algolia search_by_date API -------------------
#
# topstories.json only ever returns what is top *right now* -- a story
# that peaked two days ago is unreachable from it at any page size. The
# Algolia HN Search API is queryable by time (created_at_i) and paginates
# via page=/nbPages, so it can answer "everything since my last run".
# Verified live 2026-08-22/23 via curl: hn.algolia.com/api/v1/search_by_date
# returns {"hits": [{"points", "created_at_i", "url", "title", "_tags", ...}],
# "nbPages", "page", "hitsPerPage", ...} exactly as used below.

def _algolia_page(hits: list[dict], *, page: int, nb_pages: int) -> dict:
    return {
        "hits": [
            {
                "title": h["title"], "url": h.get("url"), "points": h["points"],
                "created_at_i": h["created_at_i"], "_tags": ["story"],
                "objectID": str(h.get("id", 1)),
            }
            for h in hits
        ],
        "page": page, "nbPages": nb_pages, "hitsPerPage": len(hits),
    }


def test_hackernews_algolia_paginates_across_multiple_pages(monkeypatch):
    import feed.sources.hackernews as hn_module

    page0 = _algolia_page([
        {"id": 1, "title": "Recent story A", "url": "https://example.com/a",
         "points": 200, "created_at_i": 1755800000},
        {"id": 2, "title": "Recent story B", "url": "https://example.com/b",
         "points": 200, "created_at_i": 1755790000},
    ], page=0, nb_pages=2)
    page1 = _algolia_page([
        {"id": 3, "title": "Older story C", "url": "https://example.com/c",
         "points": 200, "created_at_i": 1755780000},
    ], page=1, nb_pages=2)
    responses = [page0, page1]
    calls = []

    def fake_get(url, *, params, timeout):
        calls.append(dict(params))
        return responses[params["page"]]

    monkeypatch.setattr(hn_module, "_get", fake_get)
    src = build_source("hackernews", "hn", {"min_score": 100, "limit": 2})
    items = list(src.fetch(since=None))

    assert [i.title for i in items] == ["Recent story A", "Recent story B", "Older story C"]
    assert len(calls) == 2  # proves it paginated, not just fetched page 0
    assert calls[0]["page"] == 0 and calls[1]["page"] == 1
    assert "points>=100" in calls[0]["numericFilters"]


def test_hackernews_algolia_stops_when_nbpages_reached(monkeypatch):
    import feed.sources.hackernews as hn_module

    page0 = _algolia_page([
        {"id": 1, "title": "Only story", "url": "https://example.com/a",
         "points": 200, "created_at_i": 1755800000},
    ], page=0, nb_pages=1)
    calls = []

    def fake_get(url, *, params, timeout):
        calls.append(dict(params))
        return page0

    monkeypatch.setattr(hn_module, "_get", fake_get)
    src = build_source("hackernews", "hn", {"min_score": 100})
    items = list(src.fetch(since=None))

    assert [i.title for i in items] == ["Only story"]
    assert len(calls) == 1  # nbPages=1 -> must not request page 1


def test_hackernews_algolia_since_becomes_created_at_i_filter(monkeypatch):
    import feed.sources.hackernews as hn_module

    captured = {}

    def fake_get(url, *, params, timeout):
        captured.update(params)
        return _algolia_page([], page=0, nb_pages=1)

    monkeypatch.setattr(hn_module, "_get", fake_get)
    src = build_source("hackernews", "hn", {"min_score": 50})
    since = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)
    list(src.fetch(since=since))

    assert f"created_at_i>{int(since.timestamp())}" in captured["numericFilters"]
    assert "points>=50" in captured["numericFilters"]


def test_hackernews_algolia_max_pages_cap(monkeypatch, caplog):
    import logging
    import feed.sources.hackernews as hn_module

    def fake_get(url, *, params, timeout):
        page = params["page"]
        return _algolia_page([
            {"id": page, "title": f"Story page {page}", "url": f"https://example.com/{page}",
             "points": 200, "created_at_i": 1755800000 - page},
        ], page=page, nb_pages=50)  # far more pages available than the cap

    monkeypatch.setattr(hn_module, "_get", fake_get)
    src = build_source("hackernews", "hn", {"min_score": 100, "max_pages": 2})
    with caplog.at_level(logging.WARNING, logger="feed.sources.hackernews"):
        items = list(src.fetch(since=None))

    assert len(items) == 2  # pages 0 and 1 only, despite nbPages=50
    assert any("max_pages" in r.message for r in caplog.records)


def test_hackernews_old_path_fixture_mechanism_still_works():
    # Unchanged from before A2: the flat-list JSON fixture format keeps
    # working exactly as it did against the old topstories.json shape.
    src = build_source("hackernews", "hn", {"path": str(FIX / "sample_hn.json"), "min_score": 100})
    items = list(src.fetch(since=None))
    assert [i.title for i in items] == ["Show HN: DeepSeek V4 weights are up"]


@pytest.mark.slow
def test_hackernews_algolia_endpoint_live_and_shaped_as_expected(monkeypatch):
    # Verifies the real Algolia endpoint before-the-fact assumption baked
    # into hackernews.py: field names (hits/points/created_at_i/url/title),
    # and that nbPages/page/hitsPerPage exist for pagination to work off
    # of. Marked slow: real network.
    import httpx as real_httpx
    import feed.sources.hackernews as hn_module

    def real_get(url, *, params, timeout):
        resp = real_httpx.get(url, params=params, timeout=timeout,
                               headers={"User-Agent": "feed/0.1 (personal reader)"})
        resp.raise_for_status()
        return resp.json()

    monkeypatch.setattr(hn_module, "_get", real_get)
    since = datetime.now(timezone.utc).replace(microsecond=0)
    since = since.fromtimestamp(since.timestamp() - 3 * 86400, tz=timezone.utc)
    src = build_source("hackernews", "hn", {"min_score": 50, "limit": 5})
    items = list(src.fetch(since=since))

    assert len(items) > 0
    assert all(i.title for i in items)
    assert all(i.url for i in items)
    assert all(i.published_at is None or i.published_at > since for i in items)
