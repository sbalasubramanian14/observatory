import json
from datetime import datetime, timezone
from pathlib import Path
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
