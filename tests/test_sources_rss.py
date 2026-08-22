from datetime import datetime, timezone
from pathlib import Path
from feed.sources.base import canonical_url, url_hash
from feed.sources.registry import build_source

FIXTURE = Path(__file__).parent / "fixtures" / "sample_rss.xml"
UNDATED_FIXTURE = Path(__file__).parent / "fixtures" / "sample_rss_undated.xml"

def test_canonical_url_strips_tracking_params():
    got = canonical_url("https://example.com/a?utm_source=rss&utm_medium=feed&id=7")
    assert got == "https://example.com/a?id=7"

def test_canonical_url_is_stable_across_trivial_differences():
    a = canonical_url("https://Example.com/a/")
    b = canonical_url("http://example.com/a")
    assert a == b
    assert url_hash(a) == url_hash(b)

def test_rss_source_parses_fixture():
    src = build_source("rss", "rss:example", {"path": str(FIXTURE)})
    items = list(src.fetch(since=None))
    assert len(items) == 2
    first = items[0]
    assert first.title == "DeepSeek releases V4"
    assert first.url == "https://example.com/deepseek-v4"
    assert first.published_at == datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)

def test_rss_source_filters_by_since():
    src = build_source("rss", "rss:example", {"path": str(FIXTURE)})
    cutoff = datetime(2026, 8, 18, 10, 0, tzinfo=timezone.utc)
    items = list(src.fetch(since=cutoff))
    assert [i.title for i in items] == ["EU delays AI Act"]

def test_undated_item_survives_the_since_filter():
    # Adjacent to C1: an entry with no <pubDate> at all must never be
    # dropped by `since` filtering just because we can't compare it against
    # a cutoff. Every source's fetch() applies `since is not None and
    # published is not None and published <= since` -- the `published is
    # not None` guard means an undated item (published=None) always
    # survives, regardless of how far in the past `since` is. This pins
    # that already-correct behaviour with committed coverage.
    src = build_source("rss", "rss:example", {"path": str(UNDATED_FIXTURE)})
    cutoff = datetime(2026, 8, 19, 0, 0, tzinfo=timezone.utc)  # after both entries' dates
    items = list(src.fetch(since=cutoff))
    titles = [i.title for i in items]
    assert "Mystery post with no publish date" in titles
    undated = next(i for i in items if i.title == "Mystery post with no publish date")
    assert undated.published_at is None
    # The dated entry, safely before the cutoff, is correctly filtered out --
    # proving this isn't just a case where the filter does nothing at all.
    assert "DeepSeek releases V4" not in titles


def test_unknown_plugin_raises():
    import pytest
    with pytest.raises(KeyError):
        build_source("nope", "x", {})

def test_canonical_url_keeps_params_that_only_share_a_prefix_with_tracking_names():
    assert canonical_url("https://example.com/a?ref_src_page=1") == "https://example.com/a?ref_src_page=1"
    assert canonical_url("https://example.com/a?gclidx=abc") == "https://example.com/a?gclidx=abc"
    assert canonical_url("https://example.com/a?fbclidx=abc") == "https://example.com/a?fbclidx=abc"

def test_canonical_url_strips_exact_match_tracking_params():
    assert canonical_url("https://example.com/a?fbclid=abc&id=7") == "https://example.com/a?id=7"
    assert canonical_url("https://example.com/a?gclid=abc&id=7") == "https://example.com/a?id=7"
    assert canonical_url("https://example.com/a?mc_cid=abc&id=7") == "https://example.com/a?id=7"
    assert canonical_url("https://example.com/a?mc_eid=abc&id=7") == "https://example.com/a?id=7"
    assert canonical_url("https://example.com/a?ref_src=abc&id=7") == "https://example.com/a?id=7"

def test_canonical_url_still_strips_any_utm_prefixed_param():
    assert canonical_url("https://example.com/a?utm_anything=1&id=7") == "https://example.com/a?id=7"

def test_canonical_url_keeps_param_that_merely_starts_with_utm():
    assert canonical_url("https://example.com/a?utmost_view=1") == "https://example.com/a?utmost_view=1"
