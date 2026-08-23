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


# --- A3: RSS window-truncation heuristic ------------------------------
#
# A feed document only ever carries the publisher's last N entries; there
# is no pagination to fall back on. What CAN be detected: if every dated
# entry this fetch saw postdates `since`, none of them overlap with the
# last run, which means the feed's window most likely rolled entirely past
# `since` between runs and whatever published in between is gone. A mix
# (some entries <= since) means the window still reaches back far enough
# and must stay silent -- that's the routine, non-lossy case.

import logging


def test_rss_truncation_warning_fires_when_every_entry_postdates_since(caplog):
    # Both fixture entries are 09:00 and 11:30; a cutoff before both means
    # neither overlaps with "since" -- exactly the truncation signature.
    cutoff = datetime(2026, 8, 18, 8, 0, tzinfo=timezone.utc)
    src = build_source("rss", "rss:example", {"path": str(FIXTURE)})
    with caplog.at_level(logging.WARNING, logger="feed.sources.rss"):
        items = list(src.fetch(since=cutoff))
    assert len(items) == 2  # sanity: both entries were fetched
    assert src.coverage_warning is not None
    assert "rss:example" in src.coverage_warning
    assert any(r.levelno == logging.WARNING and "rss:example" in r.message
               for r in caplog.records)


def test_rss_truncation_warning_stays_silent_when_some_entries_predate_since(caplog):
    # cutoff sits between the two fixture entries (09:00 and 11:30) -- one
    # entry is older than `since`, proving the feed's window still reaches
    # back far enough. Must NOT warn.
    cutoff = datetime(2026, 8, 18, 10, 0, tzinfo=timezone.utc)
    src = build_source("rss", "rss:example", {"path": str(FIXTURE)})
    with caplog.at_level(logging.WARNING, logger="feed.sources.rss"):
        items = list(src.fetch(since=cutoff))
    assert [i.title for i in items] == ["EU delays AI Act"]  # sanity: filter still ran
    assert src.coverage_warning is None
    assert caplog.records == []


def test_rss_truncation_warning_is_skipped_on_a_first_run_with_no_since():
    # since=None (first run / no prior last_run_at) has no gap to compare
    # against -- every entry trivially "postdates" nothing, and flagging
    # that would make every brand-new source warn on its very first fetch.
    src = build_source("rss", "rss:example", {"path": str(FIXTURE)})
    items = list(src.fetch(since=None))
    assert len(items) == 2
    assert src.coverage_warning is None


def test_rss_truncation_warning_stays_silent_on_a_mixed_dated_undated_feed():
    # UNDATED_FIXTURE has one dated entry (09:00, predates this cutoff) and
    # one entry with no date at all. The dated entry predating `since`
    # means the window still reaches back far enough -- must stay silent.
    cutoff = datetime(2026, 8, 19, 0, 0, tzinfo=timezone.utc)
    src = build_source("rss", "rss:example", {"path": str(UNDATED_FIXTURE)})
    items = list(src.fetch(since=cutoff))
    assert len(items) == 1  # sanity: the undated entry always survives
    assert src.coverage_warning is None


def test_rss_truncation_warning_is_skipped_when_no_entry_has_a_parseable_date(tmp_path):
    # No dated entries at all -- there is nothing to judge truncation from,
    # so this must not fire regardless of `since`.
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item><title>No date A</title><link>https://example.com/a</link></item>
  <item><title>No date B</title><link>https://example.com/b</link></item>
</channel></rss>
"""
    p = tmp_path / "all_undated.xml"
    p.write_text(xml, encoding="utf-8")
    cutoff = datetime(2026, 8, 19, 0, 0, tzinfo=timezone.utc)
    src = build_source("rss", "rss:example", {"path": str(p)})
    items = list(src.fetch(since=cutoff))
    assert len(items) == 2
    assert src.coverage_warning is None
