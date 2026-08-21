from datetime import datetime, timezone
from pathlib import Path
from feed.sources.base import canonical_url, url_hash
from feed.sources.registry import build_source

FIXTURE = Path(__file__).parent / "fixtures" / "sample_rss.xml"

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

def test_unknown_plugin_raises():
    import pytest
    with pytest.raises(KeyError):
        build_source("nope", "x", {})
