from __future__ import annotations
import pytest
from pydantic import ValidationError
from feed.catalogue import CatalogueEntry, load_catalogue


def _write(tmp_path, text):
    p = tmp_path / "cat.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_catalogue_parses_entries(tmp_path):
    p = _write(tmp_path, """
[[source]]
id = "a"
plugin = "rss"
territory = "research"
cadence_minutes = 60
authority = 0.8
config = { url = "https://example.com/a.xml" }

[[source]]
id = "b"
plugin = "scraper"
territory = "industry"
config = { url = "https://example.com/b", item_selector = "article" }
""")
    entries = load_catalogue(p)
    assert [e.id for e in entries] == ["a", "b"]
    assert entries[0].territory == "research"
    assert entries[0].authority == 0.8
    assert entries[1].cadence_minutes == 60  # default
    assert entries[1].authority == 0.5  # default


def test_load_catalogue_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_catalogue(tmp_path / "does-not-exist.toml")


def test_load_catalogue_rejects_duplicate_ids(tmp_path):
    p = _write(tmp_path, """
[[source]]
id = "dup"
plugin = "rss"
territory = "research"
config = { url = "https://example.com/1.xml" }

[[source]]
id = "dup"
plugin = "rss"
territory = "industry"
config = { url = "https://example.com/2.xml" }
""")
    with pytest.raises(ValueError, match="duplicate"):
        load_catalogue(p)


def test_catalogue_entry_rejects_unknown_territory():
    with pytest.raises(ValidationError):
        CatalogueEntry(id="x", plugin="rss", territory="business", config={})


def test_catalogue_entry_defaults():
    e = CatalogueEntry(id="x", plugin="rss", territory="policy", config={"url": "u"})
    assert e.enabled is True
    assert e.max_backfill_days is None
    assert e.cadence_minutes == 60
    assert e.authority == 0.5
