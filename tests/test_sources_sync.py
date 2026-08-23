from __future__ import annotations
from sqlalchemy import select
from feed.catalogue import CatalogueEntry
from feed.models import Item, Source, Stage
from feed.stages.sync import sync_sources


def _entry(**kw) -> CatalogueEntry:
    base = dict(id="src", plugin="rss", territory="research",
               cadence_minutes=60, authority=0.7, config={"url": "https://example.com/f.xml"})
    base.update(kw)
    return CatalogueEntry(**base)


def test_sync_adds_a_new_source(session):
    res = sync_sources(session, [_entry(id="new-src")])
    assert res.added == ["new-src"]
    assert res.updated == []
    row = session.get(Source, "new-src")
    assert row is not None
    assert row.plugin == "rss"
    assert row.territory == "research"
    assert row.enabled is True


def test_sync_updates_an_existing_source_with_changed_config(session):
    session.add(Source(id="s", plugin="rss", config={"url": "https://old.example.com/f.xml"},
                       cadence_minutes=30, authority=0.4, territory="industry", enabled=True))
    session.commit()

    res = sync_sources(session, [_entry(id="s", config={"url": "https://new.example.com/f.xml"},
                                        authority=0.9, territory="research")])
    assert res.updated == ["s"]
    row = session.get(Source, "s")
    assert row.config == {"url": "https://new.example.com/f.xml"}
    assert row.authority == 0.9
    assert row.territory == "research"


def test_sync_is_idempotent_unchanged_entries_reported_as_unchanged(session):
    entry = _entry(id="stable")
    sync_sources(session, [entry])
    res2 = sync_sources(session, [entry])
    assert res2.added == []
    assert res2.updated == []
    assert res2.unchanged == ["stable"]


def test_sync_clears_stale_failure_state_when_config_changes(session):
    session.add(Source(id="broken", plugin="rss", config={"url": "https://dead.example.com/f.xml"},
                       cadence_minutes=60, authority=0.5, enabled=True,
                       consecutive_failures=7, last_error="404 not found",
                       coverage_warning="stale warning"))
    session.commit()

    sync_sources(session, [_entry(id="broken", config={"url": "https://fixed.example.com/f.xml"})])
    row = session.get(Source, "broken")
    assert row.consecutive_failures == 0
    assert row.last_error is None
    assert row.coverage_warning is None


def test_sync_preserves_failure_state_when_nothing_changed(session):
    entry = _entry(id="flaky")
    sync_sources(session, [entry])
    row = session.get(Source, "flaky")
    row.consecutive_failures = 3
    row.last_error = "timeout"
    session.commit()

    sync_sources(session, [entry])  # identical entry, re-synced
    row2 = session.get(Source, "flaky")
    assert row2.consecutive_failures == 3
    assert row2.last_error == "timeout"


def test_sync_deletes_a_removed_source_with_zero_items(session):
    session.add(Source(id="dead", plugin="rss", config={"url": "https://dead.example.com/f.xml"},
                       cadence_minutes=60, authority=0.5, enabled=True))
    session.commit()

    res = sync_sources(session, [])  # catalogue no longer mentions it
    assert res.deleted == ["dead"]
    assert session.get(Source, "dead") is None


def test_sync_disables_rather_than_deletes_a_removed_source_with_items(session):
    session.add(Source(id="had-items", plugin="rss", config={"url": "https://x.example.com/f.xml"},
                       cadence_minutes=60, authority=0.5, enabled=True))
    session.commit()
    session.add(Item(source_id="had-items", url="https://x.example.com/1", url_hash="h1",
                     title="t", stage=Stage.COLLECTED))
    session.commit()

    res = sync_sources(session, [])
    assert res.deleted == []
    assert res.disabled == ["had-items"]
    row = session.get(Source, "had-items")
    assert row is not None
    assert row.enabled is False


def test_sync_leaves_an_already_disabled_removed_source_alone(session):
    session.add(Source(id="already-off", plugin="rss", config={"url": "https://x.example.com/f.xml"},
                       cadence_minutes=60, authority=0.5, enabled=False))
    session.commit()

    res = sync_sources(session, [])
    assert res.disabled == []
    assert res.deleted == []


def test_sync_reenables_a_previously_disabled_source_present_in_catalogue(session):
    session.add(Source(id="s", plugin="rss", config={"url": "https://x.example.com/f.xml"},
                       cadence_minutes=60, authority=0.5, enabled=False,
                       consecutive_failures=2, last_error="old error"))
    session.commit()

    res = sync_sources(session, [_entry(id="s")])
    assert res.updated == ["s"]
    row = session.get(Source, "s")
    assert row.enabled is True
    assert row.consecutive_failures == 0
    assert row.last_error is None


def test_sync_reports_multiple_sources_independently(session):
    session.add(Source(id="keep", plugin="rss", config={"url": "https://example.com/f.xml"},
                       cadence_minutes=60, authority=0.7, enabled=True, territory="research"))
    session.add(Source(id="drop", plugin="rss", config={"url": "https://x.example.com/drop.xml"},
                       cadence_minutes=60, authority=0.5, enabled=True, territory="research"))
    session.commit()

    res = sync_sources(session, [
        _entry(id="keep", territory="research"),
        _entry(id="brand-new"),
    ])
    assert res.added == ["brand-new"]
    assert res.unchanged == ["keep"]
    assert res.deleted == ["drop"]
