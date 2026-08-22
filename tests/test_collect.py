from datetime import datetime, timedelta, timezone
from pathlib import Path
from feed.models import Item, Source, Stage
from feed.stages.collect import collect

FIX = Path(__file__).parent / "fixtures" / "sample_rss.xml"
# Ruling: the brief's NOW (2026-08-19 12:00) is after both fixture items
# (2026-08-18 09:00 and 11:30), so any collect() using it as `since` on a
# later call would drop both items, making idempotency/cadence assertions
# unpassable. Using a NOW before both fixture items keeps `since=last_run_at`
# semantics intact while letting later collect() calls still see them.
NOW = datetime(2026, 8, 18, 8, 0, tzinfo=timezone.utc)

def _add_source(session, **kw):
    defaults = dict(id="rss:example", plugin="rss", config={"path": str(FIX)},
                    cadence_minutes=30, enabled=True)
    defaults.update(kw)
    session.add(Source(**defaults))
    session.commit()

def test_collect_inserts_items_at_collected_stage(session):
    _add_source(session)
    res = collect(session, now=NOW)
    assert res.new_items == 2
    items = session.query(Item).all()
    assert len(items) == 2
    assert all(i.stage is Stage.COLLECTED for i in items)

def test_collect_is_idempotent_on_url_hash(session):
    _add_source(session)
    collect(session, now=NOW)
    res = collect(session, now=NOW + timedelta(hours=1))
    assert res.new_items == 0
    assert res.skipped_duplicates == 2
    assert session.query(Item).count() == 2

def test_disabled_source_is_skipped(session):
    _add_source(session, enabled=False)
    res = collect(session, now=NOW)
    assert res.new_items == 0

def test_cadence_prevents_early_refetch(session):
    _add_source(session, cadence_minutes=60)
    collect(session, now=NOW)
    session.query(Item).delete()
    session.commit()
    res = collect(session, now=NOW + timedelta(minutes=10))
    assert res.new_items == 0          # too soon
    res = collect(session, now=NOW + timedelta(minutes=61))
    assert res.new_items == 2

def test_broken_source_is_recorded_and_does_not_raise(session):
    _add_source(session, id="bad", config={"path": "/nonexistent.xml"})
    res = collect(session, now=NOW)
    assert "bad" in res.source_errors
    src = session.get(Source, "bad")
    assert src.consecutive_failures == 1
    assert src.last_error is not None
