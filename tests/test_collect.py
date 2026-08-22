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


# --- I7: a DB error during the item-insert loop or commit must isolate ----
# just that source, not abort the whole collect() run --------------------
#
# Only fetch() was inside the original try -- the item-insert loop and both
# commits sat outside it, so a DB error there (e.g. a full disk, a
# constraint violation, any commit() failure) escaped collect() entirely
# and aborted every other still-due source in the same run. Spec S3.1
# requires per-source isolation. This mutates session.commit() to fail on
# whichever source's commit call happens first (regardless of iteration
# order), and proves the OTHER source is still fully collected in the same
# collect() call.
def test_db_error_during_item_insert_isolates_that_source_from_others(session, monkeypatch):
    _add_source(session, id="a", config={"path": str(FIX)})
    _add_source(session, id="b", config={"path": str(FIX)})

    real_commit = session.commit
    calls = {"n": 0}

    def flaky_commit():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("db exploded during item insert")
        return real_commit()

    monkeypatch.setattr(session, "commit", flaky_commit)

    res = collect(session, now=NOW)  # must not raise

    assert len(res.source_errors) == 1
    failed_id = next(iter(res.source_errors))
    ok_id = "a" if failed_id == "b" else "b"
    assert failed_id in ("a", "b") and ok_id in ("a", "b") and failed_id != ok_id

    ok_src = session.get(Source, ok_id)
    assert ok_src.consecutive_failures == 0
    assert ok_src.last_run_at == NOW
    assert session.query(Item).filter_by(source_id=ok_id).count() == 2, (
        "the source processed after the failing one must still be fully "
        "collected in the SAME collect() call, not skipped or partial"
    )

    bad_src = session.get(Source, failed_id)
    assert bad_src.consecutive_failures == 1
    assert bad_src.last_error is not None
    assert session.query(Item).filter_by(source_id=failed_id).count() == 0, (
        "the failing source's partial inserts must be rolled back, not "
        "left half-committed"
    )


# --- Required test 1: source health -----------------------------------
#
# Both behaviours below are correct in today's collect() but had zero
# committed coverage before this change; spec S4.2 makes source health
# (consecutive_failures, `feed sources list`'s "FAILING x{n}" state)
# load-bearing.
def test_consecutive_failures_resets_to_zero_on_a_later_success(session):
    _add_source(session, id="flaky", config={"path": "/nonexistent.xml"})
    collect(session, now=NOW)
    src = session.get(Source, "flaky")
    assert src.consecutive_failures == 1
    assert src.last_error is not None

    # Operator repairs the source; the next cadence-eligible run succeeds.
    src.config = {"path": str(FIX)}
    session.commit()
    res = collect(session, now=NOW + timedelta(minutes=31))

    src = session.get(Source, "flaky")
    assert src.consecutive_failures == 0
    assert src.last_error is None
    assert res.new_items == 2


def test_cross_source_url_hash_collision_dedupes(session):
    # The same article URL arriving via two different sources (e.g. an RSS
    # mirror and a source republishing the same canonical link) must dedupe
    # on url_hash regardless of which source it came from -- the exists
    # check in collect() is not scoped by source_id.
    _add_source(session, id="a", config={"path": str(FIX)})
    _add_source(session, id="b", config={"path": str(FIX)})

    res = collect(session, now=NOW)

    assert res.new_items == 2            # only the first-processed source's items are new
    assert res.skipped_duplicates == 2   # the other source's identical URLs dedupe
    assert session.query(Item).count() == 2
