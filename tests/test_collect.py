import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from feed.config import CollectConfig
from feed.models import Item, Source, Stage
from feed.stages.collect import _effective_since, collect

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


# --- A4: backfill cap --------------------------------------------------
#
# "if the gap is large, at least it should pull 2 days data" -- a machine
# off for three weeks (or longer) must not ask a source for three weeks of
# history, and a brand-new source must not drag in its entire archive.

NOW2 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def test_effective_since_pure_function_first_run_always_capped():
    since, capped = _effective_since(None, NOW2, 2)
    assert since == NOW2 - timedelta(days=2)
    assert capped is True


def test_effective_since_pure_function_recent_last_run_is_not_capped():
    last_run = NOW2 - timedelta(hours=1)
    since, capped = _effective_since(last_run, NOW2, 2)
    assert since == last_run
    assert capped is False


def test_effective_since_pure_function_30_day_gap_caps_to_2_days():
    last_run = NOW2 - timedelta(days=30)
    since, capped = _effective_since(last_run, NOW2, 2)
    assert since == NOW2 - timedelta(days=2)
    assert capped is True


def _rss_fixture_with_two_ages(tmp_path, *, near_days_ago: float, far_days_ago: float):
    near = NOW2 - timedelta(days=near_days_ago)
    far = NOW2 - timedelta(days=far_days_ago)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item><title>Within cap</title><link>https://example.com/within</link>
        <pubDate>{near.strftime('%a, %d %b %Y %H:%M:%S GMT')}</pubDate></item>
  <item><title>Outside cap but after last_run_at</title><link>https://example.com/outside</link>
        <pubDate>{far.strftime('%a, %d %b %Y %H:%M:%S GMT')}</pubDate></item>
</channel></rss>
"""
    p = tmp_path / "backfill.xml"
    p.write_text(xml, encoding="utf-8")
    return p


def test_30_day_gap_produces_a_2_day_effective_window_and_is_logged(tmp_path, session, caplog):
    # Item ages straddle the default 2-day cap but both sit inside the
    # 30-day gap since last_run_at -- proving the cap, not `since` alone,
    # is what excludes the older one.
    p = _rss_fixture_with_two_ages(tmp_path, near_days_ago=1, far_days_ago=10)
    session.add(Source(id="rss:cap", plugin="rss", config={"path": str(p)},
                       cadence_minutes=30, enabled=True,
                       last_run_at=NOW2 - timedelta(days=30)))
    session.commit()

    with caplog.at_level(logging.WARNING, logger="feed.stages.collect"):
        res = collect(session, now=NOW2)

    assert res.new_items == 1
    assert [i.title for i in session.query(Item).all()] == ["Within cap"]

    assert any("max_backfill_days" in r.message for r in caplog.records)
    src = session.get(Source, "rss:cap")
    assert src.coverage_warning is not None
    assert "max_backfill_days" in src.coverage_warning


def test_no_backfill_cap_warning_when_gap_is_within_the_cap(tmp_path, session):
    # last_run_at is only an hour old -- well within the default 2-day cap
    # -- so the cap must not narrow `since` and must not warn. Item ages
    # chosen so A3's RSS-truncation heuristic doesn't fire either (one
    # entry predates `since`, so the mixed case stays silent there too).
    p = _rss_fixture_with_two_ages(tmp_path, near_days_ago=0.01, far_days_ago=0.2)
    last_run = NOW2 - timedelta(hours=1)
    session.add(Source(id="rss:nocap", plugin="rss", config={"path": str(p)},
                       cadence_minutes=30, enabled=True, last_run_at=last_run))
    session.commit()

    res = collect(session, now=NOW2)

    src = session.get(Source, "rss:nocap")
    assert src.coverage_warning is None


def test_max_backfill_days_is_configurable_globally(tmp_path, session):
    # Global cap widened to 15 days: the far item (10 days old) now falls
    # inside the window and must be collected, unlike the default-cap test
    # above where it was excluded.
    p = _rss_fixture_with_two_ages(tmp_path, near_days_ago=1, far_days_ago=10)
    session.add(Source(id="rss:wide", plugin="rss", config={"path": str(p)},
                       cadence_minutes=30, enabled=True,
                       last_run_at=NOW2 - timedelta(days=30)))
    session.commit()

    res = collect(session, now=NOW2, cfg=CollectConfig(max_backfill_days=15))

    # The 30-day gap still exceeds even the widened 15-day cap, so this run
    # is still capped (and still warns) -- but capped to a wider window,
    # which is what lets the far item through where the default-cap test
    # above excluded it.
    assert res.new_items == 2
    src = session.get(Source, "rss:wide")
    assert src.coverage_warning is not None
    assert "max_backfill_days=15" in src.coverage_warning


def test_per_source_max_backfill_days_overrides_the_global_default(tmp_path, session):
    # Global default stays at 2 days (cfg omitted below), but this source
    # carries its own wider override -- proving the per-source column, not
    # just the global config value, is honoured.
    p = _rss_fixture_with_two_ages(tmp_path, near_days_ago=1, far_days_ago=10)
    session.add(Source(id="rss:override", plugin="rss", config={"path": str(p)},
                       cadence_minutes=30, enabled=True,
                       last_run_at=NOW2 - timedelta(days=30),
                       max_backfill_days=15))
    session.commit()

    res = collect(session, now=NOW2)  # default CollectConfig() -> global cap is 2 days

    # Same 30-day-gap-still-exceeds-the-cap reasoning as the global-config
    # test above -- the per-source 15-day override still gets capped, but
    # to ITS OWN 15-day window (not the global 2-day default), which is
    # what lets the far item through.
    assert res.new_items == 2
    src = session.get(Source, "rss:override")
    assert src.coverage_warning is not None
    assert "max_backfill_days=15" in src.coverage_warning


def test_first_run_backfill_bound_is_logged_and_recorded(tmp_path, session, caplog):
    p = _rss_fixture_with_two_ages(tmp_path, near_days_ago=1, far_days_ago=10)
    session.add(Source(id="rss:first", plugin="rss", config={"path": str(p)},
                       cadence_minutes=30, enabled=True, last_run_at=None))
    session.commit()

    with caplog.at_level(logging.WARNING, logger="feed.stages.collect"):
        res = collect(session, now=NOW2)

    assert res.new_items == 1  # only the within-cap item; a brand-new source
                                # does not drag in its whole history
    assert any("first run" in r.message for r in caplog.records)
    src = session.get(Source, "rss:first")
    assert src.coverage_warning is not None
    assert "first run" in src.coverage_warning


# --- A1-followup: arXiv 429 handling at the collect() level ---------------
#
# The live sources page showed arxiv:ai Degraded from a single transient
# 429 -- these prove collect() itself now distinguishes "rate-limited, will
# retry" (never touches consecutive_failures) from "genuinely broken"
# (only after retries are exhausted), and that the cross-run politeness
# clock (Source.last_request_at) is actually read back and honoured by a
# LATER, separate collect() call -- the exact scenario ("run the pipeline
# twice in close succession") that tripped the original bug.

ARXIV_FIX = Path(__file__).parent / "fixtures" / "sample_arxiv.xml"
# sample_arxiv.xml's entries are dated well before NOW/the frozen clocks
# below -- a generous backfill cap keeps _effective_since from filtering
# them out, which is irrelevant to what these tests are actually proving
# (429 handling / cross-run pacing, not the A4 backfill cap).
_NO_BACKFILL_CAP = CollectConfig(max_backfill_days=3650)


def _http_429():
    import httpx
    request = httpx.Request("GET", "https://export.arxiv.org/api/query")
    response = httpx.Response(429, request=request)
    return httpx.HTTPStatusError("429 rate limited", request=request, response=response)


def test_collect_does_not_degrade_source_when_429_is_retried_and_succeeds(session, monkeypatch):
    import feed.sources.arxiv as arxiv_module
    calls = {"n": 0}
    xml_bytes = ARXIV_FIX.read_bytes()

    def flaky_get(url, *, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_429()
        return xml_bytes

    monkeypatch.setattr(arxiv_module, "_get", flaky_get)
    _add_source(session, id="arxiv:ai", plugin="arxiv",
               config={"categories": ["cs.AI"]})

    res = collect(session, now=NOW, cfg=_NO_BACKFILL_CAP)

    assert "arxiv:ai" not in res.source_errors
    src = session.get(Source, "arxiv:ai")
    assert src.consecutive_failures == 0
    assert src.last_error is None
    assert res.new_items > 0
    assert calls["n"] == 2  # proves it genuinely retried, not a lucky first call


def test_collect_degrades_source_only_after_429_retries_exhausted(session, monkeypatch):
    import feed.sources.arxiv as arxiv_module

    def always_429(url, *, timeout):
        raise _http_429()

    monkeypatch.setattr(arxiv_module, "_get", always_429)
    _add_source(session, id="arxiv:ai", plugin="arxiv",
               config={"categories": ["cs.AI"], "max_retries": 1, "retry_backoff_base": 0.01})

    res = collect(session, now=NOW, cfg=_NO_BACKFILL_CAP)

    assert "arxiv:ai" in res.source_errors
    src = session.get(Source, "arxiv:ai")
    assert src.consecutive_failures == 1
    assert "429" in src.last_error


def test_collect_persists_and_honours_last_request_at_across_two_separate_runs(session, monkeypatch):
    """The actual A1-followup regression: two separate collect() calls
    (standing in for two separate `feed run` process invocations) close
    together in wall-clock time. The second must pace its opening request
    against the FIRST run's last request, read back from the DB -- an
    in-memory-only timestamp cannot do this, since each collect() call
    here uses a freshly-built ArxivSource with no memory of the first.
    """
    import feed.sources.arxiv as arxiv_module
    from datetime import datetime as real_datetime

    class _FrozenClock(real_datetime):
        current: "real_datetime | None" = None

        @classmethod
        def now(cls, tz=None):
            return cls.current

    monkeypatch.setattr(arxiv_module, "datetime", _FrozenClock)
    sleeps: list[float] = []
    monkeypatch.setattr(arxiv_module, "_sleep", lambda s: sleeps.append(s))
    xml_bytes = ARXIV_FIX.read_bytes()
    monkeypatch.setattr(arxiv_module, "_get", lambda url, *, timeout: xml_bytes)

    _add_source(session, id="arxiv:ai", plugin="arxiv",
               config={"categories": ["cs.AI"], "rate_limit_seconds": 3.0},
               cadence_minutes=0)

    t1 = real_datetime(2026, 8, 22, 12, 0, 0, tzinfo=timezone.utc)
    _FrozenClock.current = t1
    collect(session, now=t1, cfg=_NO_BACKFILL_CAP)
    src = session.get(Source, "arxiv:ai")
    assert src.last_request_at == t1
    assert sleeps == []  # nothing to pace against on the very first request ever

    # A second, independent collect() call 1s later -- well under the 3s
    # politeness delay. Without last_request_at surviving in the DB, this
    # would sleep 0s (a brand-new ArxivSource sees last_request_at=None)
    # and land right on the first run's heels, exactly like the live bug.
    t2 = t1 + timedelta(seconds=1)
    _FrozenClock.current = t2
    collect(session, now=t2, cfg=_NO_BACKFILL_CAP)
    src = session.get(Source, "arxiv:ai")
    assert sleeps == [pytest.approx(2.0)]  # 3.0s owed - 1.0s elapsed
    assert src.last_request_at == t2
