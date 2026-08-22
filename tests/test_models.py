from datetime import datetime, timezone
from feed.models import Source, Item, Story, Stage, StoryStatus

def test_item_starts_in_collected_stage(session):
    src = Source(id="rss:example", plugin="rss", config={"url": "http://x"}, cadence_minutes=30)
    session.add(src)
    item = Item(source_id="rss:example", url="http://x/1", url_hash="h1", title="T")
    session.add(item)
    session.commit()
    assert item.stage is Stage.COLLECTED
    assert item.error is None

def test_url_hash_is_unique(session):
    session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
    session.add(Item(source_id="s", url="http://a", url_hash="dup", title="A"))
    session.commit()
    session.add(Item(source_id="s", url="http://b", url_hash="dup", title="B"))
    import pytest, sqlalchemy.exc
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        session.commit()

def test_story_tracks_item_count_and_updated_at(session):
    st = Story(title="S", first_seen=datetime.now(timezone.utc),
               updated_at=datetime.now(timezone.utc), item_count=0)
    session.add(st)
    session.commit()
    assert st.id is not None
    assert st.score is None

def test_story_defaults_to_new_status_with_no_enrichment(session):
    st = Story(title="S", first_seen=datetime.now(timezone.utc),
               updated_at=datetime.now(timezone.utc), item_count=0)
    session.add(st)
    session.commit()
    session.expire_all()
    reloaded = session.get(Story, st.id)
    assert reloaded.status is StoryStatus.NEW
    assert reloaded.summary is None
    assert reloaded.category is None
    assert reloaded.analysis is None
    assert reloaded.analysis_provider is None
    assert reloaded.analyzed_at is None

def test_story_enrichment_fields_round_trip(session):
    now = datetime.now(timezone.utc)
    st = Story(title="Canonical headline", first_seen=now, updated_at=now,
               item_count=1, summary="Two sentence summary.", category="research",
               analysis="Why this matters.", analysis_provider="claude-code:claude-code",
               analyzed_at=now, status=StoryStatus.ANALYZED)
    session.add(st)
    session.commit()
    session.expire_all()
    reloaded = session.get(Story, st.id)
    assert reloaded.summary == "Two sentence summary."
    assert reloaded.category == "research"
    assert reloaded.analysis == "Why this matters."
    assert reloaded.analysis_provider == "claude-code:claude-code"
    assert reloaded.analyzed_at == now
    assert reloaded.status is StoryStatus.ANALYZED

def test_source_last_run_at_round_trips_as_aware_utc(session):
    written = datetime(2026, 8, 18, 8, 0, tzinfo=timezone.utc)
    session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30,
                        last_run_at=written))
    session.commit()
    session.expire_all()  # force a genuine reload from sqlite, not the identity map
    reloaded = session.get(Source, "s")
    assert reloaded.last_run_at.tzinfo is not None
    assert reloaded.last_run_at == written

def test_item_published_at_round_trips_as_aware_utc(session):
    written = datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)
    session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
    item = Item(source_id="s", url="http://x/1", url_hash="h1", title="T",
                published_at=written)
    session.add(item)
    session.commit()
    item_id = item.id
    session.expire_all()  # force a genuine reload from sqlite, not the identity map
    reloaded = session.get(Item, item_id)
    assert reloaded.published_at.tzinfo is not None
    assert reloaded.published_at == written
