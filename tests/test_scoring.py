from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from feed.config import ScoringConfig
from feed.embedding.base import pack
from feed.models import Entity, Item, Source, Stage, Story, StoryEntity
from feed.scoring.combine import combine
from feed.scoring.signals import authority, entity_weight, novelty, velocity
from feed.stages.score import score_stories

NOW = datetime(2026, 8, 19, 12, tzinfo=timezone.utc)


def _story(session, *, outlets, sources_authority=0.5, vec=(1.0, 0.0), age_h=1):
    for i, name in enumerate(outlets):
        if session.get(Source, name) is None:
            auth = sources_authority(name) if callable(sources_authority) else sources_authority
            session.add(Source(id=name, plugin="rss", config={},
                               cadence_minutes=30, authority=auth))
    st = Story(title="S", first_seen=NOW - timedelta(hours=age_h),
               updated_at=NOW - timedelta(hours=age_h),
               item_count=len(outlets), outlet_count=len(set(outlets)),
               centroid=pack(np.array(vec, dtype=np.float32)))
    session.add(st)
    session.flush()
    for i, name in enumerate(outlets):
        session.add(Item(source_id=name, url=f"http://x/{name}/{i}",
                         url_hash=f"{name}{i}", title="t", text="t",
                         embedding=pack(np.array(vec, dtype=np.float32)),
                         embedding_model_id="fake/v1", story_id=st.id,
                         published_at=NOW - timedelta(hours=age_h),
                         stage=Stage.CLUSTERED))
    session.commit()
    return st


def test_velocity_rises_with_independent_outlets(session):
    one = _story(session, outlets=["a"])
    many = _story(session, outlets=["b", "c", "d", "e", "f"], vec=(0.0, 1.0))
    assert velocity(many) > velocity(one)


def test_velocity_counts_outlets_not_articles(session):
    syndicated = _story(session, outlets=["a", "a", "a", "a"])
    genuine = _story(session, outlets=["b", "c", "d", "e"], vec=(0.0, 1.0))
    assert velocity(genuine) > velocity(syndicated)


def test_velocity_uses_story_outlet_count_field(session):
    # A story whose outlet_count field disagrees with its item outlets must
    # still be driven by outlet_count -- that column is the single source of
    # truth the cluster stage maintains, and velocity() takes a bare Story
    # (no session), so it cannot recompute outlets from items itself.
    st = _story(session, outlets=["a", "b", "c"])
    st.outlet_count = 1
    assert velocity(st) == pytest.approx(min(1.0, np.log1p(1) / np.log1p(10)))


def test_authority_averages_contributing_sources(session):
    st = _story(session, outlets=["hi"], sources_authority=0.9)
    assert authority(session, st) == pytest.approx(0.9)


def test_authority_averages_distinct_sources_not_items(session):
    """A prolific low-authority source filing many items must not drag the
    authority signal down by sheer item count -- authority is a per-SOURCE
    weight (spec 3.6 #1: "static per-source weight"), and item-count
    weighting would let the same syndication dynamic that manufactures fake
    velocity also distort authority, just via volume from one outlet instead
    of many.
    """
    weights = {"hi": 0.9, "lo": 0.1}
    st = _story(session, outlets=["lo", "lo", "lo", "hi"], sources_authority=weights.get)
    # Per-item average would be (0.1*3 + 0.9) / 4 = 0.3. Per-distinct-source
    # average is (0.1 + 0.9) / 2 = 0.5.
    assert authority(session, st) == pytest.approx(0.5)


def test_novelty_is_low_for_a_near_duplicate_of_an_older_story(session):
    _story(session, outlets=["a"], vec=(1.0, 0.0), age_h=200)
    follow_up = _story(session, outlets=["b"], vec=(1.0, 0.0), age_h=1)
    fresh = _story(session, outlets=["c"], vec=(0.0, 1.0), age_h=1)
    assert novelty(session, follow_up) < novelty(session, fresh)


def test_novelty_is_maximal_with_no_prior_stories(session):
    solo = _story(session, outlets=["a"], vec=(1.0, 0.0), age_h=1)
    assert novelty(session, solo) == 1.0


def test_novelty_is_not_suppressed_by_a_later_near_duplicate(session):
    # A story must not be punished by a near-duplicate that shows up AFTER
    # it -- only strictly-earlier stories count as prior art.
    early = _story(session, outlets=["a"], vec=(1.0, 0.0), age_h=100)
    _story(session, outlets=["b"], vec=(1.0, 0.0), age_h=1)  # later near-dup
    assert novelty(session, early) == 1.0


def test_novelty_handles_none_centroid_without_crashing(session):
    st = Story(title="no vec", first_seen=NOW, updated_at=NOW,
               item_count=0, outlet_count=0, centroid=None)
    session.add(st)
    session.commit()
    assert novelty(session, st) == 1.0


def test_entity_weight_defaults_to_half_with_no_entities(session):
    st = _story(session, outlets=["a"])
    assert entity_weight(session, st) == pytest.approx(0.5)


def test_entity_weight_uses_max_linked_entity(session):
    st = _story(session, outlets=["a"])
    e_low = Entity(name="minor-co", weight=0.3)
    e_high = Entity(name="major-co", weight=0.8)
    session.add_all([e_low, e_high])
    session.flush()
    session.add_all([
        StoryEntity(story_id=st.id, entity_id=e_low.id),
        StoryEntity(story_id=st.id, entity_id=e_high.id),
    ])
    session.commit()
    assert entity_weight(session, st) == pytest.approx(0.8)


def test_combine_is_a_weighted_sum_clamped_to_unit_range():
    parts = {"authority": 1.0, "velocity": 1.0, "novelty": 1.0, "entity": 1.0}
    weights = {"authority": 0.25, "velocity": 0.4, "novelty": 0.2, "entity": 0.15}
    assert combine(parts, weights) == pytest.approx(1.0)
    assert combine({"authority": 0.0}, {"authority": 1.0}) == 0.0


def test_combine_ignores_weights_with_no_matching_part():
    assert combine({"velocity": 1.0}, {"velocity": 0.5, "missing": 0.5}) == pytest.approx(0.5)


def test_combine_ignores_parts_with_no_matching_weight():
    # Symmetric to the above: an extra key in `parts` that the weights dict
    # says nothing about must not silently participate.
    assert combine({"velocity": 1.0, "mystery": 0.0}, {"velocity": 1.0}) == pytest.approx(1.0)


def test_combine_stays_in_unit_range_for_arbitrary_present_signals():
    parts = {"authority": 0.2, "velocity": 0.9, "novelty": 0.4}
    weights = {"authority": 0.25, "velocity": 0.4, "novelty": 0.2, "entity": 0.15}
    result = combine(parts, weights)
    assert 0.0 <= result <= 1.0


def test_score_stories_persists_score_and_breakdown(session):
    _story(session, outlets=["a", "b", "c"])
    res = score_stories(session, ScoringConfig(), now=NOW)
    assert res.processed == 1
    st = session.query(Story).one()
    assert st.score is not None and 0.0 <= st.score <= 1.0
    assert set(st.score_breakdown) == {"authority", "velocity", "novelty", "entity"}


def test_score_stories_advances_items_clustered_to_scored(session):
    st = _story(session, outlets=["a", "b"])
    score_stories(session, ScoringConfig(), now=NOW)
    session.refresh(st)
    for item in st.items:
        assert item.stage == Stage.SCORED


def test_score_stories_does_not_touch_already_scored_items(session):
    # Task 11's candidate query deliberately has no stage filter, precisely
    # so already-scored items stay clusterable on later runs; the score
    # stage must not reach back and disturb items past its own claim stage.
    st = _story(session, outlets=["a"])
    for item in st.items:
        item.stage = Stage.SCORED
    session.commit()

    res = score_stories(session, ScoringConfig(), now=NOW)

    assert res.processed == 0
    session.refresh(st)
    for item in st.items:
        assert item.stage == Stage.SCORED
    assert st.score is None


def test_score_stories_isolates_a_bad_story_from_the_rest_of_the_batch(session):
    # An older prior story is required so `bad`'s novelty() computation
    # actually reaches unpack(story.centroid) -- with no prior story,
    # novelty() short-circuits to 1.0 before ever touching the corrupt
    # bytes, and the intended failure would never trigger.
    prior = _story(session, outlets=["z"], vec=(0.0, 1.0), age_h=50)
    good = _story(session, outlets=["a", "b"], age_h=1)
    bad = _story(session, outlets=["c"], vec=(0.0, 1.0), age_h=1)
    # Corrupt the bad story's centroid: a byte length not a multiple of 4
    # makes np.frombuffer raise inside novelty()'s unpack() call.
    bad.centroid = b"\x00\x00\x00"
    session.commit()

    res = score_stories(session, ScoringConfig(), now=NOW)

    assert res.processed == 2
    assert res.failed == 1
    session.refresh(prior)
    assert prior.score is not None
    session.refresh(good)
    assert good.score is not None
    session.refresh(bad)
    assert bad.score is None


def test_achievable_score_range_is_zero_to_one_with_entity_zero_weighted(session):
    """With entity's configured weight at 0.0 (see feed/config.py), the
    achievable score range must be the full [0, 1], not the [0.075, 0.925]
    a nonzero constant entity signal would compress it to. A story with
    every real signal at its minimum must score exactly 0.0, and one with
    every real signal at its maximum must score exactly 1.0 -- entity_weight()
    still runs (and still returns its 0.5 default, since no Entity rows
    exist), but at weight 0.0 it must not move the result at all.
    """
    weights = ScoringConfig().weights
    assert weights["entity"] == 0.0

    prior = _story(session, outlets=["old"], vec=(1.0, 0.0), age_h=100)

    # Minimum: zero-authority source, zero outlets (forced), exact
    # near-duplicate of the strictly-earlier `prior` story.
    lo = _story(session, outlets=["z"], sources_authority=0.0, vec=(1.0, 0.0), age_h=1)
    lo.outlet_count = 0
    session.commit()

    # Maximum: full-authority sources, >=10 distinct outlets (saturates the
    # log1p velocity cap at exactly 1.0), orthogonal vector so novelty is
    # unsuppressed by anything earlier.
    hi = _story(session, outlets=[f"o{i}" for i in range(10)],
                sources_authority=1.0, vec=(0.0, 1.0), age_h=1)

    def _score(story):
        parts = {
            "authority": authority(session, story),
            "velocity": velocity(story),
            "novelty": novelty(session, story),
            "entity": entity_weight(session, story),
        }
        return combine(parts, weights)

    assert _score(lo) == pytest.approx(0.0)
    assert _score(hi) == pytest.approx(1.0)
