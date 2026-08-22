from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from feed.clustering.adjudicate import (NullAdjudicator, ThresholdAdjudicator, Verdict)
from feed.config import ClusteringConfig
from feed.embedding.base import pack
from feed.models import Item, Source, Stage, Story
from feed.stages.cluster import cluster

NOW = datetime(2026, 8, 19, 12, tzinfo=timezone.utc)


def _vec(*xs):
    return pack(np.array(xs, dtype=np.float32))


def _seed(session, rows):
    session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
    session.add(Source(id="t", plugin="rss", config={}, cadence_minutes=30))
    for i, (src, title, vec, offset) in enumerate(rows):
        session.add(Item(source_id=src, url=f"http://x/{i}", url_hash=f"h{i}",
                         title=title, text=title, embedding=vec,
                         embedding_model_id="fake/v1",
                         published_at=NOW + timedelta(hours=offset),
                         stage=Stage.EMBEDDED))
    session.commit()


def test_threshold_adjudicator_bands():
    adj = ThresholdAdjudicator(merge_threshold=0.50, ambiguous_band=0.06)
    assert adj.decide(0.70, None, None) is Verdict.SAME
    assert adj.decide(0.30, None, None) is Verdict.DIFFERENT
    assert adj.decide(0.52, None, None) is Verdict.AMBIGUOUS


def test_null_adjudicator_never_merges_on_ambiguity():
    adj = NullAdjudicator(ThresholdAdjudicator(0.50, 0.06))
    assert adj.decide(0.52, None, None) is Verdict.DIFFERENT
    assert adj.decide(0.90, None, None) is Verdict.SAME


def test_similar_items_from_two_outlets_become_one_story(session):
    _seed(session, [
        ("s", "DeepSeek releases V4 open weights", _vec(1, 0, 0), 0),
        ("t", "DeepSeek V4 weights published by the lab", _vec(1, 0, 0), 1),
    ])
    res = cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert res.processed == 2
    stories = session.query(Story).all()
    assert len(stories) == 1
    assert stories[0].item_count == 2
    assert stories[0].outlet_count == 2


def test_unrelated_items_stay_separate(session):
    _seed(session, [
        ("s", "DeepSeek releases V4", _vec(1, 0, 0), 0),
        ("t", "Texas grid capacity warning", _vec(0, 1, 0), 1),
    ])
    cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert session.query(Story).count() == 2


def test_items_outside_the_time_window_do_not_merge(session):
    _seed(session, [
        ("s", "DeepSeek releases V4", _vec(1, 0, 0), 0),
        ("t", "DeepSeek releases V4", _vec(1, 0, 0), 200),   # far outside 48h
    ])
    cluster(session, ClusteringConfig(window_hours=48),
            NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert session.query(Story).count() == 2


def test_outlet_count_does_not_double_count_one_source(session):
    _seed(session, [
        ("s", "DeepSeek releases V4", _vec(1, 0, 0), 0),
        ("s", "DeepSeek V4 is out now", _vec(1, 0, 0), 1),
    ])
    cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    story = session.query(Story).one()
    assert story.item_count == 2
    assert story.outlet_count == 1


def test_items_advance_to_clustered(session):
    _seed(session, [("s", "A story", _vec(1, 0, 0), 0)])
    cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert session.query(Item).one().stage is Stage.CLUSTERED


# --- Ruling 1: candidate query must not filter on Stage.CLUSTERED ---------
#
# Task 13 later advances clustered+scored items to Stage.SCORED. If the
# candidate query in cluster() filtered on Item.stage == Stage.CLUSTERED,
# every item that has progressed past clustering would become invisible as
# a clustering candidate on the NEXT pipeline run: a genuinely matching
# second article would never find its story and would spin up a duplicate
# one instead. This test simulates exactly that two-run scenario and would
# fail if the stage predicate were present, because the first item's stage
# is advanced to SCORED before the second (matching) item is clustered.
def test_cross_run_clustering_finds_stories_past_the_clustered_stage(session):
    _seed(session, [
        ("s", "DeepSeek releases V4 open weights", _vec(1, 0, 0), 0),
    ])
    adjudicator = NullAdjudicator(ThresholdAdjudicator(0.5, 0.06))
    cfg = ClusteringConfig()

    first_run = cluster(session, cfg, adjudicator, now=NOW)
    assert first_run.processed == 1
    first_item = session.query(Item).one()
    assert first_item.stage is Stage.CLUSTERED
    story_id = first_item.story_id

    # Simulate Task 13 advancing the already-clustered item further down the
    # pipeline, on what would be a later, separate pipeline run.
    first_item.stage = Stage.SCORED
    session.commit()

    # A second, later-arriving article about the same story shows up.
    session.add(Item(source_id="t", url="http://x/late", url_hash="hlate",
                     title="DeepSeek V4 weights published by the lab",
                     text="DeepSeek V4 weights published by the lab",
                     embedding=_vec(1, 0, 0), embedding_model_id="fake/v1",
                     published_at=NOW + timedelta(hours=1), stage=Stage.EMBEDDED))
    session.commit()

    second_run = cluster(session, cfg, adjudicator, now=NOW + timedelta(hours=1))
    assert second_run.processed == 1

    second_item = session.query(Item).filter(Item.id != first_item.id).one()
    assert second_item.story_id == story_id, (
        "second item should join the existing story even though the first "
        "item has already advanced past Stage.CLUSTERED"
    )
    assert session.query(Story).count() == 1


# --- Ruling 2: model-scoped threshold, not a single global one ------------
#
# cfg.merge_thresholds maps embedding_model_id -> threshold because the two
# supported embedding models produce different similarity scales. cluster()
# must consult cfg.threshold_for(item.embedding_model_id), not a single
# fixed value, or a per-model override in the config has no effect.
def test_high_model_threshold_blocks_a_merge_the_default_would_allow(session):
    # Under the default ClusteringConfig() (merge_threshold=0.50) this exact
    # pair merges into one story -- see
    # test_similar_items_from_two_outlets_become_one_story above. Setting a
    # near-maximum model-specific threshold for "fake/v1" must block that
    # merge, proving cluster() actually consults the per-model value instead
    # of the adjudicator's own fixed 0.50.
    cfg = ClusteringConfig(merge_thresholds={"fake/v1": 1.0})
    _seed(session, [
        ("s", "DeepSeek releases V4 open weights", _vec(1, 0, 0), 0),
        ("t", "DeepSeek V4 weights published by the lab", _vec(1, 0, 0), 1),
    ])
    cluster(session, cfg, NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert session.query(Story).count() == 2


def test_low_model_threshold_allows_a_merge_the_default_would_block(session):
    # cos([1,0,0],[0.6,0.8,0]) = 0.6, entity overlap is 0 (no shared
    # capitalised/versioned tokens), so the blended score is well under the
    # default 0.50 threshold and these two items stay separate under the
    # default config. A very low model-specific threshold for "fake/v1"
    # must let them merge, proving cluster() lowers the bar, not just
    # raises it.
    cfg = ClusteringConfig(merge_thresholds={"fake/v1": 0.05})
    _seed(session, [
        ("s", "alpha item one", _vec(1, 0, 0), 0),
        ("t", "alpha item two", _vec(0.6, 0.8, 0), 1),
    ])
    cluster(session, cfg, NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert session.query(Story).count() == 1


# --- Risk point: centroid is the mean of member vectors, recomputed on ----
# every membership change, not just set once at story creation.
def test_story_centroid_is_recomputed_as_the_mean_of_member_vectors(session):
    from feed.embedding.base import unpack

    _seed(session, [
        ("s", "DeepSeek releases V4 open weights", _vec(1, 0, 0), 0),
        ("t", "DeepSeek V4 weights published by the lab", _vec(0.8, 0.6, 0), 1),
    ])
    cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    story = session.query(Story).one()
    assert story.item_count == 2  # sanity: confirms these two actually merged
    centroid = unpack(story.centroid)
    # Mean of (1,0,0) and (0.8,0.6,0) is (0.9, 0.3, 0).
    assert centroid == pytest.approx([0.9, 0.3, 0.0])


# --- Risk point: story.updated_at tracks the newest member's published_at,
# not wall-clock "now" -----------------------------------------------------
def test_story_updated_at_reflects_newest_member_published_at_not_now(session):
    _seed(session, [
        ("s", "DeepSeek releases V4 open weights", _vec(1, 0, 0), 0),
        ("t", "DeepSeek V4 weights published by the lab", _vec(1, 0, 0), 5),
    ])
    cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    story = session.query(Story).one()
    assert story.updated_at == NOW + timedelta(hours=5)
    assert story.updated_at != NOW


# --- Ruling 3: corrupt (non-finite) embeddings fail visibly ---------------
#
# feed.clustering.signals.cosine() returns 0.0 for a non-finite vector
# rather than raising, so a corrupt embedding would otherwise cluster
# silently and permanently wrong (or just never match) with nothing
# recorded. The cluster stage must validate finiteness itself and mark the
# item FAILED with a clear reason, per spec success criterion 1 ("no silent
# coverage loss").
def test_non_finite_embedding_is_marked_failed_not_silently_dropped(session):
    bad = pack(np.array([np.nan, 0.0, 0.0], dtype=np.float32))
    _seed(session, [("s", "Corrupted embedding item", bad, 0)])
    res = cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert res.failed == 1
    assert res.processed == 0
    item = session.query(Item).one()
    assert item.stage is Stage.FAILED
    assert item.error is not None
    assert "finite" in item.error.lower() or "nan" in item.error.lower()
    assert session.query(Story).count() == 0


def test_non_finite_embedding_does_not_stall_the_rest_of_the_batch(session):
    bad = pack(np.array([np.inf, 0.0, 0.0], dtype=np.float32))
    _seed(session, [
        ("s", "Corrupted embedding item", bad, 0),
        ("t", "A perfectly fine second item", _vec(0, 1, 0), 0),
    ])
    res = cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert res.failed == 1
    assert res.processed == 1
    stages = {i.title: i.stage for i in session.query(Item).all()}
    assert stages["Corrupted embedding item"] is Stage.FAILED
    assert stages["A perfectly fine second item"] is Stage.CLUSTERED
