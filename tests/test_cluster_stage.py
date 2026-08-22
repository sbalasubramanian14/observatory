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
        # offset=None means "undated" -- published_at stays NULL, exactly
        # like an RSS entry with no pubDate, an HN record with no `time`,
        # or a GitHub release with no updated/published timestamp.
        published = NOW + timedelta(hours=offset) if offset is not None else None
        session.add(Item(source_id=src, url=f"http://x/{i}", url_hash=f"h{i}",
                         title=title, text=title, embedding=vec,
                         embedding_model_id="fake/v1",
                         published_at=published,
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


# --- Ruling 2 (post-review redesign): model-scoped threshold lives inside
# the adjudicator, not in cluster() -----------------------------------------
#
# cfg.merge_thresholds maps embedding_model_id -> threshold because the two
# supported embedding models produce different similarity scales.
# ThresholdAdjudicator's `threshold_for` parameter is how a caller wires
# ClusteringConfig.threshold_for into the decision. cluster() itself passes
# the RAW pair_score straight through to adjudicator.decide() -- it has no
# threshold logic of its own -- so these tests wire `threshold_for` onto the
# adjudicator at construction time, exactly as real pipeline wiring would.
def test_high_model_threshold_blocks_a_merge_the_default_would_allow(session):
    # Under the default ClusteringConfig() (merge_threshold=0.50) this exact
    # pair merges into one story -- see
    # test_similar_items_from_two_outlets_become_one_story above. Setting a
    # near-maximum model-specific threshold for "fake/v1" must block that
    # merge, proving the adjudicator actually consults the per-model value
    # instead of its own fixed 0.50.
    cfg = ClusteringConfig(merge_thresholds={"fake/v1": 1.0})
    adjudicator = NullAdjudicator(
        ThresholdAdjudicator(cfg.merge_threshold, 0.06, threshold_for=cfg.threshold_for)
    )
    _seed(session, [
        ("s", "DeepSeek releases V4 open weights", _vec(1, 0, 0), 0),
        ("t", "DeepSeek V4 weights published by the lab", _vec(1, 0, 0), 1),
    ])
    cluster(session, cfg, adjudicator, now=NOW)
    assert session.query(Story).count() == 2


def test_low_model_threshold_allows_a_merge_the_default_would_block(session):
    # cos([1,0,0],[0.6,0.8,0]) = 0.6, entity overlap is 0 (no shared
    # capitalised/versioned tokens), so the blended score is well under the
    # default 0.50 threshold and these two items stay separate under the
    # default config. A very low model-specific threshold for "fake/v1"
    # must let them merge, proving the wiring lowers the bar, not just
    # raises it.
    cfg = ClusteringConfig(merge_thresholds={"fake/v1": 0.05})
    adjudicator = NullAdjudicator(
        ThresholdAdjudicator(cfg.merge_threshold, 0.06, threshold_for=cfg.threshold_for)
    )
    _seed(session, [
        ("s", "alpha item one", _vec(1, 0, 0), 0),
        ("t", "alpha item two", _vec(0.6, 0.8, 0), 1),
    ])
    cluster(session, cfg, adjudicator, now=NOW)
    assert session.query(Story).count() == 1


def test_threshold_for_callable_is_actually_consulted_when_given():
    # Direct unit-level pin of the coupling (Finding 3): construct the
    # adjudicator with a threshold_for that returns a very high value for
    # one model id and confirm decide() uses it instead of merge_threshold,
    # via a real Item-shaped `left` (a plain namespace with the attribute
    # ThresholdAdjudicator reads) rather than going through cluster().
    from types import SimpleNamespace

    def threshold_for(model_id):
        return 0.95 if model_id == "strict-model" else 0.50

    adj = ThresholdAdjudicator(merge_threshold=0.50, ambiguous_band=0.06,
                               threshold_for=threshold_for)
    left = SimpleNamespace(embedding_model_id="strict-model")
    other_left = SimpleNamespace(embedding_model_id="normal-model")

    # 0.80 clears the default band's high edge (0.53) but not 0.95's.
    assert adj.decide(0.80, left, None) is Verdict.DIFFERENT
    assert adj.decide(0.80, other_left, None) is Verdict.SAME
    # No `left` at all -> falls back to merge_threshold, matching the
    # brief's own unit tests (decide(0.70, None, None) is SAME).
    assert adj.decide(0.80, None, None) is Verdict.SAME


# --- Finding 2: candidates must never cross embedding models --------------
#
# feed/models.py stores embedding_model_id per item because vectors from
# different models are not comparable (spec S3.3). The candidate query in
# cluster() filters on Item.embedding_model_id == item.embedding_model_id;
# deleting that filter left all 14 original tests green, so it was entirely
# unguarded. Two items with identical titles and identical raw vector bytes
# but different embedding_model_id values must NOT be treated as candidates
# for each other.
def test_candidates_never_cross_embedding_models(session):
    session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
    session.add(Source(id="t", plugin="rss", config={}, cadence_minutes=30))
    session.add(Item(source_id="s", url="http://x/0", url_hash="h0",
                     title="DeepSeek releases V4", text="DeepSeek releases V4",
                     embedding=_vec(1, 0, 0), embedding_model_id="model-a",
                     published_at=NOW, stage=Stage.EMBEDDED))
    session.add(Item(source_id="t", url="http://x/1", url_hash="h1",
                     title="DeepSeek releases V4", text="DeepSeek releases V4",
                     embedding=_vec(1, 0, 0), embedding_model_id="model-b",
                     published_at=NOW + timedelta(hours=1), stage=Stage.EMBEDDED))
    session.commit()
    cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert session.query(Story).count() == 2


# --- Validation of the redesign: AMBIGUOUS resolves to DIFFERENT through --
# the real cluster() call path, not just the isolated adjudicator unit test.
#
# The score reaching decide() is now the raw pair_score (no shift), so this
# reconfirms, through the actual stage rather than a synthetic float, that a
# pair whose real blended similarity lands inside the ambiguous band still
# ends up split when adjudicated by NullAdjudicator.
class _AlwaysSame:
    """Test double proving the pair below is a genuine clustering candidate
    (passes the time window and embedding-model filters) and would merge if
    the adjudicator allowed it -- i.e. the split in the test below is caused
    specifically by the AMBIGUOUS-to-DIFFERENT resolution, not by the pair
    failing to become a candidate for some unrelated reason."""

    def decide(self, pair_score, left=None, right=None):
        return Verdict.SAME


def test_ambiguous_pair_resolves_to_different_through_the_real_stage(session):
    # "bravo item one" / "bravo item two" share no capitalised/versioned
    # entities, so the blend is pure cosine * time_proximity. Vector chosen
    # (confirmed empirically -- see task-11-report.md) so raw pair_score is
    # ~0.4994, inside the default [0.47, 0.53) ambiguous band with margin
    # from both edges.
    _seed(session, [
        ("s", "bravo item one", _vec(1, 0, 0), 0),
        ("t", "bravo item two", _vec(0.85, 0.5268, 0), 1),
    ])
    cluster(session, ClusteringConfig(), NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert session.query(Story).count() == 2, (
        "an ambiguous-scoring pair must split, not merge, under NullAdjudicator"
    )


def test_ambiguous_pair_would_have_merged_under_a_permissive_adjudicator(session):
    # Non-vacuity check for the test above: the identical scenario, scored
    # by an adjudicator that always says SAME, merges into one story. This
    # confirms the pair is a real candidate (correct model id, inside the
    # time window) and that NullAdjudicator's split above is caused by the
    # ambiguous-to-DIFFERENT resolution, not by the pair never reaching
    # adjudication at all.
    _seed(session, [
        ("s", "bravo item one", _vec(1, 0, 0), 0),
        ("t", "bravo item two", _vec(0.85, 0.5268, 0), 1),
    ])
    cluster(session, ClusteringConfig(), _AlwaysSame(), now=NOW)
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


# --- C1: undated items must still be able to cluster -----------------------
#
# cluster()'s candidate query filtered on `Item.published_at >= cutoff`. In
# SQL, comparing NULL against anything (including `>=`) evaluates to NULL,
# which is falsy in a WHERE clause -- so an item with published_at IS NULL
# was silently excluded from the candidate set no matter how similar it was
# to an already-clustered item, AND, once its own turn came around as
# `item` (candidates are queried freshly per item, not filtered on the
# outer item's date), any dated item already in a story was invisible to
# it too, because the predicate is on the CANDIDATE row regardless of which
# item is being matched. Two items that both lack a date could therefore
# never merge with anything, forever, even at blended similarity 0.978.
#
# The in-Python guard just below in cluster() (`if item.published_at and
# other.published_at: ... apply time_proximity ...`) already tolerates an
# undated pair by skipping the decay multiplier entirely -- i.e. the
# intended semantics are "an item with no date is never penalised or
# excluded on time grounds, because we have no time information to filter
# on". The SQL predicate silently overrode that intent for the candidate
# set itself. These tests seed items with offset=None (published_at IS
# NULL) and prove they can still become candidates and merge.
def test_two_undated_items_still_merge(session):
    _seed(session, [
        ("s", "DeepSeek releases V4 open weights", _vec(1, 0, 0), None),
        ("t", "DeepSeek V4 weights published by the lab", _vec(1, 0, 0), None),
    ])
    res = cluster(session, ClusteringConfig(),
                  NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert res.processed == 2
    stories = session.query(Story).all()
    assert len(stories) == 1
    assert stories[0].item_count == 2


def test_an_undated_item_still_clusters_with_a_dated_item(session):
    # Order matters for reproducing the actual bug: the UNDATED item is
    # seeded (and therefore clustered) FIRST, so it is already a Story
    # member -- and therefore a CANDIDATE row with published_at IS NULL --
    # by the time the dated item is processed and queries for candidates.
    # `Item.published_at >= cutoff` on that candidate row is SQL NULL,
    # which is falsy, so the buggy query drops it silently.
    _seed(session, [
        ("s", "DeepSeek V4 weights published by the lab", _vec(1, 0, 0), None),
        ("t", "DeepSeek releases V4 open weights", _vec(1, 0, 0), 0),
    ])
    res = cluster(session, ClusteringConfig(),
                  NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert res.processed == 2
    stories = session.query(Story).all()
    assert len(stories) == 1
    assert stories[0].item_count == 2


def test_undated_items_do_not_widen_the_candidate_set_for_dated_items(session):
    # A dated item well outside the time window must still be excluded from
    # a dated item's candidates, even though undated candidates now bypass
    # the cutoff -- the fix must not turn off time filtering altogether.
    _seed(session, [
        ("s", "DeepSeek releases V4", _vec(1, 0, 0), 0),
        ("t", "DeepSeek releases V4", _vec(1, 0, 0), 200),   # far outside 48h
    ])
    cluster(session, ClusteringConfig(window_hours=48),
            NullAdjudicator(ThresholdAdjudicator(0.5, 0.06)), now=NOW)
    assert session.query(Story).count() == 2


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
