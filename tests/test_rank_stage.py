"""Top 50: Claude Code judges importance, the arithmetic score only nominates.

The published `score` (authority + velocity + novelty) is deliberately
reader-independent and knows nothing about meaning -- it cannot tell a
frontier model release from a well-syndicated funding round. This stage
asks the DEEP provider to read the shortlist and place each story in an
importance band, so "Top 50" reflects judgement rather than arithmetic.
"""
from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone

import pytest

from feed.config import ProvidersConfig
from feed.models import Item, Source, Story, StoryStatus
from feed.providers.base import ProviderError, Tier
from feed.providers.router import Router
from feed.stages.rank import BANDS, rank_top


class _FakeProvider:
    def __init__(self, name, model, tier, *, responses=None, healthy=True):
        self.name = name
        self.model = model
        self.tier = tier
        self._responses = list(responses or [])
        self._healthy = healthy
        self.prompts: list[str] = []

    def complete(self, prompt, *, schema=None):
        self.prompts.append(prompt)
        if not self._responses:
            raise ProviderError(f"{self.name}: no canned response left")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def health(self):
        from feed.providers.base import ProviderHealth
        return ProviderHealth(healthy=self._healthy)


def _seed(session, n, *, base_score=0.9, age_days=0.5):
    """n scored stories, descending arithmetic score, all inside the window."""
    now = datetime.now(timezone.utc)
    if session.get(Source, "s") is None:
        session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
        session.flush()
    ids = []
    for i in range(n):
        when = now - timedelta(days=age_days)
        st = Story(title=f"Story {i}", first_seen=when, updated_at=when, item_count=1,
                   outlet_count=1, score=base_score - i * 0.01,
                   summary=f"Summary {i}.", status=StoryStatus.ENRICHED)
        session.add(st)
        session.flush()
        session.add(Item(source_id="s", url=f"http://x/{st.id}", url_hash=f"h{st.id}",
                         title=f"Story {i}", story_id=st.id, published_at=when))
        ids.append(st.id)
    session.commit()
    return ids


def _verdict(ids, bands=None):
    bands = bands or ["landmark"] + ["significant"] * (len(ids) - 1)
    return json.dumps({"ranked": [
        {"id": sid, "rank": i + 1, "band": bands[i], "reason": f"Reason {i}."}
        for i, sid in enumerate(ids)
    ]})


def _router(*responses, healthy=True):
    return Router(bulk=_FakeProvider("groq", "m", Tier.BULK),
                  deep=_FakeProvider("claude-code", "claude-code", Tier.DEEP,
                                     responses=list(responses), healthy=healthy))


def test_rank_writes_rank_band_and_reason_from_the_deep_provider(session):
    ids = _seed(session, 3)
    res = rank_top(session, _router(_verdict(ids)), ProvidersConfig(), top_n=3)

    assert res.ranked == 3
    rows = session.query(Story).order_by(Story.importance_rank).all()
    assert [r.importance_rank for r in rows] == [1, 2, 3]
    assert [r.importance_band for r in rows] == ["landmark", "significant", "significant"]
    assert rows[0].importance_reason == "Reason 0."
    assert rows[0].ranked_by == "claude-code:claude-code"
    assert rows[0].ranked_at is not None


def test_rank_overrides_the_arithmetic_order(session):
    """The whole point: Claude Code may disagree with score. If the stage
    simply re-published score order, it would be an expensive no-op."""
    ids = _seed(session, 3)
    reversed_ids = list(reversed(ids))
    rank_top(session, _router(_verdict(reversed_ids)), ProvidersConfig(), top_n=3)

    by_rank = session.query(Story).order_by(Story.importance_rank).all()
    assert [s.id for s in by_rank] == reversed_ids
    # ... and that really is the opposite of score order.
    by_score = session.query(Story).order_by(Story.score.desc()).all()
    assert [s.id for s in by_score] == ids


def test_rank_only_considers_stories_inside_the_window(session):
    """Top 50 is the top of the FEED, not of all history -- an old story
    must not occupy a slot a reader can no longer see."""
    fresh = _seed(session, 2, age_days=0.5)
    stale = _seed(session, 1, base_score=0.99, age_days=40)  # highest score, too old

    rank_top(session, _router(_verdict(fresh)), ProvidersConfig(),
             top_n=50, window_days=5)

    ranked = {s.id for s in session.query(Story).filter(Story.importance_rank.is_not(None))}
    assert ranked == set(fresh)
    assert session.get(Story, stale[0]).importance_rank is None


def test_rank_clears_previous_ranks_so_the_list_never_grows_stale(session):
    """Yesterday's number 3 must not stay number 3 forever once it falls
    out of the window -- otherwise Top 50 accumulates instead of rotating."""
    ids = _seed(session, 3)
    rank_top(session, _router(_verdict(ids)), ProvidersConfig(), top_n=3)
    assert session.get(Story, ids[2]).importance_rank == 3

    rank_top(session, _router(_verdict(ids[:2])), ProvidersConfig(), top_n=3)

    assert session.get(Story, ids[2]).importance_rank is None
    assert session.get(Story, ids[2]).importance_band is None


def test_rank_rejects_an_unknown_band(session):
    """A hallucinated band would render as an empty group in the UI. Fail
    the story rather than write a category the client cannot display."""
    ids = _seed(session, 2)
    bad = json.dumps({"ranked": [
        {"id": ids[0], "rank": 1, "band": "earth-shattering", "reason": "x"},
        {"id": ids[1], "rank": 2, "band": "notable", "reason": "y"},
    ]})
    res = rank_top(session, _router(bad), ProvidersConfig(), top_n=2)

    assert session.get(Story, ids[0]).importance_band is None
    assert session.get(Story, ids[1]).importance_band == "notable"
    assert res.rejected == 1


def test_rank_ignores_ids_that_were_not_on_the_shortlist(session):
    """Guards against the model inventing an id, which would otherwise
    write a rank onto an unrelated (possibly out-of-window) story."""
    ids = _seed(session, 2)
    smuggled = json.dumps({"ranked": [
        {"id": ids[0], "rank": 1, "band": "landmark", "reason": "x"},
        {"id": 999999, "rank": 2, "band": "landmark", "reason": "not a real id"},
    ]})
    res = rank_top(session, _router(smuggled), ProvidersConfig(), top_n=2)

    assert res.ranked == 1
    assert res.rejected == 1
    assert session.get(Story, ids[1]).importance_rank is None


def test_rank_does_nothing_when_there_are_no_stories(session):
    res = rank_top(session, _router(), ProvidersConfig(), top_n=50)
    assert res.ranked == 0
    assert res.error is None


def test_rank_leaves_existing_ranks_alone_when_the_provider_fails(session):
    """A failed ranking run must not blank the Top 50 the site is already
    serving -- stale judgement beats an empty page."""
    ids = _seed(session, 2)
    rank_top(session, _router(_verdict(ids)), ProvidersConfig(), top_n=2)

    res = rank_top(session, _router(ProviderError("claude code unavailable")),
                   ProvidersConfig(), top_n=2)

    assert res.error is not None
    assert res.ranked == 0
    assert session.get(Story, ids[0]).importance_rank == 1


def test_bands_are_ordered_most_important_first():
    """The client renders groups in this order; a reshuffle here would
    silently reorder the page."""
    assert BANDS == ("landmark", "significant", "notable")
