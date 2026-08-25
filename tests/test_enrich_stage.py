import json
from datetime import datetime, timedelta, timezone
import pytest
from feed.config import ProvidersConfig
from feed.models import Item, Source, Story, StoryStatus
from feed.providers.base import ProviderError, ProviderHealth, Tier
from feed.providers.router import Router
from feed.stages.enrich import enrich, enrich_tier1, enrich_tier2


class _FakeProvider:
    """Records prompts, replays canned responses (or raises) in order."""

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
        return ProviderHealth(healthy=self._healthy)


def _seed_story(session, *, score=0.8, status=StoryStatus.NEW, titles=("A", "B"),
                analyzed_at=None) -> Story:
    now = datetime.now(timezone.utc)
    if session.get(Source, "s") is None:
        session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
        session.flush()
    story = Story(title=titles[0], first_seen=now, updated_at=now, item_count=len(titles),
                 score=score, status=status, analyzed_at=analyzed_at)
    session.add(story)
    session.flush()
    for i, t in enumerate(titles):
        session.add(Item(source_id="s", url=f"http://x/{i}", url_hash=f"h{i}-{story.id}",
                         title=t, story_id=story.id))
    session.commit()
    return story


TIER1_JSON = json.dumps({
    "headline": "Canonical headline",
    "summary": "Two sentence summary.",
    "category": "research",
})


# --- Tier 1 ---------------------------------------------------------------

def test_tier1_enriches_a_new_scored_story(session):
    story = _seed_story(session)
    bulk = _FakeProvider("gemini", "gemini-flash-latest", Tier.BULK, responses=[TIER1_JSON])
    router = Router(bulk=bulk)

    result = enrich_tier1(session, router, ProvidersConfig())

    assert result.tier1_processed == 1
    assert result.tier1_failed == 0
    session.refresh(story)
    assert story.title == "Canonical headline"
    assert story.summary == "Two sentence summary."
    assert story.category == "research"
    assert story.status is StoryStatus.ENRICHED
    # Provenance requirement: "which provider AND model produced each
    # summary/analysis" -- tier 1's half of that (tier 2's is covered by
    # the analysis_provider assertions further down this file).
    assert story.summary_provider == "gemini:gemini-flash-latest"


def test_tier1_ignores_unscored_stories(session):
    _seed_story(session, score=None)
    bulk = _FakeProvider("gemini", "gemini-flash-latest", Tier.BULK, responses=[])
    router = Router(bulk=bulk)

    result = enrich_tier1(session, router, ProvidersConfig())

    assert result.tier1_processed == 0
    assert bulk.prompts == []


def test_tier1_ignores_already_enriched_stories(session):
    _seed_story(session, status=StoryStatus.ENRICHED)
    bulk = _FakeProvider("gemini", "gemini-flash-latest", Tier.BULK, responses=[])
    router = Router(bulk=bulk)

    result = enrich_tier1(session, router, ProvidersConfig())

    assert result.tier1_processed == 0


def test_tier1_strips_a_markdown_json_fence(session):
    _seed_story(session)
    fenced = "```json\n" + TIER1_JSON + "\n```"
    bulk = _FakeProvider("gemini", "gemini-flash-latest", Tier.BULK, responses=[fenced])
    router = Router(bulk=bulk)

    result = enrich_tier1(session, router, ProvidersConfig())

    assert result.tier1_processed == 1


def test_tier1_one_bad_story_does_not_stop_the_batch(session):
    s1 = _seed_story(session, titles=("first",))
    s2 = _seed_story(session, titles=("second",))
    # story ids assigned in insertion order; first response is garbage,
    # second is valid -- whichever story is claimed first must fail in
    # isolation and the other must still be processed.
    bulk = _FakeProvider("gemini", "gemini-flash-latest", Tier.BULK,
                         responses=["not json at all", TIER1_JSON])
    router = Router(bulk=bulk)

    result = enrich_tier1(session, router, ProvidersConfig())

    assert result.tier1_processed == 1
    assert result.tier1_failed == 1
    statuses = {s.id: s.status for s in (session.get(Story, s1.id), session.get(Story, s2.id))}
    assert StoryStatus.ENRICHED in statuses.values()
    assert StoryStatus.NEW in statuses.values()


def test_tier1_never_sends_item_text_in_the_prompt(session):
    story = _seed_story(session, titles=("Public headline",))
    item = story.items[0]
    item.text = "SECRET FULL ARTICLE BODY that must never reach a third party LLM prompt"
    session.commit()
    bulk = _FakeProvider("gemini", "gemini-flash-latest", Tier.BULK, responses=[TIER1_JSON])
    router = Router(bulk=bulk)

    enrich_tier1(session, router, ProvidersConfig())

    assert "SECRET FULL ARTICLE BODY" not in bulk.prompts[0]


# --- Tier 2 -----------------------------------------------------------------

def test_tier2_analyzes_an_eligible_story_above_the_score_cut(session):
    story = _seed_story(session, score=0.9, status=StoryStatus.ENRICHED)
    deep = _FakeProvider("claude-code", "claude-code", Tier.DEEP, responses=["deep analysis"])
    bulk = _FakeProvider("gemini", "gemini-flash-latest", Tier.BULK, responses=[])
    router = Router(bulk=bulk, deep=deep)
    cfg = ProvidersConfig(tier2_score_cut=0.5, daily_budget=20)

    result = enrich_tier2(session, router, cfg)

    assert result.tier2_processed == 1
    assert result.tier2_degraded == 0
    session.refresh(story)
    assert story.analysis == "deep analysis"
    assert story.analysis_provider == "claude-code:claude-code"
    assert story.status is StoryStatus.ANALYZED
    assert story.analyzed_at is not None


def test_tier2_excludes_stories_below_the_score_cut(session):
    _seed_story(session, score=0.2, status=StoryStatus.ENRICHED)
    deep = _FakeProvider("claude-code", "claude-code", Tier.DEEP, responses=[])
    bulk = _FakeProvider("gemini", "gemini-flash-latest", Tier.BULK, responses=[])
    router = Router(bulk=bulk, deep=deep)
    cfg = ProvidersConfig(tier2_score_cut=0.5)

    result = enrich_tier2(session, router, cfg)

    assert result.tier2_processed == 0


def test_tier2_excludes_stories_not_yet_through_tier1(session):
    _seed_story(session, score=0.95, status=StoryStatus.NEW)
    deep = _FakeProvider("claude-code", "claude-code", Tier.DEEP, responses=[])
    router = Router(bulk=_FakeProvider("gemini", "gemini-flash-latest", Tier.BULK), deep=deep)
    cfg = ProvidersConfig(tier2_score_cut=0.5)

    result = enrich_tier2(session, router, cfg)

    assert result.tier2_processed == 0


def test_tier2_picks_the_highest_scoring_stories_first_within_budget(session):
    low = _seed_story(session, score=0.6, status=StoryStatus.ENRICHED, titles=("low",))
    high = _seed_story(session, score=0.95, status=StoryStatus.ENRICHED, titles=("high",))
    deep = _FakeProvider("claude-code", "claude-code", Tier.DEEP, responses=["only one"])
    router = Router(bulk=_FakeProvider("gemini", "gemini-flash-latest", Tier.BULK), deep=deep)
    cfg = ProvidersConfig(tier2_score_cut=0.5, daily_budget=1)

    result = enrich_tier2(session, router, cfg)

    assert result.tier2_processed == 1
    session.refresh(high)
    session.refresh(low)
    assert high.status is StoryStatus.ANALYZED
    assert low.status is StoryStatus.ENRICHED  # untouched, budget exhausted


def test_tier2_degrades_to_bulk_and_flags_story_for_retry(session):
    story = _seed_story(session, score=0.9, status=StoryStatus.ENRICHED)
    deep = _FakeProvider("claude-code", "claude-code", Tier.DEEP, healthy=False)
    bulk = _FakeProvider("gemini", "gemini-flash-latest", Tier.BULK, responses=["cheap gloss"])
    router = Router(bulk=bulk, deep=deep)
    cfg = ProvidersConfig(tier2_score_cut=0.5)

    result = enrich_tier2(session, router, cfg)

    assert result.tier2_processed == 1
    assert result.tier2_degraded == 1
    session.refresh(story)
    assert story.analysis == "cheap gloss"
    assert story.analysis_provider == "gemini:gemini-flash-latest"
    assert story.status is StoryStatus.RETRY


def test_tier2_retries_a_previously_degraded_story(session):
    _seed_story(session, score=0.9, status=StoryStatus.RETRY)
    deep = _FakeProvider("claude-code", "claude-code", Tier.DEEP, responses=["real analysis"])
    router = Router(bulk=_FakeProvider("gemini", "gemini-flash-latest", Tier.BULK), deep=deep)
    cfg = ProvidersConfig(tier2_score_cut=0.5)

    result = enrich_tier2(session, router, cfg)

    assert result.tier2_processed == 1
    assert result.tier2_degraded == 0


def test_tier2_respects_daily_budget_across_prior_runs_today(session):
    now = datetime.now(timezone.utc)
    # Discovered flaky during phaseA live verification: `now - 1 hour` can
    # land on the PREVIOUS UTC calendar day whenever the test runs within
    # the first hour after UTC midnight, so the "3 already spent today"
    # setup silently seeds yesterday instead -- clamp to the start of
    # today's UTC day so "prior runs today" is true regardless of wall
    # clock time.
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    earlier_today = max(start_of_today, now - timedelta(hours=1))
    for i in range(3):
        _seed_story(session, score=0.9, status=StoryStatus.ANALYZED, titles=(f"old{i}",),
                   analyzed_at=earlier_today)
    fresh = _seed_story(session, score=0.9, status=StoryStatus.ENRICHED, titles=("fresh",))
    deep = _FakeProvider("claude-code", "claude-code", Tier.DEEP, responses=["analysis"])
    router = Router(bulk=_FakeProvider("gemini", "gemini-flash-latest", Tier.BULK), deep=deep)
    cfg = ProvidersConfig(tier2_score_cut=0.5, daily_budget=3)

    result = enrich_tier2(session, router, cfg, now=now)

    assert result.tier2_processed == 0  # budget of 3 already spent today
    session.refresh(fresh)
    assert fresh.status is StoryStatus.ENRICHED


def test_tier2_a_new_day_resets_the_budget(session):
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    for i in range(5):
        _seed_story(session, score=0.9, status=StoryStatus.ANALYZED, titles=(f"old{i}",),
                   analyzed_at=yesterday)
    fresh = _seed_story(session, score=0.9, status=StoryStatus.ENRICHED, titles=("fresh",))
    deep = _FakeProvider("claude-code", "claude-code", Tier.DEEP, responses=["analysis"])
    router = Router(bulk=_FakeProvider("gemini", "gemini-flash-latest", Tier.BULK), deep=deep)
    cfg = ProvidersConfig(tier2_score_cut=0.5, daily_budget=3)

    result = enrich_tier2(session, router, cfg, now=datetime.now(timezone.utc))

    assert result.tier2_processed == 1
    session.refresh(fresh)
    assert fresh.status is StoryStatus.ANALYZED


def test_tier2_one_bad_story_does_not_stop_the_batch(session):
    s1 = _seed_story(session, score=0.9, status=StoryStatus.ENRICHED, titles=("first",))
    s2 = _seed_story(session, score=0.85, status=StoryStatus.ENRICHED, titles=("second",))
    deep = _FakeProvider("claude-code", "claude-code", Tier.DEEP,
                         responses=[ProviderError("boom"), "ok"])
    router = Router(bulk=_FakeProvider("gemini", "gemini-flash-latest", Tier.BULK), deep=deep)
    cfg = ProvidersConfig(tier2_score_cut=0.5, daily_budget=20)

    result = enrich_tier2(session, router, cfg)

    assert result.tier2_processed == 1
    assert result.tier2_failed == 1


# --- combined enrich() ------------------------------------------------------

def test_enrich_runs_tier1_then_tier2_in_one_pass(session):
    _seed_story(session, score=0.95)  # NEW -> tier1 then eligible for tier2
    bulk = _FakeProvider("gemini", "gemini-flash-latest", Tier.BULK, responses=[TIER1_JSON])
    deep = _FakeProvider("claude-code", "claude-code", Tier.DEEP, responses=["deep analysis"])
    router = Router(bulk=bulk, deep=deep)
    cfg = ProvidersConfig(tier2_score_cut=0.5, daily_budget=20)

    result = enrich(session, router, cfg)

    assert result.tier1_processed == 1
    assert result.tier2_processed == 1


def test_tier1_enriches_the_newest_stories_first(session):
    """Tier 1 used to select `.order_by(Story.id)` -- insertion order, i.e.
    OLDEST first. That is backwards for a feed that publishes a rolling
    window of recent news: with a backlog larger than the per-run limit,
    every call is spent on stories too old to be published, and the ones a
    reader actually sees stay "Uncategorized" indefinitely. Measured on the
    live bundle before this fix: 328 published stories, 110 summarized.

    Story ids here deliberately run opposite to the dates, so ordering by
    id and ordering by recency give different answers -- otherwise the
    assertion would pass either way.
    """
    now = datetime.now(timezone.utc)
    session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
    session.flush()
    ids: dict[str, int] = {}
    for age_days, title in ((30, "oldest"), (10, "middle"), (1, "newest")):
        when = now - timedelta(days=age_days)
        story = Story(title=title, first_seen=when, updated_at=when, item_count=1,
                      score=0.5, status=StoryStatus.NEW)
        session.add(story)
        session.flush()
        ids[title] = story.id
        session.add(Item(source_id="s", url=f"http://x/{title}", url_hash=f"h-{title}",
                         title=title, story_id=story.id))
    session.commit()
    # Insertion order is oldest -> newest, so the lowest id is the oldest
    # story: `.order_by(Story.id)` would pick exactly the wrong one.
    assert ids["oldest"] < ids["newest"]

    bulk = _FakeProvider("g", "m", Tier.BULK, responses=[TIER1_JSON])
    router = Router(bulk=bulk, deep=_FakeProvider("c", "c", Tier.DEEP))

    enrich_tier1(session, router, ProvidersConfig(), limit=1)

    # Asserted on id, not title: Tier 1 overwrites title with the canonical
    # headline, so the seeded name is gone by the time we look.
    enriched = session.query(Story).filter(Story.status == StoryStatus.ENRICHED).all()
    assert [s.id for s in enriched] == [ids["newest"]]
