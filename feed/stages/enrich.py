from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from feed.config import ProvidersConfig
from feed.models import Story, StoryStatus
from feed.providers.base import Tier
from feed.providers.router import Router

log = logging.getLogger(__name__)

# Tier 2 candidates below (spec 3.5): stories that already went through
# Tier 1 (ENRICHED) or had a degraded Tier 2 attempt last time (RETRY,
# spec 3.5: "flagged for retry").
_TIER2_ELIGIBLE = (StoryStatus.ENRICHED, StoryStatus.RETRY)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class EnrichResult:
    tier1_processed: int = 0
    tier1_failed: int = 0
    tier2_processed: int = 0
    tier2_failed: int = 0
    tier2_degraded: int = 0
    errors: list[tuple[int, str]] = field(default_factory=list)


def _parse_json_object(text: str) -> dict:
    """Best-effort JSON extraction from an LLM response.

    Models routinely wrap JSON in a ```json ... ``` fence despite being
    asked not to; strip that before parsing. Raises ValueError (caught by
    the per-story isolation in this module, exactly like any other
    provider failure) if the result still isn't a JSON object.
    """
    cleaned = _FENCE.sub("", text.strip()).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _story_context(session: Session, story: Story, *, max_items: int = 8) -> str:
    """Titles only -- deliberately never item.text or item.summary here.

    Not because this prompt is published (it is local-only, sent to a
    third-party LLM API, and never touches the bundle -- spec 4.2 governs
    what gets *published*, not what an internal enrichment prompt may
    read). The restriction here is practical: story.items can be large and
    full article bodies would blow the prompt budget for no benefit --
    titles across outlets are what Tier 1 needs to produce a canonical
    headline, summary, and category.
    """
    items = list(story.items)[:max_items]
    lines = [f"- {it.title}" for it in items]
    return "\n".join(lines)


def _tier1_prompt(session: Session, story: Story) -> str:
    return (
        "You are a news editor. Below are headlines from multiple outlets "
        "covering the same underlying AI news event. Produce a single "
        "canonical headline, a two-sentence summary, and a short category "
        "label (one of: research, product, industry, policy, "
        "infrastructure, other).\n\n"
        f"Headlines:\n{_story_context(session, story)}\n\n"
        'Respond with ONLY a JSON object of the form '
        '{"headline": "...", "summary": "...", "category": "..."}. '
        "No markdown, no code fence, no extra text."
    )


def _tier2_prompt(session: Session, story: Story) -> str:
    return (
        "You are a senior AI industry analyst. A story has cleared this "
        "outlet's importance bar. Given its canonical headline and summary "
        "below, write a short \"why this matters\" analysis: what is "
        "genuinely new versus prior art, what it affects, and any "
        "connections to earlier developments you can infer.\n\n"
        f"Headline: {story.title}\n"
        f"Summary: {story.summary}\n"
        f"Category: {story.category}\n\n"
        "Respond with plain text, a few sentences. No markdown headers."
    )


def _tier2_fallback_prompt(story: Story) -> str:
    """The "simpler prompt" the router falls back to on the BULK provider
    when DEEP is unavailable (spec 3.5). Cheaper and asks for less than the
    full analysis prompt -- a one-paragraph gloss rather than deep
    reasoning, matching what the BULK model can produce reliably.
    """
    return (
        f"In one short paragraph, explain why this AI news story matters: "
        f"{story.title}. {story.summary or ''}"
    )


def enrich_tier1(session: Session, router: Router, cfg: ProvidersConfig,
                 *, limit: int = 100) -> EnrichResult:
    """Tier 1 (BULK): once per story, never per item (spec 3.5).

    Runs on every scored story not yet through Tier 1 -- there is no score
    cut here, only Tier 2 is gated by one. Per-story failure isolation: one
    bad story is rolled back and recorded, the rest of the batch proceeds.
    """
    result = EnrichResult()
    story_ids = session.scalars(
        select(Story.id)
        .where(Story.status == StoryStatus.NEW, Story.score.is_not(None))
        .order_by(Story.id)
        .limit(limit)
    ).all()

    for story_id in story_ids:
        story = session.get(Story, story_id)
        if story is None:
            continue
        try:
            route = router.complete(_tier1_prompt(session, story), tier=Tier.BULK)
            parsed = _parse_json_object(route.text)
            headline = (parsed.get("headline") or "").strip()
            summary = (parsed.get("summary") or "").strip()
            category = (parsed.get("category") or "").strip() or None
            if headline:
                story.title = headline
            story.summary = summary or None
            story.category = category
            story.status = StoryStatus.ENRICHED
            session.commit()
            result.tier1_processed += 1
        except Exception as exc:
            session.rollback()
            result.tier1_failed += 1
            result.errors.append((story_id, str(exc)))
            log.warning("enrich tier1: story=%s failed: %s", story_id, exc)

    return result


def enrich_tier2(session: Session, router: Router, cfg: ProvidersConfig,
                 *, now: datetime | None = None) -> EnrichResult:
    """Tier 2 (DEEP): only stories above cfg.tier2_score_cut, budgeted by
    cfg.daily_budget calls/day (spec 3.5). Budget is tracked by counting
    stories already analyzed (Story.analyzed_at) since local midnight UTC,
    across however many pipeline runs happened today -- not per-invocation
    -- so a `feed run --enrich` scheduled every 30 minutes still respects
    one daily ceiling rather than resetting it on every run.
    """
    result = EnrichResult()
    now = now or datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    already_today = session.scalar(
        select(func.count()).select_from(Story).where(Story.analyzed_at >= day_start)
    ) or 0
    budget_remaining = max(0, cfg.daily_budget - already_today)
    if budget_remaining == 0:
        log.info("enrich tier2: daily budget of %d already spent today", cfg.daily_budget)
        return result

    candidates = session.scalars(
        select(Story)
        .where(Story.status.in_(_TIER2_ELIGIBLE), Story.score >= cfg.tier2_score_cut)
        .order_by(Story.score.desc())
        .limit(budget_remaining)
    ).all()

    for story in candidates:
        story_id = story.id
        try:
            route = router.complete(
                _tier2_prompt(session, story),
                tier=Tier.DEEP,
                deep_prompt=_tier2_fallback_prompt(story),
            )
            story.analysis = route.text.strip()
            story.analysis_provider = f"{route.provider}:{route.model}"
            story.analyzed_at = now
            story.status = StoryStatus.RETRY if route.degraded else StoryStatus.ANALYZED
            session.commit()
            result.tier2_processed += 1
            if route.degraded:
                result.tier2_degraded += 1
        except Exception as exc:
            session.rollback()
            result.tier2_failed += 1
            result.errors.append((story_id, str(exc)))
            log.warning("enrich tier2: story=%s failed: %s", story_id, exc)

    return result


def enrich(session: Session, router: Router, cfg: ProvidersConfig,
          *, now: datetime | None = None) -> EnrichResult:
    """Run Tier 1 then Tier 2 in one pass, combining both results."""
    r1 = enrich_tier1(session, router, cfg)
    r2 = enrich_tier2(session, router, cfg, now=now)
    return EnrichResult(
        tier1_processed=r1.tier1_processed,
        tier1_failed=r1.tier1_failed,
        tier2_processed=r2.tier2_processed,
        tier2_failed=r2.tier2_failed,
        tier2_degraded=r2.tier2_degraded,
        errors=r1.errors + r2.errors,
    )
