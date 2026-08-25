"""Top 50 -- importance as judged by the DEEP provider (Claude Code).

The published `score` is authority + velocity + novelty: reader-independent
by design, and entirely structural. It counts how many distinct outlets
picked something up and how new the wording is, which is exactly why it
cannot tell a frontier model release from a thoroughly-syndicated funding
round. Both look identical to it.

So the score's job here is only to NOMINATE. This stage hands the
shortlist to Claude Code, which reads the headlines and summaries and
places each story in an importance band with a one-line justification.
That judgement is what the Top 50 view renders.

One provider call ranks the whole shortlist rather than one call per
story: the question "which of these matters most" is inherently
comparative, and asking it fifty times in isolation would answer a
different, worse question -- besides costing fifty times the Tier 2 budget.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from feed.config import ProvidersConfig
from feed.models import Story
from feed.providers.base import Tier
from feed.providers.router import Router

log = logging.getLogger(__name__)

# Rendered as groups, in this order. Three bands, not ten: the point is a
# reader's glance, and a scale finer than the judgement behind it is false
# precision.
BANDS: tuple[str, ...] = ("landmark", "significant", "notable")

DEFAULT_TOP_N = 50

# How many stories the arithmetic score nominates per slot. Overshooting
# lets Claude Code promote something the score underrated -- with a 1:1
# shortlist the "judgement" could only ever reorder, never rescue.
SHORTLIST_MULTIPLIER = 2

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_PROMPT = """\
You are ranking AI news stories by importance for a reader who follows the \
field closely and wants to know what actually matters today.

Below are {n} candidate stories, each with an id, headline, and summary. \
Choose the {top_n} most important and rank them 1..N, most important first.

Judge by consequence, not by how much coverage something received. A \
frontier capability result, a major model or hardware release, a binding \
regulation, or a safety finding outranks a funding round, a partnership \
announcement, a product tweak, or an incremental benchmark -- however \
widely those were reported.

Assign each chosen story exactly one band:
  "landmark"    - changes what is possible or what is permitted; a reader \
who missed it would be out of date
  "significant" - matters to people working in the area; worth knowing this \
week
  "notable"     - genuinely interesting, but nothing downstream depends on it

Give each a "reason": one short sentence, at most 20 words, saying why it \
placed there. Do not restate the headline.

Return ONLY a JSON object, no prose and no code fence:
{{"ranked": [{{"id": <int>, "rank": <int>, "band": "<band>", "reason": "<text>"}}]}}

Use only ids from the list. Omit stories that do not deserve a place.

CANDIDATES:
{candidates}
"""


@dataclass
class RankResult:
    ranked: int = 0
    rejected: int = 0
    cleared: int = 0
    provider: str | None = None
    error: str | None = None


def _candidate_block(stories: list[Story]) -> str:
    lines = []
    for s in stories:
        summary = (s.summary or "").strip().replace("\n", " ")
        if len(summary) > 300:
            summary = summary[:300] + "..."
        lines.append(
            f"- id={s.id} | outlets={s.outlet_count} | {s.title.strip()}"
            + (f"\n    {summary}" if summary else "")
        )
    return "\n".join(lines)


def _parse_ranked(text: str) -> list[dict]:
    cleaned = _FENCE.sub("", text.strip()).strip()
    parsed = json.loads(cleaned)
    if isinstance(parsed, list):        # tolerate a bare array
        return parsed
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    ranked = parsed.get("ranked")
    if not isinstance(ranked, list):
        raise ValueError("response has no 'ranked' array")
    return ranked


def rank_top(
    session: Session,
    router: Router,
    cfg: ProvidersConfig,
    *,
    top_n: int = DEFAULT_TOP_N,
    window_days: int = 5,
    now: datetime | None = None,
) -> RankResult:
    """Rank the most important stories in the current publish window.

    Ranks are rewritten wholesale on every successful run: a story that
    drops out of the window loses its rank, so the list rotates instead of
    accumulating. A FAILED run changes nothing -- the site keeps serving
    yesterday's judgement, which beats an empty Top 50.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)
    result = RankResult()

    candidates = list(session.scalars(
        select(Story)
        .where(Story.score.is_not(None), Story.updated_at >= cutoff)
        .order_by(Story.score.desc(), Story.id)
        .limit(top_n * SHORTLIST_MULTIPLIER)
    ))
    if not candidates:
        log.info("rank: no stories in the last %d days -- nothing to rank", window_days)
        return result

    prompt = _PROMPT.format(
        n=len(candidates), top_n=min(top_n, len(candidates)),
        candidates=_candidate_block(candidates),
    )

    try:
        route = router.complete(prompt, tier=Tier.DEEP)
        ranked = _parse_ranked(route.text)
    except Exception as exc:  # provider failure or unparseable response
        # Deliberately before any write: existing ranks survive untouched.
        log.warning("rank: ranking failed, keeping the previous Top %d: %s", top_n, exc)
        result.error = str(exc)
        return result

    allowed = {s.id: s for s in candidates}
    accepted: list[tuple[int, int, str, str]] = []   # (rank, id, band, reason)
    seen_ids: set[int] = set()
    seen_ranks: set[int] = set()

    for entry in ranked:
        if not isinstance(entry, dict):
            result.rejected += 1
            continue
        sid, band = entry.get("id"), entry.get("band")
        reason = (entry.get("reason") or "").strip()
        try:
            rank = int(entry.get("rank"))
        except (TypeError, ValueError):
            result.rejected += 1
            continue
        # Every rejection below is a model mistake that would otherwise
        # corrupt the page: an id outside the shortlist writes a rank onto
        # an unrelated (possibly out-of-window) story; an unknown band
        # renders as an empty group; a duplicate id or rank makes the
        # ordering ambiguous.
        if sid not in allowed or sid in seen_ids:
            result.rejected += 1
            continue
        if band not in BANDS:
            log.warning("rank: story=%s unknown band %r -- rejected", sid, band)
            result.rejected += 1
            continue
        if rank in seen_ranks or rank < 1:
            result.rejected += 1
            continue
        seen_ids.add(sid)
        seen_ranks.add(rank)
        accepted.append((rank, sid, band, reason))

    if not accepted:
        result.error = "no usable entries in the ranking response"
        log.warning("rank: %s -- previous Top %d left in place", result.error, top_n)
        return result

    accepted.sort()
    accepted = accepted[:top_n]

    # Renumber 1..N. The model's own numbering can have gaps once entries
    # are rejected, and a Top 50 that jumps from 7 to 9 looks broken.
    keep = {sid for _, sid, _, _ in accepted}
    cleared = session.execute(
        update(Story)
        .where(Story.importance_rank.is_not(None), Story.id.not_in(keep))
        .values(importance_rank=None, importance_band=None, importance_reason=None)
    )
    result.cleared = cleared.rowcount or 0

    provider = f"{route.provider}:{route.model}"
    for position, (_, sid, band, reason) in enumerate(accepted, start=1):
        story = allowed[sid]
        story.importance_rank = position
        story.importance_band = band
        story.importance_reason = reason or None
        story.ranked_at = now
        story.ranked_by = provider

    session.commit()

    result.ranked = len(accepted)
    result.provider = provider
    log.info("rank: top %d written by %s (rejected=%d, cleared=%d)",
             result.ranked, provider, result.rejected, result.cleared)
    return result
