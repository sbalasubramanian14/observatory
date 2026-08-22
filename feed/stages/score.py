from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.config import ScoringConfig
from feed.models import Item, Stage, Story
from feed.scoring.combine import combine
from feed.scoring.signals import authority, entity_weight, novelty, velocity
from feed.stages.base import StageResult


def score_stories(session: Session, cfg: ScoringConfig, *,
                  now: datetime | None = None) -> StageResult:
    """Compute the reader-independent importance score for every story that
    has at least one item newly arrived at Stage.CLUSTERED, and advance
    those items to Stage.SCORED.

    Only items currently at CLUSTERED are advanced -- a story can have a mix
    of already-SCORED items (from an earlier run) and freshly-CLUSTERED ones
    (a new article joined it since); re-touching the already-scored items
    would be pointless and, per Task 11's ruling, the cluster stage's own
    candidate query never filters on Item.stage, so nothing here should
    either. Per spec 3.1, one bad story is isolated (rolled back, recorded
    as a failure) and does not stop the rest of the batch.
    """
    now = now or datetime.now(timezone.utc)
    result = StageResult(name="score")

    story_ids = session.scalars(
        select(Item.story_id).where(Item.stage == Stage.CLUSTERED).distinct()
    ).all()

    for story_id in story_ids:
        story = session.get(Story, story_id)
        if story is None:
            continue
        try:
            parts = {
                "authority": authority(session, story),
                "velocity": velocity(story),
                "novelty": novelty(session, story),
                "entity": entity_weight(session, story),
            }
            story.score = combine(parts, cfg.weights)
            story.score_breakdown = parts
            for item in session.scalars(
                select(Item).where(Item.story_id == story.id,
                                   Item.stage == Stage.CLUSTERED)
            ):
                item.stage = Stage.SCORED
            session.commit()
            result.processed += 1
        except Exception as exc:
            session.rollback()
            result.failed += 1
            result.errors.append((story_id, str(exc)))

    return result
