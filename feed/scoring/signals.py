from __future__ import annotations
import math
from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.clustering.signals import cosine
from feed.embedding.base import unpack
from feed.models import Entity, Item, Source, Story, StoryEntity


def authority(session: Session, story: Story) -> float:
    """Mean authority of the DISTINCT sources contributing to this story.

    Deliberately not a per-item average. Source.authority is a "static
    per-source weight" (spec 3.6 #1) -- averaging over items instead of
    distinct sources would let one prolific outlet drag (or inflate) the
    signal in proportion to how many items it filed, the same volume-based
    distortion that outlet-counting in velocity() exists to prevent, just
    surfacing through a different signal.
    """
    rows = session.scalars(
        select(Source.authority)
        .join(Item, Item.source_id == Source.id)
        .where(Item.story_id == story.id)
        .distinct()
    ).all()
    return float(sum(rows) / len(rows)) if rows else 0.5


def velocity(story: Story) -> float:
    """Independent outlets, log-compressed. Counting outlets rather than
    articles is what stops one publisher's syndication network from
    manufacturing importance. Reads Story.outlet_count directly (maintained
    by the cluster stage across ALL of a story's items, regardless of their
    current pipeline stage) rather than recomputing from items, since this
    function deliberately takes no session.
    """
    return min(1.0, math.log1p(max(0, story.outlet_count)) / math.log1p(10))


def novelty(session: Session, story: Story, days: int = 90) -> float:
    """1 minus the peak similarity against prior stories in the last `days`.

    High similarity to something already covered means this is a follow-up,
    not news. Only stories with first_seen STRICTLY BEFORE this story's
    count as prior art: a story must not suppress its own novelty (it can
    never be strictly-before itself) and must not be suppressed by a
    near-duplicate that shows up later.
    """
    if story.centroid is None:
        return 1.0
    cutoff = (story.first_seen or datetime.now(timezone.utc)) - timedelta(days=days)
    others = session.scalars(
        select(Story).where(Story.id != story.id, Story.first_seen >= cutoff,
                            Story.first_seen < story.first_seen,
                            Story.centroid.is_not(None))
    ).all()
    if not others:
        return 1.0
    mine = unpack(story.centroid)
    peak = max(cosine(mine, unpack(o.centroid)) for o in others)
    return float(max(0.0, 1.0 - peak))


def entity_weight(session: Session, story: Story) -> float:
    """Importance of the most significant org/model entity involved.

    NOTE: as of Phase 1, nothing populates the Entity/StoryEntity tables --
    feed.clustering.entities.extract_entities() is used only transiently
    inside the cluster stage's pairwise entity_overlap() signal and its
    output is never persisted. `rows` is therefore always empty in
    practice, and this always returns the 0.5 default. See task-13-report.md
    for the consequence for the combined score. Implemented per the Task 13
    interface anyway; populating those tables is out of scope here.
    """
    rows = session.scalars(
        select(Entity.weight).join(StoryEntity, StoryEntity.entity_id == Entity.id)
        .where(StoryEntity.story_id == story.id)
    ).all()
    return float(max(rows)) if rows else 0.5
