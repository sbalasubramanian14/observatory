from __future__ import annotations
from datetime import datetime, timedelta, timezone
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.clustering.adjudicate import Adjudicator, Verdict
from feed.clustering.entities import extract_entities
from feed.clustering.signals import (blend, cosine, entity_overlap, link_overlap,
                                     time_proximity)
from feed.config import ClusteringConfig
from feed.embedding.base import pack, unpack
from feed.models import Item, Stage, Story
from feed.stages.base import StageResult


def _finite_embedding(item: Item) -> np.ndarray:
    """Unpack an item's embedding and reject it if it is not finite.

    feed.clustering.signals.cosine() already fails safe on a corrupt
    (nan/inf) vector by returning 0.0 instead of raising or returning nan,
    but that is silent: the item would simply never match anything, forever,
    with no record of why. A pure signal function must not log, so
    visibility belongs here. Raising lets the per-item failure isolation in
    cluster() mark the item Stage.FAILED with a clear reason instead of
    letting it rot unclustered with no trace.
    """
    vec = unpack(item.embedding)
    if not np.isfinite(vec).all():
        raise ValueError(
            f"embedding for item {item.id} contains non-finite values (nan/inf)"
        )
    return vec


def pair_score(left: Item, right: Item, cfg: ClusteringConfig) -> float:
    """Blend the four signals from spec 3.4.

    cosine and entity overlap carry the measured weights. Shared outbound
    links nudge upward only - two articles citing the same primary document
    are strong evidence of one event, but not citing one is no evidence
    against. Time proximity decays the whole score toward the window edge, so
    a same-day match beats an otherwise identical match two days apart.

    `left` is the item currently being clustered, already validated finite
    by the caller. `right` is a persisted candidate that was itself
    validated finite when it went through this same path earlier; if its
    stored embedding has since become corrupt, cosine()'s own fail-safe
    (return 0.0) keeps this function from raising on someone else's item.
    """
    cos = cosine(unpack(left.embedding), unpack(right.embedding))
    ents = entity_overlap(
        extract_entities(f"{left.title} {left.text or ''}"),
        extract_entities(f"{right.title} {right.text or ''}"),
    )
    score = blend(cos, ents, cosine_weight=cfg.cosine_weight,
                  entity_weight=cfg.entity_weight)

    links = link_overlap(left.outbound_links or [], right.outbound_links or [])
    score = min(1.0, score + 0.10 * links)

    if left.published_at and right.published_at:
        score *= time_proximity(left.published_at, right.published_at,
                                cfg.window_hours)
    return score


def cluster(session: Session, cfg: ClusteringConfig, adjudicator: Adjudicator,
            *, now: datetime | None = None, limit: int = 200) -> StageResult:
    now = now or datetime.now(timezone.utc)
    result = StageResult(name="cluster")
    cutoff = now - timedelta(hours=cfg.window_hours)

    pending = list(session.scalars(
        select(Item).where(Item.stage == Stage.EMBEDDED).order_by(Item.id).limit(limit)
    ))

    for item in pending:
        try:
            _finite_embedding(item)

            # Ruling 1: no Item.stage predicate here. Item.story_id.is_not(None)
            # is the real condition for "this item belongs to a story" -- an
            # item keeps that membership no matter how far down the pipeline
            # it has since travelled (e.g. Task 13 advances it to
            # Stage.SCORED). Filtering on Stage.CLUSTERED would make every
            # already-scored item invisible as a candidate on the next
            # pipeline run, silently breaking cross-run clustering forever.
            # The embedding_model_id filter stays: vectors from different
            # models are not comparable.
            candidates = list(session.scalars(
                select(Item).where(
                    Item.story_id.is_not(None),
                    Item.published_at >= cutoff,
                    Item.embedding_model_id == item.embedding_model_id,
                )
            ))

            threshold = cfg.threshold_for(item.embedding_model_id)
            best_score, best_story_id, best_other = float("-inf"), None, None
            for other in candidates:
                if item.published_at and other.published_at:
                    gap = abs((item.published_at - other.published_at).total_seconds())
                    if gap > cfg.window_hours * 3600:
                        continue
                s = pair_score(item, other, cfg)
                if s > best_score:
                    best_score, best_story_id, best_other = s, other.story_id, other

            if best_story_id is not None:
                # Ruling 2: the adjudicator's own band is a fixed
                # cfg.merge_threshold-centred reference (by convention, the
                # value it is constructed with matches cfg.merge_threshold).
                # The two embedding models this project supports have
                # measurably different similarity scales (bge-small
                # same-story minimum 0.695, MiniLM 0.412), so the RAW
                # blended score cannot be compared against that fixed
                # reference directly. Shift it so that clearing the
                # model-scoped threshold_for() lands exactly on
                # cfg.merge_threshold -- the adjudicator's ambiguous_band
                # then stays meaningful (a relative margin around the
                # effective threshold) regardless of which model produced
                # the vectors.
                adjusted = best_score - threshold + cfg.merge_threshold
                verdict = adjudicator.decide(adjusted, item, best_other)
            else:
                verdict = Verdict.DIFFERENT

            if verdict is Verdict.SAME and best_story_id is not None:
                story = session.get(Story, best_story_id)
            else:
                story = Story(title=item.title, first_seen=item.published_at or now,
                              updated_at=item.published_at or now, item_count=0)
                session.add(story)
                session.flush()

            item.story_id = story.id
            item.stage = Stage.CLUSTERED
            item.error = None
            session.flush()

            members = list(session.scalars(select(Item).where(Item.story_id == story.id)))
            story.item_count = len(members)
            story.outlet_count = len({m.source_id for m in members})
            story.updated_at = max(m.published_at or now for m in members)
            vectors = np.array([unpack(m.embedding) for m in members], dtype=np.float32)
            story.centroid = pack(vectors.mean(axis=0))
            session.commit()
            result.processed += 1
        except Exception as exc:
            session.rollback()
            fresh = session.get(Item, item.id)
            if fresh is not None:
                fresh.stage = Stage.FAILED
                fresh.error = f"cluster: {type(exc).__name__}: {exc}"
                session.commit()
            result.failed += 1
            result.errors.append((item.id, str(exc)))

    return result
