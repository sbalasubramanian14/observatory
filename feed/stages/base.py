from __future__ import annotations
import logging
import traceback
from dataclasses import dataclass, field
from typing import Callable
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.models import Item, Stage

log = logging.getLogger(__name__)


@dataclass
class StageResult:
    name: str
    processed: int = 0
    failed: int = 0
    # Only meaningful for feed.stages.relevance.gate_relevance: how many of
    # `processed` were judged off-topic and routed to Stage.REJECTED
    # rather than left to continue to clustering. A subset of `processed`,
    # not additional to it -- see that stage's docstring for why an item
    # counts as "processed" either way (drain()'s termination check relies
    # on processed+failed shrinking the backlog every round). Left at the
    # dataclass default 0 for every other stage.
    rejected: int = 0
    errors: list[tuple[int, str]] = field(default_factory=list)
    # Number of drain() rounds folded into this result. A stage called
    # directly (not through drain()) is one round by definition.
    rounds: int = 1


Handler = Callable[[Session, Item], None]


def run_stage(
    session: Session,
    *,
    name: str,
    claim_stage: Stage,
    next_stage: Stage,
    handler: Handler,
    limit: int = 100,
) -> StageResult:
    """Claim items at `claim_stage`, run `handler` on each, advance to `next_stage`.

    Failure is isolated per row: a raising handler marks that row FAILED with
    the traceback and the batch continues. One broken source must never stall
    the pipeline.
    """
    result = StageResult(name=name)
    stmt = select(Item).where(Item.stage == claim_stage).order_by(Item.id).limit(limit)
    items = list(session.scalars(stmt))
    item_ids = [item.id for item in items]

    for item_id in item_ids:
        item = session.get(Item, item_id)
        try:
            handler(session, item)
        except Exception as exc:
            session.rollback()
            fresh = session.get(Item, item_id)
            if fresh is not None:
                fresh.stage = Stage.FAILED
                fresh.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                session.commit()
            result.failed += 1
            result.errors.append((item_id, str(exc)))
            log.warning("stage=%s item=%s failed: %s", name, item_id, exc)
        else:
            item.stage = next_stage
            item.error = None
            session.commit()
            result.processed += 1

    return result


DEFAULT_MAX_ROUNDS = 50

StageCall = Callable[[], StageResult]


def drain(stage_fn: StageCall, *, max_rounds: int = DEFAULT_MAX_ROUNDS) -> StageResult:
    """Call `stage_fn` repeatedly until a call makes no further progress.

    I1 fix: every stage in this pipeline claims a fixed-size batch per call
    (normalize's default limit=100, cluster's default limit=200, embed's
    limit=cfg.embedding.batch_size). `feed run` used to call each stage
    exactly once, so any day with more collected items than one batch (the
    spec's 600-1200 items/day easily clears a 100- or 200-item batch) left a
    growing remainder stranded at the prior stage, with nothing printed to
    say so. Looping the same stage call until it reports zero processed AND
    zero failed drains however many batches actually exist, in one `feed
    run` invocation.

    `max_rounds` is a safety cap, not the normal exit condition. A row that
    fails a stage is marked Stage.FAILED and leaves the claim query for
    that stage, exactly like a processed row does -- so a genuinely
    exhausted backlog always terminates the loop on its own, whether every
    remaining row succeeded, failed, or some mix of both. The cap only
    guards a hypothetical stage bug that reports nonzero progress each
    round without ever actually shrinking the backlog (e.g. a broken claim
    query), which would otherwise spin forever.
    """
    total: StageResult | None = None
    rounds = 0
    while rounds < max_rounds:
        res = stage_fn()
        rounds += 1
        if total is None:
            total = res
        else:
            total.processed += res.processed
            total.failed += res.failed
            total.rejected += res.rejected
            total.errors.extend(res.errors)
        if res.processed == 0 and res.failed == 0:
            break
    total.rounds = rounds
    return total
