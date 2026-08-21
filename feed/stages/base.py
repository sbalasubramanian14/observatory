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
    errors: list[tuple[int, str]] = field(default_factory=list)


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
