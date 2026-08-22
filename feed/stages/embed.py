from __future__ import annotations
import logging
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.embedding.base import Embedder, pack
from feed.models import Item, Stage
from feed.stages.base import StageResult

log = logging.getLogger(__name__)


def embed_text_for(item: Item) -> str:
    """Title carries most of the story identity; text disambiguates.

    Both models truncate (256 tokens for MiniLM, 512 for bge), so putting the
    title first guarantees the most identifying text survives truncation.
    """
    return f"{item.title}\n\n{item.text or ''}".strip()


def _embed_one(session: Session, embedder: Embedder, item_id: int, result: StageResult) -> None:
    """Fallback path: encode and persist a single item, isolating its failure.

    Only reached after the batched call above already raised, so this is
    off the hot path deliberately -- see the module docstring rationale in
    the task-9 report for why a batch failure must not condemn every row
    in the batch to a terminal Stage.FAILED.

    The try covers the *entire* per-item operation -- encode, pack, and
    commit -- not just encode(). A failure in pack() (e.g. a malformed
    vector shape) or in commit() (e.g. a constraint violation or full disk)
    must be caught here exactly like an encode() failure, or it escapes
    this function, the calling loop, and embed() itself, leaving every
    later item in the batch unattempted with no StageResult returned.
    Mirrors feed.stages.base.run_stage: on failure, roll back and re-fetch
    the item by id before writing the FAILED state, since a raised
    commit() can leave the in-memory object's pending changes rolled back
    and the object expired.
    """
    item = session.get(Item, item_id)
    try:
        vec = embedder.encode([embed_text_for(item)])[0]
        item.embedding = pack(vec)
        item.embedding_model_id = embedder.model_id
        item.stage = Stage.EMBEDDED
        item.error = None
        session.commit()
    except Exception as exc:
        session.rollback()
        fresh = session.get(Item, item_id)
        if fresh is not None:
            fresh.stage = Stage.FAILED
            fresh.error = f"embedding failed: {type(exc).__name__}: {exc}"
            session.commit()
        result.failed += 1
        result.errors.append((item_id, str(exc)))
        log.warning("embed item=%s failed: %s", item_id, exc)
    else:
        result.processed += 1


def embed(session: Session, embedder: Embedder, limit: int = 256) -> StageResult:
    result = StageResult(name="embed")
    stmt = (select(Item).where(Item.stage == Stage.NORMALIZED)
            .order_by(Item.id).limit(limit))
    items = list(session.scalars(stmt))
    if not items:
        return result

    try:
        vectors = embedder.encode([embed_text_for(i) for i in items])
    except Exception as exc:
        # The fast batched path is what this stage exists for -- see the
        # module-level rationale. But Stage.FAILED is terminal, and a
        # single malformed row (or a transient backend hiccup) must not
        # strand every other healthy item in the batch. Fall back to
        # encoding one item at a time so only the genuinely-bad rows end
        # up FAILED; this fallback runs ONLY on this error path.
        session.rollback()
        log.warning(
            "embed batch of %d failed, falling back to per-item encoding: %s",
            len(items), exc,
        )
        for item in items:
            _embed_one(session, embedder, item.id, result)
        return result

    for item, vec in zip(items, vectors):
        item.embedding = pack(vec)
        item.embedding_model_id = embedder.model_id
        item.stage = Stage.EMBEDDED
        item.error = None
    session.commit()
    result.processed = len(items)
    return result
