from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session
import feed.sources  # noqa: F401  (registers plugins)
from feed.models import Item, Source, Stage
from feed.sources.base import url_hash
from feed.sources.registry import build_source

log = logging.getLogger(__name__)


@dataclass
class CollectResult:
    new_items: int = 0
    skipped_duplicates: int = 0
    source_errors: dict[str, str] = field(default_factory=dict)


def collect(session: Session, *, now: datetime | None = None) -> CollectResult:
    now = now or datetime.now(timezone.utc)
    result = CollectResult()

    for src in session.scalars(select(Source).where(Source.enabled.is_(True))):
        if src.last_run_at is not None:
            due = src.last_run_at + timedelta(minutes=src.cadence_minutes)
            if now < due:
                continue
        # I7 fix: the item-insert loop and both commits below used to sit
        # OUTSIDE this try -- only fetch() was covered. A DB error there
        # (a constraint violation, a full disk, any commit() failure)
        # escaped collect() entirely and aborted every other still-due
        # source in the same run, violating spec S3.1's per-source
        # isolation. The whole per-source body -- fetch, insert, and the
        # success commit -- is now one unit: any failure anywhere in it
        # rolls back and is recorded exactly like a fetch() failure always
        # was, and the loop moves on to the next source.
        try:
            plugin = build_source(src.plugin, src.id, dict(src.config or {}))
            raw_items = list(plugin.fetch(since=src.last_run_at))

            # Counted locally and only merged into `result` after a
            # successful commit -- if this source's insert loop or commit
            # fails partway through, the rollback below undoes the DB
            # writes, so the result counters must not reflect them either.
            new_items = 0
            skipped_duplicates = 0
            for raw in raw_items:
                h = url_hash(raw.url)
                exists = session.scalar(select(Item.id).where(Item.url_hash == h))
                if exists:
                    skipped_duplicates += 1
                    continue
                session.add(Item(
                    source_id=src.id,
                    url=raw.url,
                    url_hash=h,
                    title=raw.title,
                    summary=raw.summary,
                    outbound_links=raw.outbound_links or [],
                    published_at=raw.published_at,
                    fetched_at=now,
                    stage=Stage.COLLECTED,
                ))
                new_items += 1

            src.last_run_at = now
            src.last_error = None
            src.consecutive_failures = 0
            session.commit()
            result.new_items += new_items
            result.skipped_duplicates += skipped_duplicates
        except Exception as exc:
            # Do not advance last_run_at on failure: a broken source must
            # keep being retried on its normal cadence (and once fixed, it
            # should still fetch everything since the last successful run),
            # not silently go quiet because we stamped a run that never
            # actually collected anything. Roll back and re-fetch the
            # Source row by id before writing the failure state, since a
            # raised commit() can leave the in-memory object's pending
            # changes rolled back and the object expired (mirrors
            # feed.stages.base.run_stage's rollback/re-fetch pattern).
            session.rollback()
            fresh = session.get(Source, src.id)
            error = f"{type(exc).__name__}: {exc}"
            if fresh is not None:
                fresh.consecutive_failures += 1
                fresh.last_error = error
                session.commit()
            result.source_errors[src.id] = error
            log.warning("source=%s fetch failed: %s", src.id, exc)
            continue

    return result
