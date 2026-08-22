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
        try:
            plugin = build_source(src.plugin, src.id, dict(src.config or {}))
            raw_items = list(plugin.fetch(since=src.last_run_at))
        except Exception as exc:
            # Do not advance last_run_at on failure: a broken source must
            # keep being retried on its normal cadence (and once fixed, it
            # should still fetch everything since the last successful run),
            # not silently go quiet because we stamped a run that never
            # actually collected anything.
            src.consecutive_failures += 1
            src.last_error = f"{type(exc).__name__}: {exc}"
            session.commit()
            result.source_errors[src.id] = src.last_error
            log.warning("source=%s fetch failed: %s", src.id, exc)
            continue

        for raw in raw_items:
            h = url_hash(raw.url)
            exists = session.scalar(select(Item.id).where(Item.url_hash == h))
            if exists:
                result.skipped_duplicates += 1
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
            result.new_items += 1

        src.last_run_at = now
        src.last_error = None
        src.consecutive_failures = 0
        session.commit()

    return result
