# feed/stages/sync.py
"""Reconciles the `source` table against a parsed sources.catalogue.toml
(feed.catalogue.CatalogueEntry list). This is the mechanism behind `feed
sources sync`: the catalogue file is the source of truth, and calling this
makes the database match it.

Rules:
  - A catalogue id absent from the DB is inserted (added).
  - A catalogue id present in the DB has its plugin/config/cadence/
    authority/territory/max_backfill_days overwritten to match the
    catalogue (updated) -- and, only when the plugin or config actually
    changed, or the row was previously disabled, its stale health state
    (last_error/consecutive_failures/coverage_warning) is cleared, since
    that state describes a configuration that no longer exists. A sync
    that re-declares an already-correct, already-enabled entry unchanged
    leaves its failure history alone (unchanged) -- sync is not a
    "clear all errors" button.
  - A catalogue entry with enabled=false is written with enabled=False
    and is not counted as a removal even though the DB row differs.
  - A DB source id absent from the catalogue entirely is removed: deleted
    outright if it never collected any items (safe -- nothing references
    it), otherwise disabled and left alone (preserves Item.source_id
    history/FK integrity). This is what turns "delete two dead RSS rows
    from the catalogue" into "they stop appearing on the health page"
    rather than leaving them permanently marked FAILING.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from feed.catalogue import CatalogueEntry
from feed.models import Item, Source


@dataclass
class SyncResult:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)


def sync_sources(session: Session, entries: list[CatalogueEntry]) -> SyncResult:
    result = SyncResult()
    catalogue_ids = {e.id for e in entries}
    existing = {s.id: s for s in session.scalars(select(Source))}

    for entry in entries:
        row = existing.get(entry.id)
        if row is None:
            session.add(Source(
                id=entry.id, plugin=entry.plugin, config=entry.config,
                cadence_minutes=entry.cadence_minutes, authority=entry.authority,
                max_backfill_days=entry.max_backfill_days, territory=entry.territory,
                enabled=entry.enabled,
            ))
            result.added.append(entry.id)
            continue

        config_changed = row.plugin != entry.plugin or (row.config or {}) != entry.config
        was_disabled = not row.enabled
        meaningfully_changed = (
            config_changed
            or row.cadence_minutes != entry.cadence_minutes
            or row.authority != entry.authority
            or row.max_backfill_days != entry.max_backfill_days
            or row.territory != entry.territory
            or row.enabled != entry.enabled
        )

        row.plugin = entry.plugin
        row.config = entry.config
        row.cadence_minutes = entry.cadence_minutes
        row.authority = entry.authority
        row.max_backfill_days = entry.max_backfill_days
        row.territory = entry.territory
        row.enabled = entry.enabled

        if config_changed or was_disabled:
            # Stale health state describes a config that no longer exists
            # (or a source that was not being run at all) -- carrying it
            # forward would misreport the newly-(re)configured source as
            # still failing before it has even run once.
            row.last_error = None
            row.consecutive_failures = 0
            row.coverage_warning = None

        if meaningfully_changed:
            result.updated.append(entry.id)
        else:
            result.unchanged.append(entry.id)

    for sid, row in existing.items():
        if sid in catalogue_ids:
            continue
        if not row.enabled:
            continue  # already inert, nothing to reconcile
        item_count = session.scalar(
            select(func.count()).select_from(Item).where(Item.source_id == sid)
        ) or 0
        if item_count == 0:
            session.delete(row)
            result.deleted.append(sid)
        else:
            row.enabled = False
            row.last_error = None
            row.consecutive_failures = 0
            row.coverage_warning = None
            result.disabled.append(sid)

    session.commit()
    return result
