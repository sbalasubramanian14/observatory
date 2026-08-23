from __future__ import annotations
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from feed.models import Base

def make_engine(url: str) -> Engine:
    engine = create_engine(url, future=True)

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        if engine.dialect.name != "sqlite":
            return
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    return engine

def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)

# Additive-only migration for columns Phase 2 added to `story`. create_all()
# only creates missing TABLES, never missing COLUMNS on a table that already
# exists -- and the real feed.db from Phase 1 already has a `story` table
# (377 rows) without these. Without this, `feed init` against that db would
# leave the new columns entirely absent and every enrich/publish query would
# blow up with "no such column". Plain ALTER TABLE ADD COLUMN is additive and
# safe against existing rows (they read back NULL / the SQLAlchemy default
# on next write); there is no Alembic dependency in this project, and one
# more table's worth of columns is not worth introducing one for.
_STORY_NEW_COLUMNS: dict[str, str] = {
    "summary": "TEXT",
    "category": "VARCHAR(64)",
    "analysis": "TEXT",
    "analysis_provider": "VARCHAR(128)",
    "analyzed_at": "DATETIME",
    "status": "VARCHAR(16)",
    # Added for the multi-provider router (Tier 1 provenance, mirroring
    # analysis_provider). Same additive-migration reasoning as above: an
    # existing feed.db's `story` table predates this column.
    "summary_provider": "VARCHAR(128)",
}

# Same additive-migration reasoning, for columns the phaseA collect-stage
# fix (backfill cap + coverage-loss visibility) added to `source`: an
# existing feed.db's `source` table predates them.
_SOURCE_NEW_COLUMNS: dict[str, str] = {
    "max_backfill_days": "INTEGER",
    "coverage_warning": "TEXT",
    # Added for the source catalogue / `feed sources sync` (spec 2's four
    # coverage territories). Same additive-migration reasoning as above.
    "territory": "VARCHAR(32)",
    # A1-followup: cross-run politeness-delay persistence for rate-limited
    # source plugins (see feed.models.Source.last_request_at's docstring).
    # Same additive-migration reasoning as above.
    "last_request_at": "DATETIME",
}

# Phase D (lead images): `item.image_url`. Same additive-migration
# reasoning as above -- an existing feed.db's `item` table predates this
# column. Unlike _STORY_NEW_COLUMNS's `status` (whose Python-side default
# of StoryStatus.NEW is silently NOT applied to existing rows by a plain
# ALTER TABLE ADD COLUMN -- existing rows read back NULL, not "new", which
# then falls out of feed.stages.enrich's `Story.status == StoryStatus.NEW`
# filter and gets silently skipped forever), NULL is the semantically
# correct value here for every pre-existing row: "no lead image known for
# this item" is not a wrong default that needs backfilling, it is the
# truth (the source feeds these items came from were never asked for an
# image, and normalize() only fills this in for items it processes going
# forward). No UPDATE needed.
_ITEM_NEW_COLUMNS: dict[str, str] = {
    "image_url": "TEXT",
    # Phase D-images: records when the og:image fallback last attempted a
    # fetch for this item (see feed.models.Item.image_checked_at's
    # docstring). Same additive-migration reasoning as image_url above:
    # existing rows read back NULL, which is exactly "never attempted" --
    # the correct value, not a wrong default needing a backfill UPDATE.
    "image_checked_at": "DATETIME",
    # Relevance gate (see feed.models.Item.reject_reason's docstring).
    # Same additive-migration reasoning: NULL is exactly the correct value
    # for every pre-existing row ("never rejected"), not a placeholder
    # needing a backfill UPDATE.
    "reject_reason": "TEXT",
}


def _add_missing_columns(conn, table: str, columns: dict[str, str]) -> None:
    existing = {
        row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")
    }
    for column, ddl_type in columns.items():
        if column not in existing:
            conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


def _migrate_sqlite(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return
    with engine.begin() as conn:
        tables = {
            row[0] for row in conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "story" in tables:
            _add_missing_columns(conn, "story", _STORY_NEW_COLUMNS)
        if "source" in tables:
            _add_missing_columns(conn, "source", _SOURCE_NEW_COLUMNS)
        if "item" in tables:
            _add_missing_columns(conn, "item", _ITEM_NEW_COLUMNS)


def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    _migrate_sqlite(engine)
