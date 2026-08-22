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


def _migrate_sqlite(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return
    with engine.begin() as conn:
        tables = {
            row[0] for row in conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "story" not in tables:
            return
        existing = {
            row[1] for row in conn.exec_driver_sql("PRAGMA table_info(story)")
        }
        for column, ddl_type in _STORY_NEW_COLUMNS.items():
            if column not in existing:
                conn.exec_driver_sql(f"ALTER TABLE story ADD COLUMN {column} {ddl_type}")


def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    _migrate_sqlite(engine)
