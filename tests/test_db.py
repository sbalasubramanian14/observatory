from sqlalchemy import text
from feed.db import create_all, make_engine


def test_non_sqlite_engine_skips_pragmas():
    """The connect-event pragma hook must only fire for sqlite.

    We can't spin up a real Postgres server here, so we build a real
    sqlite:// engine (so PRAGMA foreign_keys / PRAGMA journal_mode are
    valid statements) and then relabel its dialect as "postgresql" to
    force the guard's non-sqlite branch. If the guard is doing its job,
    connecting must NOT touch foreign_keys or journal_mode, so both
    stay at SQLite's untouched defaults (foreign_keys=0, journal_mode
    != "wal").
    """
    engine = make_engine("sqlite://")
    engine.dialect.name = "postgresql"

    with engine.connect() as conn:
        fk = conn.exec_driver_sql("PRAGMA foreign_keys").scalar()
        journal_mode = conn.exec_driver_sql("PRAGMA journal_mode").scalar()

    assert fk == 0
    assert journal_mode != "wal"


def test_create_all_migrates_a_preexisting_story_table_missing_phase2_columns():
    """Reproduces the real feed.db from Phase 1: a `story` table that
    predates the Phase 2 enrichment columns. create_all() must add them
    additively (ALTER TABLE ADD COLUMN) without touching existing rows --
    dropping and recreating the table would lose the 377 real stories
    already in it.
    """
    engine = make_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE story ("
            "id INTEGER PRIMARY KEY, title TEXT NOT NULL, kind VARCHAR(32), "
            "first_seen DATETIME NOT NULL, updated_at DATETIME NOT NULL, "
            "item_count INTEGER NOT NULL, outlet_count INTEGER NOT NULL, "
            "score FLOAT, score_breakdown JSON, centroid BLOB)"
        ))
        conn.execute(text(
            "INSERT INTO story (id, title, kind, first_seen, updated_at, "
            "item_count, outlet_count) VALUES "
            "(1, 'Old story', NULL, '2026-01-01T00:00:00+00:00', "
            "'2026-01-01T00:00:00+00:00', 1, 1)"
        ))

    create_all(engine)  # must not raise, must not touch the existing row
    create_all(engine)  # idempotent: a second call must not re-ALTER and blow up

    with engine.connect() as conn:
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(story)")}
        assert {"summary", "category", "analysis", "analysis_provider",
                "analyzed_at", "status"} <= cols
        title, summary = conn.exec_driver_sql(
            "SELECT title, summary FROM story WHERE id=1"
        ).fetchone()
        assert title == "Old story"
        assert summary is None
