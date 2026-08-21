from feed.db import make_engine


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
