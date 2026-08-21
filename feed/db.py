from __future__ import annotations
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from feed.models import Base

def make_engine(url: str) -> Engine:
    engine = create_engine(url, future=True)

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        is_memory = engine.url.database in (None, "", ":memory:")
        if not is_memory:
            cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    return engine

def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)

def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)
