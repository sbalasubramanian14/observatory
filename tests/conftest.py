import pytest
from sqlalchemy.orm import Session
from feed.db import make_engine, make_session_factory, create_all

@pytest.fixture
def session() -> Session:
    engine = make_engine("sqlite://")   # in-memory
    create_all(engine)
    factory = make_session_factory(engine)
    with factory() as s:
        yield s
