import pytest
from sqlalchemy.orm import Session
from feed.db import make_engine, make_session_factory, create_all


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch):
    """Suite-wide guard: no test, in this file or any other, may reach the
    real network via trafilatura.fetch_url.

    feed.stages.normalize._extract() falls back to trafilatura.fetch_url()
    for non-arXiv items with no usable summary. That fallback is only
    avoided by luck in the normalize tests we happen to have today; a
    future test (e.g. an end-to-end pipeline test seeding a non-arXiv item
    with a short summary and calling normalize()) could otherwise hit the
    real internet with no guard, from any test file. This fixture makes
    that structurally impossible: any real call raises loudly instead of
    making an HTTP request. Tests that need the fallback path exercised
    should monkeypatch feed.stages.normalize._fetch_remote_text (the
    seam), not trafilatura.fetch_url directly.
    """
    def _boom(*args, **kwargs):
        raise AssertionError(
            "real network fetch (trafilatura.fetch_url) attempted during a test; "
            "monkeypatch feed.stages.normalize._fetch_remote_text instead"
        )

    monkeypatch.setattr("trafilatura.fetch_url", _boom)


@pytest.fixture
def session() -> Session:
    engine = make_engine("sqlite://")   # in-memory
    create_all(engine)
    factory = make_session_factory(engine)
    with factory() as s:
        yield s
