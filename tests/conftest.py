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


@pytest.fixture(autouse=True)
def _block_real_gemini_network(monkeypatch):
    """Same guard, for the Gemini provider (feed/providers/gemini.py).

    complete() calls the module-level _post() seam rather than httpx
    directly so this can be monkeypatched cleanly. Tests exercising
    GeminiProvider must monkeypatch feed.providers.gemini._post instead of
    letting this raise.
    """
    def _boom(*args, **kwargs):
        raise AssertionError(
            "real network call (feed.providers.gemini._post) attempted during "
            "a test; monkeypatch feed.providers.gemini._post instead"
        )

    monkeypatch.setattr("feed.providers.gemini._post", _boom)


@pytest.fixture(autouse=True)
def _block_real_claude_cli(monkeypatch):
    """Same guard, for the Claude Code provider (feed/providers/claude_code.py).

    complete() calls the module-level _run_cli() seam rather than
    subprocess.run directly so this can be monkeypatched cleanly. Tests
    exercising ClaudeCodeProvider must monkeypatch
    feed.providers.claude_code._run_cli instead of letting this raise.
    Other subprocess.run call sites (e.g. tests/test_ci_workflow.py, which
    shells out to pytest itself, not to `claude`) are untouched -- this
    guard only replaces the provider's own seam function, not subprocess
    globally.
    """
    def _boom(*args, **kwargs):
        raise AssertionError(
            "real `claude` CLI invocation (feed.providers.claude_code._run_cli) "
            "attempted during a test; monkeypatch "
            "feed.providers.claude_code._run_cli instead"
        )

    monkeypatch.setattr("feed.providers.claude_code._run_cli", _boom)


@pytest.fixture(autouse=True)
def _block_real_openai_compatible_network(monkeypatch):
    """Same guard, for feed.providers.openai_compatible.OpenAICompatibleProvider
    (Groq/Mistral/OpenRouter/Cerebras all go through this one class)."""
    def _boom(*args, **kwargs):
        raise AssertionError(
            "real network call (feed.providers.openai_compatible._post) "
            "attempted during a test; monkeypatch "
            "feed.providers.openai_compatible._post instead"
        )

    monkeypatch.setattr("feed.providers.openai_compatible._post", _boom)


@pytest.fixture(autouse=True)
def _block_real_hackernews_network(monkeypatch):
    """Same guard, for feed.sources.hackernews.HackerNewsSource (Algolia HN
    Search API, A2). fetch() calls the module-level _get() seam rather than
    httpx directly so this can be monkeypatched cleanly. A slow-marked test
    that genuinely wants the live endpoint should monkeypatch this back to
    a real implementation itself (see tests/test_sources_more.py), not rely
    on this guard being absent.
    """
    def _boom(*args, **kwargs):
        raise AssertionError(
            "real network call (feed.sources.hackernews._get) attempted "
            "during a test; monkeypatch feed.sources.hackernews._get or "
            "use path= instead"
        )

    monkeypatch.setattr("feed.sources.hackernews._get", _boom)


@pytest.fixture(autouse=True)
def _block_real_arxiv_network(monkeypatch):
    """Same guard, for feed.sources.arxiv.ArxivSource (A1 pagination).
    fetch() calls the module-level _get() seam rather than httpx directly
    so this can be monkeypatched cleanly. A slow-marked test that genuinely
    wants the live API should monkeypatch this back to a real
    implementation itself, not rely on this guard being absent.
    """
    def _boom(*args, **kwargs):
        raise AssertionError(
            "real network call (feed.sources.arxiv._get) attempted during "
            "a test; monkeypatch feed.sources.arxiv._get or use path=/"
            "paths= instead"
        )

    monkeypatch.setattr("feed.sources.arxiv._get", _boom)


@pytest.fixture(autouse=True)
def _block_real_scraper_network(monkeypatch):
    """Same guard, for feed.sources.scraper.ScraperSource. fetch() (and the
    robots.txt check it makes first) calls the module-level _get_text() seam
    rather than httpx directly so this can be monkeypatched cleanly. A
    slow-marked test that genuinely wants the live site should monkeypatch
    this back to a real implementation itself, not rely on this guard being
    absent.
    """
    def _boom(*args, **kwargs):
        raise AssertionError(
            "real network call (feed.sources.scraper._get_text) attempted "
            "during a test; monkeypatch feed.sources.scraper._get_text or "
            "use path= instead"
        )

    monkeypatch.setattr("feed.sources.scraper._get_text", _boom)


@pytest.fixture(autouse=True)
def _no_real_retry_sleep(monkeypatch):
    """feed.providers._retry.call_with_retry() backs off with a real
    time.sleep() between attempts in production. Left un-mocked, a test
    exercising a couple of retries would genuinely sleep for
    backoff_base*(2**attempt) seconds each time -- this makes every such
    test instant without needing each test to remember to patch it.
    """
    monkeypatch.setattr("feed.providers._retry._sleep", lambda seconds: None)


@pytest.fixture
def session() -> Session:
    engine = make_engine("sqlite://")   # in-memory
    create_all(engine)
    factory = make_session_factory(engine)
    with factory() as s:
        yield s
