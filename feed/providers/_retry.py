from __future__ import annotations
import time
from typing import Callable, TypeVar
from feed.providers.base import TransientProviderError

T = TypeVar("T")


def _sleep(seconds: float) -> None:
    """The real-delay seam. Tests must monkeypatch this to a no-op rather
    than let the suite actually sleep -- see tests/conftest.py."""
    time.sleep(seconds)


def call_with_retry(
    fn: Callable[[], T], *, max_retries: int = 2, backoff_base: float = 0.5
) -> T:
    """Bounded retry with exponential backoff for TRANSIENT failures only
    (spec requirement 5: "Bounded retry with backoff for transient
    failures before falling through to the next provider" -- previously a
    single Gemini 503 wasted a story's whole attempt for the run).

    Deliberately narrow: only TransientProviderError (5xx / timeout /
    connection error) is retried here. RateLimitError and
    PaymentRequiredError propagate immediately -- retrying the same
    provider into a live rate limit or a billing block cannot help, and
    the failover chain (feed.providers.failover.FailoverProvider) is what
    advances to the next provider for those. Any other exception (a bad
    API key, an unparseable response, ...) also propagates immediately;
    retrying is only ever a bet on transient infrastructure trouble.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except TransientProviderError:
            if attempt >= max_retries:
                raise
            _sleep(backoff_base * (2**attempt))
            attempt += 1
