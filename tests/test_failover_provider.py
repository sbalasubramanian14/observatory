import pytest
from feed.db import create_all, make_engine, make_session_factory
from feed.providers.base import (
    PaymentRequiredError,
    ProviderError,
    ProviderHealth,
    RateLimitError,
    Tier,
    TransientProviderError,
)
from feed.providers.failover import FailoverProvider
from feed.providers.health import ProviderHealthTracker


class _StubProvider:
    """Records calls, raises the given exception (or returns text)."""

    def __init__(self, name, model="m", *, raises=None, text="ok"):
        self.name = name
        self.model = model
        self.tier = Tier.BULK
        self._raises = raises
        self._text = text
        self.calls = 0

    def complete(self, prompt, *, schema=None):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._text

    def health(self):
        return ProviderHealth(healthy=True)


@pytest.fixture
def session_factory():
    engine = make_engine("sqlite://")
    create_all(engine)
    return make_session_factory(engine)


def test_first_provider_succeeds_others_untouched(session_factory):
    a = _StubProvider("a", text="from a")
    b = _StubProvider("b", text="from b")
    fp = FailoverProvider([a, b], session_factory=session_factory)

    result = fp.complete("prompt")

    assert result == "from a"
    assert a.calls == 1
    assert b.calls == 0
    assert fp.name == "a"
    assert fp.model == "m"


def test_429_advances_to_the_next_provider(session_factory):
    a = _StubProvider("a", raises=RateLimitError("a: 429"))
    b = _StubProvider("b", text="from b")
    fp = FailoverProvider([a, b], session_factory=session_factory)

    result = fp.complete("prompt")

    assert result == "from b"
    assert a.calls == 1
    assert b.calls == 1
    assert fp.name == "b"


def test_5xx_advances_to_the_next_provider(session_factory):
    a = _StubProvider("a", raises=TransientProviderError("a: 503"))
    b = _StubProvider("b", text="from b")
    fp = FailoverProvider([a, b], session_factory=session_factory)

    assert fp.complete("prompt") == "from b"


def test_generic_provider_error_advances_to_the_next_provider(session_factory):
    a = _StubProvider("a", raises=ProviderError("a: broke"))
    b = _StubProvider("b", text="from b")
    fp = FailoverProvider([a, b], session_factory=session_factory)

    assert fp.complete("prompt") == "from b"


def test_all_providers_failing_raises_with_every_provider_represented(session_factory):
    a = _StubProvider("a", raises=RateLimitError("a: 429 rate limited"))
    b = _StubProvider("b", raises=TransientProviderError("b: 503 server error"))
    c = _StubProvider("c", raises=ProviderError("c: 401 unauthorized"))
    fp = FailoverProvider([a, b, c], session_factory=session_factory)

    with pytest.raises(ProviderError) as exc_info:
        fp.complete("prompt")

    message = str(exc_info.value)
    # Requirement 1: "the record should show what each one said" -- every
    # enabled provider must have been tried, and its own message present.
    assert a.calls == 1 and b.calls == 1 and c.calls == 1
    assert "a: 429 rate limited" in message
    assert "b: 503 server error" in message
    assert "c: 401 unauthorized" in message


def test_a_402_disables_that_provider_for_the_day_and_it_is_skipped_next_call(session_factory):
    a = _StubProvider("a", raises=PaymentRequiredError("a: 402 payment required"))
    b = _StubProvider("b", text="from b")
    fp = FailoverProvider([a, b], session_factory=session_factory)

    fp.complete("prompt")
    assert a.calls == 1

    # Second call: `a` must be skipped without being invoked again -- its
    # daily health record is now disabled.
    fp.complete("prompt")
    assert a.calls == 1  # still 1, not called again
    assert b.calls == 2


def test_repeated_429s_disable_the_provider_after_the_threshold(session_factory):
    a = _StubProvider("a", raises=RateLimitError("a: 429"))
    b = _StubProvider("b", text="from b")
    fp = FailoverProvider([a, b], session_factory=session_factory,
                          rate_limit_disable_threshold=2)

    fp.complete("prompt")
    fp.complete("prompt")
    assert a.calls == 2  # tried both times, still under threshold after call 1

    fp.complete("prompt")
    assert a.calls == 2  # 3rd call: `a` now disabled after 2 consecutive 429s, skipped
    assert b.calls == 3


def test_disabled_provider_shows_up_in_the_failure_record_as_skipped(session_factory):
    a = _StubProvider("a", raises=PaymentRequiredError("a: 402"))
    fp = FailoverProvider([a], session_factory=session_factory)
    with pytest.raises(ProviderError):
        fp.complete("prompt")  # disables `a` as a side effect of failing

    with pytest.raises(ProviderError) as exc_info:
        fp.complete("prompt")

    assert a.calls == 1  # not called again
    assert "skipped" in str(exc_info.value)


def test_health_reflects_availability_of_any_provider(session_factory):
    a = _StubProvider("a", raises=PaymentRequiredError("a: 402"))
    fp = FailoverProvider([a], session_factory=session_factory)

    assert fp.health().healthy is True  # not yet disabled
    with pytest.raises(ProviderError):
        fp.complete("prompt")  # disables `a`, the only provider
    assert fp.health().healthy is False


def test_no_providers_configured_raises_a_clear_error(session_factory):
    fp = FailoverProvider([], session_factory=session_factory)

    with pytest.raises(ProviderError):
        fp.complete("prompt")


# --- Mutation check -------------------------------------------------------
# Not part of the suite (see task report / commit for the actual mutation
# run): the failover advancement above is proven non-vacuous by literally
# breaking feed/providers/failover.py's `except RateLimitError` branch to a
# `pass`-through that re-raises immediately without trying `b`, re-running
# this file, observing test_429_advances_to_the_next_provider fail, and
# restoring the source in the same command.
