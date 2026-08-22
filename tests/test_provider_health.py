from datetime import datetime, timedelta, timezone
from feed.providers.health import ProviderHealthTracker


def test_new_provider_is_not_disabled(session):
    tracker = ProviderHealthTracker(session)
    assert tracker.is_disabled_today("groq") is False


def test_success_records_usage_and_resets_429_streak(session):
    tracker = ProviderHealthTracker(session)
    tracker.record_rate_limit("groq")
    tracker.record_success("groq")

    status = tracker.status_today("groq")
    assert status.successes == 1
    assert status.consecutive_429 == 0
    assert status.disabled is False


def test_consecutive_429s_disable_the_provider_for_the_day(session):
    tracker = ProviderHealthTracker(session, rate_limit_disable_threshold=3)
    tracker.record_rate_limit("groq")
    assert tracker.is_disabled_today("groq") is False
    tracker.record_rate_limit("groq")
    assert tracker.is_disabled_today("groq") is False
    tracker.record_rate_limit("groq")

    assert tracker.is_disabled_today("groq") is True
    status = tracker.status_today("groq")
    assert "3 consecutive 429s" in status.disabled_reason


def test_a_success_between_429s_resets_the_streak(session):
    tracker = ProviderHealthTracker(session, rate_limit_disable_threshold=2)
    tracker.record_rate_limit("groq")
    tracker.record_success("groq")
    tracker.record_rate_limit("groq")

    assert tracker.is_disabled_today("groq") is False  # streak was reset, only 1 now


def test_a_non_429_failure_resets_the_429_streak(session):
    tracker = ProviderHealthTracker(session, rate_limit_disable_threshold=2)
    tracker.record_rate_limit("groq")
    tracker.record_failure("groq", "500 server error")
    tracker.record_rate_limit("groq")

    assert tracker.is_disabled_today("groq") is False


def test_402_disables_immediately_on_a_single_occurrence(session):
    tracker = ProviderHealthTracker(session)
    tracker.record_payment_required("cerebras", "402 payment required")

    assert tracker.is_disabled_today("cerebras") is True
    status = tracker.status_today("cerebras")
    assert status.disabled_reason == "402 payment required"


def test_providers_are_tracked_independently(session):
    tracker = ProviderHealthTracker(session, rate_limit_disable_threshold=1)
    tracker.record_rate_limit("groq")

    assert tracker.is_disabled_today("groq") is True
    assert tracker.is_disabled_today("mistral") is False


def test_state_persists_across_tracker_instances_sharing_a_session(session):
    """Requirement 2: "persisted in the database so it survives restarts".
    A fresh ProviderHealthTracker (simulating a new process) reading the
    same underlying data must see prior state."""
    ProviderHealthTracker(session, rate_limit_disable_threshold=1).record_rate_limit("groq")

    fresh = ProviderHealthTracker(session)
    assert fresh.is_disabled_today("groq") is True


def test_a_new_utc_day_gets_a_fresh_record(session):
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    tracker_yesterday = ProviderHealthTracker(
        session, rate_limit_disable_threshold=1, now=lambda: yesterday,
    )
    tracker_yesterday.record_rate_limit("groq")
    assert tracker_yesterday.is_disabled_today("groq") is True

    tracker_today = ProviderHealthTracker(session)
    assert tracker_today.is_disabled_today("groq") is False
