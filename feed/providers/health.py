from __future__ import annotations
from datetime import datetime, timezone
from typing import Callable
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.models import ProviderStatus

# Requirement 2: "A provider returning repeated 429s ... is skipped for
# the remainder of the day rather than retried into the ground." Three
# consecutive 429s (not three total -- one success resets the streak) is
# the threshold; a single 402 disables immediately regardless of this
# constant (see record_payment_required).
DEFAULT_RATE_LIMIT_DISABLE_THRESHOLD = 3


class ProviderHealthTracker:
    """Persists per-provider, per-UTC-day quota/health state to
    ProviderStatus (requirement 2: "persisted in the database so it
    survives restarts"). Used by feed.providers.failover.FailoverProvider
    to decide, before spending a network call, whether a provider is
    already known to be dead for today.

    Each method opens its own transaction (commits immediately) so health
    state is never lost to an unrelated rollback elsewhere in the caller's
    session -- e.g. feed.stages.enrich rolling back a story's own field
    changes on a JSON-parse failure must not also undo a health record
    that was legitimately written moments earlier in the same session.
    """

    def __init__(self, session: Session, *,
                rate_limit_disable_threshold: int = DEFAULT_RATE_LIMIT_DISABLE_THRESHOLD,
                now: Callable[[], datetime] | None = None):
        self.session = session
        self.rate_limit_disable_threshold = rate_limit_disable_threshold
        self._now = now or (lambda: datetime.now(timezone.utc))

    def close(self) -> None:
        """Lets feed.providers.failover.FailoverProvider wrap a tracker in
        contextlib.closing() rather than reaching into `.session` itself."""
        self.session.close()

    def _today(self) -> str:
        return self._now().strftime("%Y-%m-%d")

    def _get_or_create(self, provider: str) -> ProviderStatus:
        day = self._today()
        row = self.session.scalar(
            select(ProviderStatus).where(
                ProviderStatus.provider == provider, ProviderStatus.day == day
            )
        )
        if row is None:
            row = ProviderStatus(provider=provider, day=day)
            self.session.add(row)
            self.session.flush()
        return row

    def is_disabled_today(self, provider: str) -> bool:
        return self._get_or_create(provider).disabled

    def status_today(self, provider: str) -> ProviderStatus:
        """Read-only snapshot for the `feed providers` CLI command."""
        return self._get_or_create(provider)

    def record_success(self, provider: str) -> None:
        row = self._get_or_create(provider)
        row.requests += 1
        row.successes += 1
        row.consecutive_429 = 0
        row.last_used_at = self._now()
        self.session.commit()

    def record_rate_limit(self, provider: str, detail: str = "") -> None:
        row = self._get_or_create(provider)
        row.requests += 1
        row.failures += 1
        row.consecutive_429 += 1
        row.last_error = detail
        row.last_used_at = self._now()
        if row.consecutive_429 >= self.rate_limit_disable_threshold:
            row.disabled = True
            row.disabled_reason = (
                f"{row.consecutive_429} consecutive 429s today"
            )
        self.session.commit()

    def record_payment_required(self, provider: str, detail: str = "") -> None:
        row = self._get_or_create(provider)
        row.requests += 1
        row.failures += 1
        row.disabled = True
        row.disabled_reason = detail or "402 payment required"
        row.last_error = detail
        row.last_used_at = self._now()
        self.session.commit()

    def record_failure(self, provider: str, detail: str = "") -> None:
        """Any failure that is neither a 429 nor a 402 (e.g. a 5xx/timeout/
        connection error that survived retry, or a bad-key/malformed
        response). Resets the 429 streak -- only *consecutive* 429s should
        trip the quota disable, and this failure was not one."""
        row = self._get_or_create(provider)
        row.requests += 1
        row.failures += 1
        row.consecutive_429 = 0
        row.last_error = detail
        row.last_used_at = self._now()
        self.session.commit()
