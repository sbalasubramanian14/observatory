from __future__ import annotations
from contextlib import closing
from typing import Callable
from sqlalchemy.orm import Session
from feed.providers.base import (
    PaymentRequiredError,
    Provider,
    ProviderError,
    ProviderHealth,
    RateLimitError,
    Tier,
)
from feed.providers.health import (
    DEFAULT_RATE_LIMIT_DISABLE_THRESHOLD,
    ProviderHealthTracker,
)


class FailoverProvider:
    """Requirement 1: try BULK providers in configured priority order,
    advancing to the next on 429, any 5xx, timeout, or connection error.
    Wraps a plain list of Provider objects and itself satisfies the
    Provider protocol (name/tier/model/complete/health), so it plugs into
    feed.providers.router.Router as a drop-in replacement for a single
    provider -- Router itself needed no changes.

    Bounded retry-with-backoff for transient failures (requirement 5)
    happens INSIDE each provider's own complete() (see
    feed.providers.gemini.GeminiProvider and
    feed.providers.openai_compatible.OpenAICompatibleProvider, both built
    on feed.providers._retry.call_with_retry) -- this class is only
    responsible for what happens once a provider has exhausted its own
    retries and still failed: record that against its persisted daily
    health, and move to the next one in priority order.

    `.name` / `.model` are mutated on every successful complete() to
    reflect whichever provider actually answered -- this is how
    feed.providers.router.Router's RouteResult ends up carrying correct
    provenance (requirement 4) despite Router reading `self.bulk.name` /
    `self.bulk.model` as plain attributes right after calling
    `self.bulk.complete(...)`, with no changes needed to Router itself.
    """

    tier = Tier.BULK

    def __init__(
        self,
        providers: list[Provider],
        *,
        session_factory: Callable[[], Session],
        rate_limit_disable_threshold: int = DEFAULT_RATE_LIMIT_DISABLE_THRESHOLD,
    ):
        self._providers = list(providers)
        self._session_factory = session_factory
        self.rate_limit_disable_threshold = rate_limit_disable_threshold
        first = self._providers[0] if self._providers else None
        self.name = first.name if first is not None else "none"
        self.model = first.model if first is not None else ""

    def _tracker(self) -> closing[ProviderHealthTracker]:
        # A fresh session per call, deliberately -- this must never share
        # (and be vulnerable to a rollback from) the caller's own session,
        # e.g. feed.stages.enrich rolling back a story's field changes
        # after a JSON-parse failure. See ProviderHealthTracker's docstring.
        # closing() guarantees the session is closed even if the block
        # raises, without a bespoke context-manager class for one line.
        return closing(ProviderHealthTracker(
            self._session_factory(),
            rate_limit_disable_threshold=self.rate_limit_disable_threshold,
        ))

    def complete(self, prompt: str, *, schema: type | None = None) -> str:
        if not self._providers:
            raise ProviderError("failover: no BULK providers configured")

        attempts: list[str] = []
        for provider in self._providers:
            with self._tracker() as tracker:
                if tracker.is_disabled_today(provider.name):
                    attempts.append(f"{provider.name}: skipped (disabled for today)")
                    continue

            try:
                text = provider.complete(prompt, schema=schema)
            except PaymentRequiredError as exc:
                with self._tracker() as tracker:
                    tracker.record_payment_required(provider.name, str(exc))
                attempts.append(f"{provider.name}: {exc}")
                continue
            except RateLimitError as exc:
                with self._tracker() as tracker:
                    tracker.record_rate_limit(provider.name, str(exc))
                attempts.append(f"{provider.name}: {exc}")
                continue
            except ProviderError as exc:
                with self._tracker() as tracker:
                    tracker.record_failure(provider.name, str(exc))
                attempts.append(f"{provider.name}: {exc}")
                continue

            with self._tracker() as tracker:
                tracker.record_success(provider.name)
            self.name = provider.name
            self.model = provider.model
            return text

        # Requirement 1: "A story is marked failed only after every ENABLED
        # provider has been tried, and the record should show what each one
        # said." feed.stages.enrich's per-story try/except records str(exc)
        # verbatim, so this message IS the record.
        raise ProviderError(
            f"failover: all {len(self._providers)} bulk provider(s) failed: "
            + "; ".join(attempts)
        )

    def health(self) -> ProviderHealth:
        for provider in self._providers:
            with self._tracker() as tracker:
                disabled = tracker.is_disabled_today(provider.name)
            if not disabled and provider.health().healthy:
                return ProviderHealth(healthy=True)
        return ProviderHealth(
            healthy=False, detail="all bulk providers disabled or unhealthy"
        )
