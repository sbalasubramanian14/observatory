from __future__ import annotations
import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class Tier(enum.Enum):
    """Spec 3.5. Tier 0 (embedding/entities/clustering/scoring) has no LLM
    and lives entirely outside this package."""
    BULK = "bulk"
    DEEP = "deep"


@dataclass
class ProviderHealth:
    healthy: bool
    detail: str = ""


class ProviderError(Exception):
    """Raised when a provider fails to produce a completion (bad/missing
    key, non-2xx response, non-zero exit code, timeout, unparseable
    response, ...). Callers treat this as "this provider is unusable right
    now", never as a crash -- see feed.providers.router.Router and
    feed.stages.enrich's per-story failure isolation.
    """


class RateLimitError(ProviderError):
    """429 Too Many Requests. The failover chain (feed.providers.failover)
    advances to the next provider immediately -- retrying the same
    provider into a live rate limit wastes the story's attempt -- and
    counts this against that provider's daily health record (see
    feed.providers.health.ProviderHealthTracker), which disables the
    provider for the rest of the day after enough consecutive 429s."""


class PaymentRequiredError(ProviderError):
    """402 Payment Required -- a hard quota/billing exhaustion, not a
    transient blip. Unlike a 429, a single 402 disables the provider for
    the remainder of the day (spec: "a hard 402, is skipped for the
    remainder of the day rather than retried into the ground")."""


class TransientProviderError(ProviderError):
    """5xx, timeout, or connection error. May well succeed on retry, so
    feed.providers._retry.call_with_retry gives it a bounded number of
    attempts with backoff before the failover chain advances to the next
    provider -- this is what stops a single transient 503 from wasting a
    story's whole attempt for the run (previously true of Gemini alone)."""


def raise_for_status_code(name: str, status_code: int, detail: str = "") -> None:
    """Classify an HTTP status code into the right ProviderError subclass
    and raise it. Shared by every HTTP-based provider (Gemini,
    OpenAICompatibleProvider) so failover/retry behaviour is identical
    across providers regardless of which one hit the error -- spec
    requirement 1's "On 429, any 5xx, timeout, or connection error ->
    advance to the next [provider]" must hold the same way no matter which
    provider produced the status code.
    """
    suffix = f": {detail}" if detail else ""
    if status_code == 429:
        raise RateLimitError(f"{name}: 429 rate limited{suffix}")
    if status_code == 402:
        raise PaymentRequiredError(f"{name}: 402 payment required{suffix}")
    if status_code >= 500:
        raise TransientProviderError(f"{name}: {status_code} server error{suffix}")
    raise ProviderError(f"{name}: {status_code}{suffix}")


@runtime_checkable
class Provider(Protocol):
    name: str
    tier: Tier
    model: str

    def complete(self, prompt: str, *, schema: type | None = None) -> str: ...

    def health(self) -> ProviderHealth: ...
