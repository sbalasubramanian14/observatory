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


@runtime_checkable
class Provider(Protocol):
    name: str
    tier: Tier
    model: str

    def complete(self, prompt: str, *, schema: type | None = None) -> str: ...

    def health(self) -> ProviderHealth: ...
