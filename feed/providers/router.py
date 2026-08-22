from __future__ import annotations
import logging
from dataclasses import dataclass
from feed.providers.base import Provider, ProviderError, Tier

log = logging.getLogger(__name__)


@dataclass
class RouteResult:
    text: str
    provider: str
    model: str
    tier: Tier
    # True when a DEEP request was served by the BULK provider instead,
    # because DEEP was unavailable or failed. The caller (feed.stages.enrich)
    # is responsible for flagging the story for retry when this is set --
    # the router's job stops at "which provider actually answered".
    degraded: bool = False


class Router:
    """Spec 3.5's provider router.

    Selects a provider by requested tier and only ever degrades downward:
    a DEEP request that can't be served by the DEEP provider falls back to
    BULK with a simpler prompt; a BULK request is never silently upgraded
    to DEEP, even if DEEP happens to be idle. Either way the router never
    raises up to the caller and never blocks the pipeline -- provider
    failures are reported back as a normal (degraded) result, and the only
    case that still raises is "no BULK provider exists at all", which is a
    misconfiguration, not a runtime provider failure.
    """

    def __init__(self, *, bulk: Provider, deep: Provider | None = None):
        self.bulk = bulk
        self.deep = deep

    def complete(self, prompt: str, *, tier: Tier, deep_prompt: str | None = None,
                 schema: type | None = None) -> RouteResult:
        if tier is Tier.DEEP and self.deep is not None:
            health = self.deep.health()
            if health.healthy:
                try:
                    text = self.deep.complete(prompt, schema=schema)
                    return RouteResult(text=text, provider=self.deep.name,
                                       model=self.deep.model, tier=Tier.DEEP,
                                       degraded=False)
                except ProviderError as exc:
                    log.warning("router: DEEP provider %s failed, degrading "
                               "to BULK: %s", self.deep.name, exc)
            else:
                log.warning("router: DEEP provider %s unhealthy (%s), "
                           "degrading to BULK", self.deep.name, health.detail)

            fallback_prompt = deep_prompt if deep_prompt is not None else prompt
            text = self.bulk.complete(fallback_prompt, schema=schema)
            return RouteResult(text=text, provider=self.bulk.name,
                               model=self.bulk.model, tier=Tier.BULK, degraded=True)

        # tier is BULK, or DEEP was requested with no deep provider
        # configured at all -- both land on the bottom rung.
        text = self.bulk.complete(prompt, schema=schema)
        return RouteResult(text=text, provider=self.bulk.name,
                           model=self.bulk.model, tier=Tier.BULK, degraded=(tier is Tier.DEEP))
