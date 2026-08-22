from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Protocol
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

TRACKING_PREFIXES = ("utm_",)
TRACKING_EXACT = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid", "ref_src"})


@dataclass(slots=True)
class RawItem:
    url: str
    title: str
    summary: str | None = None
    published_at: datetime | None = None
    outbound_links: list[str] = field(default_factory=list)


class Source(Protocol):
    id: str

    def fetch(self, since: datetime | None) -> Iterable[RawItem]: ...


def canonical_url(url: str) -> str:
    """Normalise a URL so the same article from two places hashes identically.

    Forces https, lowercases the host, strips a leading "www.", strips a
    trailing slash from the path (but keeps the root path as "/"), and
    drops tracking query params while keeping the rest (sorted, so the
    same set of legitimate params always canonicalises the same way
    regardless of the order they arrived in).
    """
    parts = urlsplit(url.strip())
    scheme = "https"
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parts.path.rstrip("/") or "/"
    kept = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith(TRACKING_PREFIXES)
        and k.lower() not in TRACKING_EXACT
    ]
    query = urlencode(sorted(kept))
    return urlunsplit((scheme, netloc, path, query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()
