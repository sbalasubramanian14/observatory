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
    # A publisher-supplied lead image, if the source's own feed carried one
    # (RSS <media:content>/<media:thumbnail>/<enclosure type="image/*">,
    # spec D0). None here does NOT mean "no image exists" -- it means this
    # source plugin didn't find one in its own feed; the normalize stage
    # falls back to the article page's og:image/twitter:image meta tag.
    image_url: str | None = None


class Source(Protocol):
    id: str

    def fetch(self, since: datetime | None) -> Iterable[RawItem]: ...


def extract_feed_image(entry: dict) -> str | None:
    """Pull a lead image URL out of a feedparser entry, in priority order:
    Media RSS <media:content>, <media:thumbnail>, then a plain <enclosure>
    whose type is image/*. Returns None if the entry carries none of these
    -- the normal case for most feeds, and NOT an error; callers fall back
    to the og:image scrape in feed.stages.normalize.

    feedparser exposes these as (verified against feedparser 6.0.14):
      entry.media_content   -> [{"url": ..., "medium": "image", ...}, ...] | None
      entry.media_thumbnail -> [{"url": ...}, ...] | None
      entry.enclosures      -> [{"href": ..., "type": "image/jpeg"}, ...]
    """
    media_content = entry.get("media_content") or []
    for m in media_content:
        url = m.get("url")
        if not url:
            continue
        medium = m.get("medium")
        mtype = (m.get("type") or "").lower()
        if medium == "image" or mtype.startswith("image/") or (not medium and not mtype):
            return url

    media_thumbnail = entry.get("media_thumbnail") or []
    for m in media_thumbnail:
        url = m.get("url")
        if url:
            return url

    for enc in entry.get("enclosures") or []:
        etype = (enc.get("type") or "").lower()
        href = enc.get("href") or enc.get("url")
        if href and etype.startswith("image/"):
            return href

    return None


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
