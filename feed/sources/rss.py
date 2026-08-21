from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import feedparser
import httpx
from feed.sources.base import RawItem, canonical_url
from feed.sources.registry import register


@register("rss")
class RssSource:
    """Generic RSS/Atom source.

    `url` fetches over HTTP; `path` reads a local file and exists so tests
    never touch the network.
    """

    def __init__(self, source_id: str, url: str | None = None, path: str | None = None,
                 timeout: float = 20.0):
        if not url and not path:
            raise ValueError("rss source needs either url or path")
        self.id = source_id
        self.url = url
        self.path = path
        self.timeout = timeout

    def _raw(self) -> bytes:
        if self.path:
            return Path(self.path).read_bytes()
        resp = httpx.get(self.url, timeout=self.timeout,
                          headers={"User-Agent": "feed/0.1 (personal reader)"},
                          follow_redirects=True)
        resp.raise_for_status()
        return resp.content

    def fetch(self, since: datetime | None) -> Iterable[RawItem]:
        parsed = feedparser.parse(self._raw())
        for entry in parsed.entries:
            link = entry.get("link")
            if not link:
                continue
            published = None
            tm = entry.get("published_parsed") or entry.get("updated_parsed")
            if tm:
                # feedparser normalises all dates to UTC struct_time.
                published = datetime(*tm[:6], tzinfo=timezone.utc)
            if since is not None and published is not None and published <= since:
                continue
            yield RawItem(
                url=canonical_url(link),
                title=(entry.get("title") or "").strip(),
                summary=(entry.get("summary") or None),
                published_at=published,
            )
