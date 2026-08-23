from __future__ import annotations
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import feedparser
import httpx
from feed.sources.base import RawItem, canonical_url
from feed.sources.registry import register

log = logging.getLogger(__name__)


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
        # Spec A3: set by fetch() when this run's coverage looks suspect --
        # feed.stages.collect.collect() reads this optional attribute after
        # fully consuming fetch()'s generator and persists it onto the
        # Source row / sources.json. None (the default here, and after
        # every clean fetch) means nothing was flagged.
        self.coverage_warning: str | None = None

    def _raw(self) -> bytes:
        if self.path:
            return Path(self.path).read_bytes()
        resp = httpx.get(self.url, timeout=self.timeout,
                          headers={"User-Agent": "feed/0.1 (personal reader)"},
                          follow_redirects=True)
        resp.raise_for_status()
        return resp.content

    def fetch(self, since: datetime | None) -> Iterable[RawItem]:
        self.coverage_warning = None
        parsed = feedparser.parse(self._raw())
        dated: list[datetime] = []
        for i, entry in enumerate(parsed.entries):
            link = entry.get("link")
            if not link:
                identifier = (entry.get("title") or "").strip() or f"entry #{i}"
                log.warning("rss: entry with no link, skipping: %s", identifier)
                continue
            published = None
            tm = entry.get("published_parsed") or entry.get("updated_parsed")
            if tm:
                # feedparser normalises all dates to UTC struct_time.
                published = datetime(*tm[:6], tzinfo=timezone.utc)
                dated.append(published)
            if since is not None and published is not None and published <= since:
                continue
            yield RawItem(
                url=canonical_url(link),
                title=(entry.get("title") or "").strip(),
                summary=(entry.get("summary") or None),
                published_at=published,
            )

        # Spec A3: a feed document only ever carries the publisher's last N
        # entries -- no amount of pagination fixes that (there is none to
        # do). What CAN be detected: if every dated entry this fetch saw
        # postdates `since`, none of them overlap with the last run at all,
        # which means the feed's window most likely rolled entirely past
        # `since` between runs -- whatever was published in between is
        # gone. A mix (some entries <= since) means the window still
        # reaches back far enough; that's the normal, non-lossy case and
        # must stay silent. Skipped on a first run (since is None, there is
        # no gap to compare against) and on an empty/all-undated feed
        # (nothing to judge truncation from).
        if since is not None and dated and all(p > since for p in dated):
            self.coverage_warning = (
                f"rss source={self.id!r}: all {len(dated)} dated entries in "
                f"this fetch postdate since={since.isoformat()} -- the "
                f"feed's window may have rolled past older items between "
                f"runs; coverage may be incomplete"
            )
            log.warning(self.coverage_warning)
