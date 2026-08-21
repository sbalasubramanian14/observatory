from __future__ import annotations
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import feedparser
import httpx
from feed.sources.base import RawItem
from feed.sources.registry import register

_ABS = re.compile(r"arxiv\.org/abs/(?P<id>[\d.]+?)(?:v\d+)?$")

log = logging.getLogger(__name__)


@register("arxiv")
class ArxivSource:
    """arXiv Atom API. Version suffixes are stripped so v1 and v2 of the same
    paper collapse to one URL and therefore one item."""

    API = "https://export.arxiv.org/api/query"

    def __init__(self, source_id: str, categories: list[str] | None = None,
                 max_results: int = 100, path: str | None = None, timeout: float = 30.0):
        self.id = source_id
        self.categories = categories or ["cs.AI", "cs.CL", "cs.LG"]
        self.max_results = max_results
        self.path = path
        self.timeout = timeout

    def _raw(self) -> str | bytes:
        if self.path:
            return Path(self.path).read_bytes()
        query = "+OR+".join(f"cat:{c}" for c in self.categories)
        url = (f"{self.API}?search_query={query}&sortBy=submittedDate"
               f"&sortOrder=descending&max_results={self.max_results}")
        resp = httpx.get(url, timeout=self.timeout,
                         headers={"User-Agent": "feed/0.1 (personal reader)"})
        resp.raise_for_status()
        return resp.content

    def fetch(self, since: datetime | None) -> Iterable[RawItem]:
        parsed = feedparser.parse(self._raw())
        for entry in parsed.entries:
            raw_id = entry.get("id", "")
            match = _ABS.search(raw_id)
            if not match:
                log.warning("arxiv: unparseable entry id, skipping: %r", raw_id)
                continue
            url = f"https://arxiv.org/abs/{match.group('id')}"
            tm = entry.get("published_parsed") or entry.get("updated_parsed")
            published = datetime(*tm[:6], tzinfo=timezone.utc) if tm else None
            if since is not None and published is not None and published <= since:
                continue
            yield RawItem(
                url=url,
                title=" ".join((entry.get("title") or "").split()),
                summary=" ".join((entry.get("summary") or "").split()) or None,
                published_at=published,
            )
