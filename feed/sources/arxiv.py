from __future__ import annotations
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import feedparser
import httpx
from feed.sources.base import RawItem
from feed.sources.registry import register

_ABS = re.compile(r"arxiv\.org/abs/(?P<id>[\d.]+?)(?:v\d+)?$")

log = logging.getLogger(__name__)


def _get(url: str, *, timeout: float) -> bytes:
    """The real-network seam. Tests must monkeypatch this (or use
    path=/paths= fixture loading, which never calls this), never let it
    run for real -- see tests/conftest.py's autouse guard.
    """
    resp = httpx.get(url, timeout=timeout,
                      headers={"User-Agent": "feed/0.1 (personal reader)"})
    resp.raise_for_status()
    return resp.content


@register("arxiv")
class ArxivSource:
    """arXiv Atom API, paginated. Version suffixes are stripped so v1 and
    v2 of the same paper collapse to one URL and therefore one item.

    A1/diagnosis: a single request capped at `max_results` (e.g. 60) loses
    the vast majority of a multi-day backlog -- cs.AI+cs.CL+cs.LG alone
    produce ~400 papers/day, so a 2-day gap has ~800 papers waiting and
    only the newest `max_results` of them were ever fetched. fetch() now
    pages through `start=` offsets (results are requested
    sortBy=submittedDate&sortOrder=descending, so pages get strictly
    older) until either an entry is reached that is not newer than
    `since` -- which, since collect() always computes `since` via the A4
    backfill cap, is also where the cap is honoured -- or the
    `max_pages` safety cap is hit (a defensive bound in case `since` is
    None or the feed misbehaves), or the API itself runs out of results.

    Rate limit: arXiv's API asks for ~3 seconds between requests. A real
    (non-fixture) multi-page fetch sleeps `rate_limit_seconds` (default
    3.0) between page requests; fixture-backed fetches (path=/paths=)
    never sleep, since they never call the network.
    """

    API = "https://export.arxiv.org/api/query"

    def __init__(self, source_id: str, categories: list[str] | None = None,
                 max_results: int = 100, path: str | None = None,
                 paths: list[str] | None = None, timeout: float = 60.0,
                 max_pages: int = 20, rate_limit_seconds: float = 3.0):
        self.id = source_id
        self.categories = categories or ["cs.AI", "cs.CL", "cs.LG"]
        self.max_results = max_results
        self.path = path
        # `paths`: an ordered list of fixture files, one per page, for
        # testing pagination without the network -- `path` (singular)
        # remains the old single-fixture, single-page mechanism and is
        # unaffected.
        self.paths = paths
        self.timeout = timeout
        self.max_pages = max_pages
        self.rate_limit_seconds = rate_limit_seconds

    def _is_live(self) -> bool:
        return self.path is None and self.paths is None

    def _fetch_page(self, start: int) -> bytes | None:
        """Returns None to mean "no more pages" (fixture list exhausted,
        or the single-file `path=` fixture already served its one page)."""
        if self.paths is not None:
            idx = start // self.max_results
            if idx >= len(self.paths):
                return None
            return Path(self.paths[idx]).read_bytes()
        if self.path is not None:
            if start > 0:
                return None
            return Path(self.path).read_bytes()
        query = "+OR+".join(f"cat:{c}" for c in self.categories)
        url = (f"{self.API}?search_query={query}&start={start}"
               f"&max_results={self.max_results}&sortBy=submittedDate"
               f"&sortOrder=descending")
        return _get(url, timeout=self.timeout)

    def fetch(self, since: datetime | None) -> Iterable[RawItem]:
        live = self._is_live()
        start = 0
        page_num = 0
        while True:
            raw = self._fetch_page(start)
            if raw is None:
                break
            entries = feedparser.parse(raw).entries
            if not entries:
                break

            reached_since = False
            for entry in entries:
                raw_id = entry.get("id", "")
                match = _ABS.search(raw_id)
                if not match:
                    log.warning("arxiv: unparseable entry id, skipping: %r", raw_id)
                    continue
                url = f"https://arxiv.org/abs/{match.group('id')}"
                tm = entry.get("published_parsed") or entry.get("updated_parsed")
                published = datetime(*tm[:6], tzinfo=timezone.utc) if tm else None
                if since is not None and published is not None and published <= since:
                    # Results are sortBy=submittedDate&sortOrder=descending,
                    # so every remaining entry on this page and every later
                    # page is at least this old too -- stop entirely rather
                    # than skip item by item.
                    reached_since = True
                    break
                yield RawItem(
                    url=url,
                    title=" ".join((entry.get("title") or "").split()),
                    summary=" ".join((entry.get("summary") or "").split()) or None,
                    published_at=published,
                )

            page_num += 1
            if reached_since:
                break
            if len(entries) < self.max_results:
                break  # fewer than requested: this was the last page
            if page_num >= self.max_pages:
                log.warning(
                    "arxiv: source=%s hit max_pages=%d cap while paginating; "
                    "more results may exist and were not fetched",
                    self.id, self.max_pages,
                )
                break
            start += self.max_results
            if live:
                time.sleep(self.rate_limit_seconds)
