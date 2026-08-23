from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import httpx
from feed.sources.base import RawItem, canonical_url
from feed.sources.registry import register

log = logging.getLogger(__name__)


def _get(url: str, *, params: dict, timeout: float) -> dict:
    """The real-network seam. Tests must monkeypatch this, never let it run
    for real -- see tests/conftest.py's autouse guard. Mirrors
    feed.providers.gemini._post / feed.providers.openai_compatible._post.
    """
    resp = httpx.get(url, params=params, timeout=timeout,
                      headers={"User-Agent": "feed/0.1 (personal reader)"})
    resp.raise_for_status()
    return resp.json()


@register("hackernews")
class HackerNewsSource:
    """Hacker News stories above a score floor, fetched by time window.

    A3/diagnosis: `topstories.json` only ever returns whatever is
    top *right now* -- a story that peaked two days ago and has since
    scrolled off the front page is unreachable from that endpoint at any
    page size, no matter how quickly the pipeline runs. The Algolia HN
    Search API (https://hn.algolia.com/api/v1/search_by_date, no key
    required) is queryable by `created_at_i` and paginates via `page=` /
    `nbPages`, so it can actually answer "everything since my last run"
    instead of "whatever's popular this instant". Verified live
    (2026-08-22/23): https://hn.algolia.com/api/v1/search_by_date returns
    `hits` (each with `points`, `created_at_i`, `url`, `title`, `_tags`)
    plus `nbPages`/`page`/`hitsPerPage`, exactly as documented.

    The score floor is still the point: HN volume is enormous and
    low-score stories are noise the pipeline should never pay to embed --
    now enforced server-side via `points>=min_score` rather than
    client-side after the fact.
    """

    SEARCH = "https://hn.algolia.com/api/v1/search_by_date"

    def __init__(self, source_id: str, min_score: int = 100, limit: int = 100,
                 path: str | None = None, timeout: float = 20.0, max_pages: int = 10):
        self.id = source_id
        self.min_score = min_score
        self.limit = limit
        self.path = path
        self.timeout = timeout
        # Safety net, not the primary stopping condition: normal stopping
        # is "server says no more pages" (page >= nbPages) or "since
        # excludes everything left". This only guards against a
        # misbehaving response (e.g. nbPages absurdly large) turning into
        # runaway pagination.
        self.max_pages = max_pages

    def _query(self, since: datetime | None, page: int) -> dict:
        filters = [f"points>={self.min_score}"]
        if since is not None:
            filters.append(f"created_at_i>{int(since.timestamp())}")
        params = {
            "tags": "story",
            "numericFilters": ",".join(filters),
            "page": page,
            "hitsPerPage": self.limit,
        }
        return _get(self.SEARCH, params=params, timeout=self.timeout)

    def _stories(self, since: datetime | None) -> list[dict]:
        if self.path:
            # Old fixture mechanism, kept exactly as before: a flat list of
            # raw HN item dicts (type/url/title/score/time), one "page",
            # no pagination -- see tests/fixtures/sample_hn.json and
            # tests/test_sources_more.py.
            return json.loads(Path(self.path).read_text(encoding="utf-8"))

        stories: list[dict] = []
        page = 0
        while True:
            data = self._query(since, page)
            hits = data.get("hits", [])
            for h in hits:
                stories.append({
                    "type": "story",
                    "url": h.get("url"),
                    "title": h.get("title") or "",
                    "score": h.get("points", 0),
                    "time": h.get("created_at_i"),
                })
            nb_pages = data.get("nbPages", 1)
            page += 1
            if not hits or page >= nb_pages:
                break
            if page >= self.max_pages:
                log.warning(
                    "hackernews: source=%s hit max_pages=%d cap while "
                    "paginating (nbPages=%d); more results may exist",
                    self.id, self.max_pages, nb_pages,
                )
                break
        return stories

    def fetch(self, since: datetime | None) -> Iterable[RawItem]:
        for story in self._stories(since):
            if not story or story.get("type") != "story":
                continue
            if not story.get("url"):
                continue
            if story.get("score", 0) < self.min_score:
                continue
            # HN's public API always sets a creation time, but don't crash
            # or silently drop an item if it were ever missing/malformed --
            # treat it the same as an undated item from any other source:
            # always include, never let it get flushed away by a `since`
            # comparison against nothing.
            raw_time = story.get("time")
            published = datetime.fromtimestamp(raw_time, tz=timezone.utc) if raw_time is not None else None
            if since is not None and published is not None and published <= since:
                continue
            yield RawItem(
                url=canonical_url(story["url"]),
                title=story.get("title", "").strip(),
                summary=None,
                published_at=published,
            )
