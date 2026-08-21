from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import httpx
from feed.sources.base import RawItem, canonical_url
from feed.sources.registry import register


@register("hackernews")
class HackerNewsSource:
    """Hacker News top stories above a score floor.

    The score floor is the point: HN volume is enormous and low-score stories
    are noise the pipeline should never pay to embed.
    """

    TOP = "https://hacker-news.firebaseio.com/v0/topstories.json"
    ITEM = "https://hacker-news.firebaseio.com/v0/item/{id}.json"

    def __init__(self, source_id: str, min_score: int = 100, limit: int = 100,
                 path: str | None = None, timeout: float = 20.0):
        self.id = source_id
        self.min_score = min_score
        self.limit = limit
        self.path = path
        self.timeout = timeout

    def _stories(self) -> list[dict]:
        if self.path:
            return json.loads(Path(self.path).read_text(encoding="utf-8"))
        with httpx.Client(timeout=self.timeout) as client:
            ids = client.get(self.TOP).json()[: self.limit]
            return [client.get(self.ITEM.format(id=i)).json() for i in ids]

    def fetch(self, since: datetime | None) -> Iterable[RawItem]:
        for story in self._stories():
            if not story or story.get("type") != "story":
                continue
            if not story.get("url"):
                continue
            if story.get("score", 0) < self.min_score:
                continue
            # HN's public API always sets `time`, but don't crash or silently
            # drop an item if it were ever missing/malformed -- treat it the
            # same as an undated item from any other source: always include,
            # never let it get flushed away by a `since` comparison against
            # nothing.
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
