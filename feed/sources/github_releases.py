from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import feedparser
import httpx
from feed.sources.base import RawItem, canonical_url
from feed.sources.registry import register


@register("github_releases")
class GithubReleasesSource:
    """Release notes for a repo via its public releases.atom feed."""

    def __init__(self, source_id: str, repo: str | None = None,
                 path: str | None = None, timeout: float = 20.0):
        if not repo and not path:
            raise ValueError("github_releases needs repo or path")
        self.id = source_id
        self.repo = repo
        self.path = path
        self.timeout = timeout

    @property
    def feed_url(self) -> str:
        return f"https://github.com/{self.repo}/releases.atom"

    def _raw(self) -> str | bytes:
        if self.path:
            return Path(self.path).read_bytes()
        resp = httpx.get(self.feed_url, timeout=self.timeout,
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
            tm = entry.get("updated_parsed") or entry.get("published_parsed")
            published = datetime(*tm[:6], tzinfo=timezone.utc) if tm else None
            if since is not None and published is not None and published <= since:
                continue
            yield RawItem(
                url=canonical_url(link),
                title=f"{self.repo}: {(entry.get('title') or '').strip()}",
                summary=(entry.get("summary") or None),
                published_at=published,
            )
