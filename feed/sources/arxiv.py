from __future__ import annotations
import logging
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Iterable
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

    Deliberately still a single, un-retried request: retry-with-backoff
    lives one level up, in ArxivSource._live_get_page, so tests can
    monkeypatch THIS seam to return a 429 on call N and a real payload on
    call N+1 and thereby exercise the retry loop for real, rather than
    retrying happening invisibly inside a seam tests can't see into.
    """
    resp = httpx.get(url, timeout=timeout,
                      headers={"User-Agent": "feed/0.1 (personal reader)"})
    resp.raise_for_status()
    return resp.content


def _sleep(seconds: float) -> None:
    """The real-delay seam, mirroring feed.providers._retry._sleep and
    feed.imaging._sleep. Tests must monkeypatch this to a no-op (see
    tests/conftest.py's autouse `_no_real_arxiv_sleep`) rather than let the
    suite genuinely block for the politeness delay or a 429 backoff.
    """
    time.sleep(seconds)


def _parse_retry_after(value: str | None, *, now: datetime) -> float | None:
    """Parses a `Retry-After` header value, which per RFC 9110 is either a
    plain integer number of seconds or an HTTP-date. Returns None if the
    header is absent or unparseable as either form -- callers fall back to
    exponential backoff in that case.
    """
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (dt - now).total_seconds())


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

    Rate limit / A1-followup: arXiv's API asks for ~3 seconds between
    requests. Every live request -- the first page of a run as well as
    every later page -- goes through _live_get_page, which waits out
    `rate_limit_seconds` since the LAST request this source actually made
    (self.last_request_at) before issuing the next one. That timestamp is
    read from and written back to the persisted Source.last_request_at
    column by feed.stages.collect (via plain attribute get/set -- see its
    docstring), so the delay is honoured ACROSS separate `feed run`
    invocations, not just between pages within one. Without that, a run's
    very first request had no delay at all: two runs close together could
    land the second run's opening request right on the heels of the first
    run's last page, which is what tripped the 429 this fixes.

    A live request that comes back 429 is retried up to `max_retries`
    times with exponential backoff (`retry_backoff_base * 2**attempt`),
    honouring the response's `Retry-After` header when present instead of
    the computed backoff. Only once retries are exhausted does the 429
    propagate as a real failure -- collect() only marks a source Degraded
    (increments consecutive_failures) for THAT, never for a throttle that
    a retry cleared, since a transient 429 is not "this connector is
    broken", which is what the health page is supposed to report.
    Fixture-backed fetches (path=/paths=) never sleep or retry, since they
    never call the network.
    """

    API = "https://export.arxiv.org/api/query"

    def __init__(self, source_id: str, categories: list[str] | None = None,
                 max_results: int = 100, path: str | None = None,
                 paths: list[str] | None = None, timeout: float = 60.0,
                 max_pages: int = 20, rate_limit_seconds: float = 3.0,
                 max_retries: int = 4, retry_backoff_base: float = 2.0,
                 now: Callable[[], datetime] | None = None):
        self.id = source_id
        # Injectable clock (mirrors feed.providers.health.ProviderHealthTracker),
        # so pacing/retry-delay tests can assert exact computed delays
        # against a controlled clock instead of the real wall clock.
        self._now = now or (lambda: datetime.now(timezone.utc))
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
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        # A1-followup: cross-run politeness-delay state. None means "no
        # prior request known" (first-ever run, or a source that predates
        # this column). feed.stages.collect sets this from the persisted
        # Source.last_request_at before calling fetch(), and reads it back
        # afterwards to persist whatever this instance last set it to --
        # see feed.models.Source.last_request_at's docstring. Purely
        # duck-typed: collect() only touches this attribute when it's
        # present, so no other source plugin is affected.
        self.last_request_at: datetime | None = None

    def _wait_for_pace(self) -> None:
        if self.last_request_at is None:
            return
        elapsed = (self._now() - self.last_request_at).total_seconds()
        remaining = self.rate_limit_seconds - elapsed
        if remaining > 0:
            _sleep(remaining)

    def _live_get_page(self, url: str) -> bytes:
        """Issues one live request, waiting out the politeness delay first
        and retrying a 429 with backoff. Stamps self.last_request_at on
        EVERY attempt (successful or not) since each attempt is a real
        request against arXiv's rate limiter, not just the final one that
        happens to succeed.
        """
        attempt = 0
        while True:
            self._wait_for_pace()
            try:
                content = _get(url, timeout=self.timeout)
                self.last_request_at = self._now()
                return content
            except httpx.HTTPStatusError as exc:
                self.last_request_at = self._now()
                status = exc.response.status_code if exc.response is not None else None
                if status != 429 or attempt >= self.max_retries:
                    raise
                now = self._now()
                retry_after = _parse_retry_after(
                    exc.response.headers.get("Retry-After") if exc.response is not None else None,
                    now=now,
                )
                delay = retry_after if retry_after is not None else (
                    self.retry_backoff_base * (2 ** attempt)
                )
                log.warning(
                    "arxiv: source=%s got 429, retrying in %.1fs (attempt %d/%d)%s",
                    self.id, delay, attempt + 1, self.max_retries,
                    " [Retry-After honoured]" if retry_after is not None else "",
                )
                _sleep(delay)
                attempt += 1

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
        return self._live_get_page(url)

    def fetch(self, since: datetime | None) -> Iterable[RawItem]:
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
            # No explicit sleep here: the next iteration's _fetch_page ->
            # _live_get_page calls _wait_for_pace() itself before issuing
            # the request, which is what now unifies the between-page
            # delay with the cross-run one described in the class
            # docstring. Fixture-backed fetches never reach
            # _live_get_page at all, so this is a no-op for tests.
