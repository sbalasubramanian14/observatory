from __future__ import annotations
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable
from urllib.parse import urljoin, urlsplit

import httpx
from sqlalchemy.orm import Session

from feed.models import Item

log = logging.getLogger(__name__)

# spec D0, moved here from feed.stages.normalize (Phase D-images): og:image,
# then twitter:image, in that priority order.
IMAGE_META_PROPS = ("og:image", "twitter:image", "twitter:image:src")

# spec D0 ("if a source's images reliably fail, it is better to store
# nothing than to render broken images"): hostnames whose own og:image URLs
# were measured, live, to be structurally unusable for hotlinking -- not a
# one-off dead link, but the URL *shape* itself. research.facebook.com's
# own file-hosting path (`research.facebook.com/file/<id>/<name>`) returned
# HTTP 400 for every article sampled (2/2), with and without a Referer
# header, so this is not hotlink-protection that a browser's real request
# would pass -- it is broken regardless of caller. (Meta's *other* image
# host, scontent.*.fbcdn.net, does load, but only via short-lived signed
# URLs that would go stale long before the bundle's 90-day retention window
# elapses, so it is excluded here too -- storing a URL known to expire is
# the same "store nothing instead" case spec D0 describes, just on a
# delay.) Checked against the resolved image URL's host, not the article's.
#
# Single source of truth: previously duplicated in feed.stages.normalize;
# consolidated here (Phase D-images) so the live per-item path (normalize)
# and the concurrent bulk path (this module, used by both the normalize
# post-step and `feed backfill-images`) can never drift apart -- exactly
# the "bolting on a second path" this phase's brief warned against.
UNRELIABLE_IMAGE_HOSTS = ("research.facebook.com", "fbcdn.net")

# HTTP statuses that mean "this host will not give me an image right now,
# and asking again next run is not going to help" -- a bot-challenge page
# (202), an explicit block (403), or a rate limit (429). These are recorded
# as a completed, non-retryable attempt (image_checked_at is set) exactly
# like a clean 200-with-no-og-tag, NOT treated as a transient error worth
# retrying on the very next run. A different, unexpected status (500, 404,
# ...) is still recorded the same way -- the one thing genuinely left for
# "retry later" is a raised network-level exception (timeout, DNS failure,
# connection reset), which leaves image_checked_at unset. See
# ImageFetchResult.status and _needs_image_fetch below.
#
# A1-followup deliberately did NOT give this 429 the arXiv-style
# retry-with-backoff treatment (feed.sources.arxiv._live_get_page), even
# though the live symptom that prompted that fix -- VentureBeat article
# pages returning 429 during image fetching, while VentureBeat's own RSS
# feed stays healthy -- looks superficially similar. Two things make it a
# different case: (1) this 429 never touches Source.consecutive_failures
# or the sources health page at all -- it is scoped to one item's
# cosmetic lead image, collected via a completely separate path from
# collect()'s connector-health tracking, so it can never manufacture a
# false Degraded reading the way the arXiv one did; (2) og:image is
# explicitly non-critical (see fetch_og_image's docstring), so spending
# extra latency/requests retrying a blocked image scrape has a worse
# cost/benefit than retrying a whole connector's collection run. If a
# publisher's 429 here ever turns out to be genuinely transient rather
# than a standing bot-block, the fix is to add it to NO_RETRY_STATUSES's
# opposite -- not to bolt retry logic onto this already-bulk, already
# per-host-throttled path.
NO_RETRY_STATUSES = frozenset({403, 202, 429})

DEFAULT_TIMEOUT = 15.0
DEFAULT_MAX_WORKERS = 8
# Minimum gap between two requests to the *same* host, enforced by
# HostThrottle. 1s is comfortably polite for a personal-reader bot hitting
# a handful of requests against any one publisher in a single run (the
# backlog is spread across ~30 distinct hosts, so the wall-clock cost is
# governed by max_workers, not this delay).
DEFAULT_HOST_DELAY = 1.0

USER_AGENT = "feed/0.1 (personal reader)"


def _get(url: str, *, timeout: float) -> httpx.Response:
    """The real-network seam. Tests must monkeypatch this, never let it run
    for real -- mirrors feed.sources.hackernews._get / feed.sources.arxiv._get
    / feed.sources.scraper._get_text. Deliberately does NOT call
    raise_for_status(): unlike those seams, a non-2xx response here is
    meaningful data (403/202/429 vs. a generic failure), not an error to
    propagate -- see fetch_og_image's status handling.
    """
    return httpx.get(url, timeout=timeout, follow_redirects=True,
                     headers={"User-Agent": USER_AGENT})


def is_unreliable_image_host(url: str) -> bool:
    host = urlsplit(url).netloc.lower()
    return any(host == h or host.endswith("." + h) for h in UNRELIABLE_IMAGE_HOSTS)


def extract_meta_image(html: str, base_url: str) -> str | None:
    """Pull og:image / twitter:image out of a page's <meta> tags.

    Pure (no I/O) so it is trivially testable against a fixture string. A
    candidate on UNRELIABLE_IMAGE_HOSTS is skipped (not returned as a last
    resort) -- falling through to the next meta property, and ultimately
    to None, per spec D0's "store nothing rather than a known-broken image".
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for prop in IMAGE_META_PROPS:
        tag = soup.find("meta", attrs={"property": prop}) or soup.find(
            "meta", attrs={"name": prop}
        )
        content = tag.get("content") if tag else None
        if content and content.strip():
            resolved = urljoin(base_url, content.strip())
            if is_unreliable_image_host(resolved):
                log.debug("imaging: skipping known-unreliable image host: %s", resolved)
                continue
            return resolved
    return None


@dataclass(frozen=True, slots=True)
class ImageFetchResult:
    """Outcome of one og:image fetch attempt.

    status is one of:
      "ok"            -- image_url is a usable, non-denylisted image.
      "no_og_tag"      -- page fetched fine (2xx), no usable meta tag --
                          including the case where the only candidate(s)
                          found were on UNRELIABLE_IMAGE_HOSTS and skipped
                          by extract_meta_image itself (which tries the
                          next meta property rather than reporting the
                          denylisted one; see its docstring). The denylist
                          is fully enforced either way -- a denylisted URL
                          is never returned as "ok" -- this status just
                          does not distinguish "no tag at all" from "only
                          a denylisted tag" for reporting purposes.
      "blocked_<code>" -- one of NO_RETRY_STATUSES (403/202/429).
      "http_<code>"    -- any other non-2xx response.
      "network_error:<ExceptionClassName>" -- the request itself raised
                          (timeout, DNS failure, connection reset, ...);
                          the only status that should NOT be recorded as a
                          completed attempt, since it says nothing about
                          whether the page actually has an image.
    """
    image_url: str | None
    status: str

    @property
    def is_transient(self) -> bool:
        return self.status.startswith("network_error:")


def fetch_og_image(url: str, *, timeout: float = DEFAULT_TIMEOUT) -> ImageFetchResult:
    """Fetch `url` and extract its og:image/twitter:image, classifying the
    outcome per ImageFetchResult's status contract. Never raises -- a
    missing/broken lead image is cosmetic, never a reason to fail an item
    or crash a batch (mirrors feed.stages.normalize._resolve_image's old
    docstring, which this function's caller-facing contract preserves).
    """
    try:
        resp = _get(url, timeout=timeout)
    except Exception as exc:
        return ImageFetchResult(None, f"network_error:{type(exc).__name__}")

    if resp.status_code in NO_RETRY_STATUSES:
        return ImageFetchResult(None, f"blocked_{resp.status_code}")
    if resp.status_code >= 400 or resp.status_code >= 300:
        # 3xx here means httpx's follow_redirects exhausted its redirect
        # cap or landed on a non-2xx after redirecting -- treat the same
        # as any other "did not get a usable page" outcome.
        return ImageFetchResult(None, f"http_{resp.status_code}")

    try:
        image = extract_meta_image(resp.text, str(resp.url))
    except Exception as exc:
        log.debug("imaging: parse failed for %s: %s", url, exc)
        return ImageFetchResult(None, f"parse_error:{type(exc).__name__}")

    if image is None:
        # extract_meta_image already filters UNRELIABLE_IMAGE_HOSTS
        # internally (falling through to the next meta property rather
        # than returning a denylisted candidate), so None here already
        # means "no usable, non-denylisted image" -- there is nothing
        # left for this function to additionally reject.
        return ImageFetchResult(None, "no_og_tag")
    return ImageFetchResult(image, "ok")


def _sleep(seconds: float) -> None:
    """The real-delay seam for HostThrottle, mirroring
    feed.providers._retry._sleep. Tests must monkeypatch this to a no-op
    (see tests/conftest.py's autouse `_no_real_imaging_sleep`) rather than
    let the suite genuinely block -- a batch of test items sharing one
    host (a common fixture pattern, e.g. many `https://example.com/...`
    URLs in one feed) would otherwise serialize at DEFAULT_HOST_DELAY
    seconds each, turning an unrelated, otherwise-instant test into one
    that takes minutes for no real reason: the mocked/blocked network
    seam already makes every individual fetch fail instantly, so there is
    no real host on the other end for the delay to protect.
    """
    time.sleep(seconds)


class HostThrottle:
    """Per-host politeness delay shared across a thread pool.

    Bounded concurrency (ThreadPoolExecutor's max_workers) already caps how
    many requests are in flight at once, but does nothing to stop several
    of those in-flight requests from landing on the SAME host back-to-back
    -- with ~400 items/day concentrated on ~30 publishers, a burst of
    concurrent openai.com or techcrunch.com requests is exactly the kind of
    hammering spec D0's brief warns against. wait() blocks the calling
    thread until at least `delay` seconds have passed since the last
    request *to that host*, while leaving requests to other hosts free to
    proceed concurrently -- the lock only protects the bookkeeping dict,
    never the sleep itself.
    """

    def __init__(self, delay: float = DEFAULT_HOST_DELAY):
        self.delay = delay
        self._lock = threading.Lock()
        self._next_allowed: dict[str, float] = {}

    def wait(self, host: str) -> None:
        if self.delay <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_allowed.get(host, now))
            self._next_allowed[host] = start + self.delay
        sleep_for = start - time.monotonic()
        if sleep_for > 0:
            _sleep(sleep_for)


def needs_image_fetch(item: Item) -> bool:
    """Cache/skip rule (Phase D-images brief): never re-fetch a page for an
    item that already has an image, that has already been tried (whether
    or not it found one), or that is an arXiv abstract page (refetching
    arXiv adds nothing -- same reasoning feed.stages.normalize._extract
    already applies to full-text fetching).
    """
    if item.image_url:
        return False
    if item.image_checked_at is not None:
        return False
    if item.url.startswith("https://arxiv.org/abs/"):
        return False
    return True


@dataclass
class ImageBackfillResult:
    attempted: int = 0
    gained: int = 0
    # by_source[source_id][status] -> count. "gained" is its own status key
    # (rather than "ok") so callers can report "N gained" per source
    # without re-deriving it from ImageFetchResult.status.
    by_source: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    @property
    def total_sources(self) -> int:
        return len(self.by_source)


def resolve_images(
    session: Session,
    items: Iterable[Item],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    host_delay: float = DEFAULT_HOST_DELAY,
    timeout: float = DEFAULT_TIMEOUT,
    now: datetime | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> ImageBackfillResult:
    """Concurrently resolve og:image for every candidate item, with a
    bounded worker pool and a per-host politeness delay.

    Only items passing needs_image_fetch() are attempted -- an item that
    already has an image, or was already tried, is skipped without
    touching the network (the cache/skip rule from the brief). Every
    attempted item is committed individually as its result arrives, so a
    `feed backfill-images` run interrupted partway through leaves every
    completed item's result durably recorded and is fully resumable: the
    next run's needs_image_fetch() filter picks up exactly where it left
    off, re-trying only items that never got a definitive answer (a raised
    network error, which leaves image_checked_at unset -- see
    ImageFetchResult's docstring).

    The network fetch itself (fetch_og_image, called from worker threads)
    never touches `session` -- only this function's own thread applies
    results to ORM objects and commits, which is the only SQLAlchemy-
    session-safe way to use a pool here (a Session is not thread-safe).
    """
    now = now or datetime.now(timezone.utc)
    result = ImageBackfillResult()
    throttle = HostThrottle(host_delay)

    candidates = [it for it in items if needs_image_fetch(it)]
    by_id = {it.id: it for it in candidates}
    total = len(candidates)
    if not total:
        return result

    def _fetch(item_id: int, url: str) -> tuple[int, ImageFetchResult]:
        host = urlsplit(url).netloc
        throttle.wait(host)
        return item_id, fetch_og_image(url, timeout=timeout)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = [pool.submit(_fetch, it.id, it.url) for it in candidates]
        for fut in as_completed(futures):
            item_id, fr = fut.result()
            item = by_id[item_id]

            if fr.is_transient:
                # Genuine network failure: do NOT mark as checked, so a
                # later run retries it -- see ImageFetchResult's docstring.
                log.debug("imaging: transient failure for item=%s url=%s: %s",
                         item_id, item.url, fr.status)
                continue

            result.attempted += 1
            item.image_checked_at = now
            if fr.status == "ok":
                item.image_url = fr.image_url
                result.gained += 1
                result.by_source[item.source_id]["gained"] += 1
            else:
                result.by_source[item.source_id][fr.status] += 1
            session.commit()
            if on_progress is not None:
                on_progress(result.attempted, total)

    return result
