from __future__ import annotations
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from dateutil import parser as dateutil_parser

from feed.sources.base import RawItem, canonical_url
from feed.sources.registry import register

log = logging.getLogger(__name__)

USER_AGENT = (
    "ObservatoryFeedBot/0.1 (+personal AI news aggregator, non-commercial, "
    "single reader; contact via the project's GitHub repo)"
)

# Spec requirement: "rate-limit politely (a courteous delay between
# requests; state what you chose)". 2 seconds is applied once per fetch()
# call, between the robots.txt request and the page request -- the two
# real HTTP requests a single ScraperSource.fetch() makes against the
# target host. Chosen as a round, clearly-deliberate number comfortably
# above what any reasonable site would consider hammering for a source
# polled on a cadence of hours, not seconds (see sources.catalogue.toml,
# where every `scraper` entry runs at 120min+ cadence).
DEFAULT_DELAY_SECONDS = 2.0


def _get_text(url: str, *, timeout: float) -> str:
    """The real-network seam. Tests must monkeypatch this, never let it run
    for real -- mirrors feed.sources.hackernews._get / feed.sources.arxiv._get.
    Used for both the page fetch and the robots.txt fetch (see
    _allowed_by_robots), since both are plain GETs.
    """
    resp = httpx.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT},
                      follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _robots_url(page_url: str) -> str:
    parts = urlsplit(page_url)
    return urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))


def _allowed_by_robots(url: str, *, timeout: float) -> bool:
    """Fetch and consult robots.txt for `url`'s host. A robots.txt that
    cannot be fetched at all (missing, network error, whatever) is treated
    as "no restrictions" -- the same default urllib.robotparser applies
    when its own .read() fails, and is the conventional interpretation of
    an absent robots.txt across the web.
    """
    robots_url = _robots_url(url)
    parser = RobotFileParser()
    try:
        text = _get_text(robots_url, timeout=timeout)
    except Exception as exc:
        log.debug("scraper: could not fetch robots.txt at %s (%s); "
                  "treating as allowed", robots_url, exc)
        return True
    parser.parse(text.splitlines())
    return parser.can_fetch(USER_AGENT, url)


def _parse_date(raw: str | None) -> datetime | None:
    """Flexible date parsing that falls back to None rather than crashing.
    Undated items are a normal, expected shape (see RawItem.published_at
    and every other source plugin's `since` handling) -- a scraped page's
    date text is the least structured input of any source this project
    has, so this must never raise.
    """
    if not raw or not raw.strip():
        return None
    try:
        dt = dateutil_parser.parse(raw.strip())
    except (ValueError, OverflowError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


@register("scraper")
class ScraperSource:
    """Config-driven source for sites that publish no feed at all: a URL
    plus CSS selectors for the item container, title, link, date, and
    summary. This is what makes "extendable to many sources" real --
    adding a feedless site is editing sources.catalogue.toml, never
    writing Python.

    Config:
        url              page to scrape (mutually exclusive with `path`)
        path             local HTML fixture, for offline tests -- bypasses
                         robots.txt and the network entirely, same
                         convention as every other source plugin's `path=`.
        item_selector    CSS selector for each item's container element
                         (required)
        title_selector   CSS selector, relative to the item container, for
                         the title text. None means the container's own
                         text.
        link_selector    CSS selector, relative to the item container, for
                         the element carrying the link. None means the
                         container element itself carries the link
                         attribute (e.g. the container IS an <a>).
        link_attr        attribute holding the URL (default "href")
        date_selector    CSS selector, relative to the item container, for
                         the date element. None means no date is scraped
                         (published_at is always None).
        date_attr        attribute to read the date from (e.g. a
                         `datetime="..."` attribute). None (default) reads
                         the element's text content instead.
        summary_selector CSS selector, relative to the item container, for
                         a summary/dek element. None means no summary.
        timeout          per-request timeout in seconds (default 20.0)
        delay            seconds to sleep between the robots.txt request
                         and the page request (default DEFAULT_DELAY_SECONDS)
        respect_robots   if False, skip the robots.txt check entirely
                         (default True; only meant for tests of the
                         scraping logic itself in isolation -- production
                         config should never set this)
    """

    def __init__(self, source_id: str, url: str | None = None, path: str | None = None, *,
                 item_selector: str | None = None, title_selector: str | None = None,
                 link_selector: str | None = None, link_attr: str = "href",
                 date_selector: str | None = None, date_attr: str | None = None,
                 summary_selector: str | None = None, timeout: float = 20.0,
                 delay: float = DEFAULT_DELAY_SECONDS, respect_robots: bool = True):
        if not url and not path:
            raise ValueError("scraper source needs either url or path")
        if not item_selector:
            raise ValueError("scraper source needs item_selector")
        self.id = source_id
        self.url = url
        self.path = path
        self.item_selector = item_selector
        self.title_selector = title_selector
        self.link_selector = link_selector
        self.link_attr = link_attr
        self.date_selector = date_selector
        self.date_attr = date_attr
        self.summary_selector = summary_selector
        self.timeout = timeout
        self.delay = delay
        self.respect_robots = respect_robots
        # Set by fetch() when robots.txt disallows the page -- mirrors
        # RssSource.coverage_warning's contract: feed.stages.collect.collect()
        # reads this optional attribute after fully consuming fetch()'s
        # generator and persists it onto the Source row / sources.json.
        self.coverage_warning: str | None = None

    def _html(self) -> tuple[str, str] | tuple[None, None]:
        """Returns (html_text, base_url_for_relative_links), or (None, None)
        if robots.txt disallowed the fetch."""
        if self.path:
            return Path(self.path).read_text(encoding="utf-8"), (self.url or "")

        if self.respect_robots and not _allowed_by_robots(self.url, timeout=self.timeout):
            self.coverage_warning = (
                f"scraper source={self.id!r}: robots.txt disallows {self.url}; "
                f"skipped entirely this run"
            )
            log.warning(self.coverage_warning)
            return None, None

        time.sleep(self.delay)  # polite gap between the robots.txt and page requests
        return _get_text(self.url, timeout=self.timeout), self.url

    def fetch(self, since: datetime | None) -> Iterable[RawItem]:
        self.coverage_warning = None
        html, base_url = self._html()
        if html is None:
            return

        soup = BeautifulSoup(html, "lxml")
        nodes = soup.select(self.item_selector)
        for i, node in enumerate(nodes):
            link_node = node.select_one(self.link_selector) if self.link_selector else node
            href = link_node.get(self.link_attr) if link_node is not None else None
            if not href:
                log.warning("scraper: source=%s item #%d matched item_selector "
                           "but has no link, skipping", self.id, i)
                continue
            absolute = urljoin(base_url, href)

            title_node = node.select_one(self.title_selector) if self.title_selector else node
            title = title_node.get_text(strip=True) if title_node is not None else ""

            published = None
            if self.date_selector:
                date_node = node.select_one(self.date_selector)
                if date_node is not None:
                    raw_date = (date_node.get(self.date_attr) if self.date_attr
                               else date_node.get_text(strip=True))
                    published = _parse_date(raw_date)

            if since is not None and published is not None and published <= since:
                continue

            summary = None
            if self.summary_selector:
                summary_node = node.select_one(self.summary_selector)
                if summary_node is not None:
                    summary = summary_node.get_text(strip=True) or None

            yield RawItem(
                url=canonical_url(absolute),
                title=title,
                summary=summary,
                published_at=published,
            )
