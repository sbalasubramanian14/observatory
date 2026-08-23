from __future__ import annotations
import hashlib
import logging
import re
from urllib.parse import urljoin, urlsplit
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.models import Item, Stage
from feed.stages.base import StageResult, run_stage

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
MIN_TEXT_CHARS = 20

# spec D0: og:image, then twitter:image, in that priority order -- the
# fallback used only when neither the source feed nor a prior fallback
# already supplied item.image_url.
_IMAGE_META_PROPS = ("og:image", "twitter:image", "twitter:image:src")

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
_UNRELIABLE_IMAGE_HOSTS = ("research.facebook.com", "fbcdn.net")


def _is_unreliable_image_host(url: str) -> bool:
    host = urlsplit(url).netloc.lower()
    return any(host == h or host.endswith("." + h) for h in _UNRELIABLE_IMAGE_HOSTS)


class DuplicateContent(Exception):
    pass


def content_hash(text: str) -> str:
    collapsed = _WS.sub(" ", text or "").strip().lower()
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()


def _extract_og_image(html: str, base_url: str) -> str | None:
    """Pull og:image / twitter:image out of a page's <meta> tags.

    Pure (no I/O) so it is trivially testable against a fixture string --
    the network seams below (`_fetch_remote_page`, `_fetch_og_image`) are
    the only things that ever touch the network, and this is deliberately
    kept out of them. A candidate on `_UNRELIABLE_IMAGE_HOSTS` is skipped
    (not returned as a last resort) -- falling through to the next meta
    property, and ultimately to None, per spec D0's "store nothing rather
    than a known-broken image".
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for prop in _IMAGE_META_PROPS:
        tag = soup.find("meta", attrs={"property": prop}) or soup.find(
            "meta", attrs={"name": prop}
        )
        content = tag.get("content") if tag else None
        if content and content.strip():
            resolved = urljoin(base_url, content.strip())
            if _is_unreliable_image_host(resolved):
                log.debug("normalize: skipping known-unreliable image host: %s", resolved)
                continue
            return resolved
    return None


def _fetch_remote_text(url: str) -> str:
    """Fetch a page and extract its article text over the network.

    This is the ONLY place in the normalize stage that makes a real HTTP
    request. It is deliberately factored out to a module-level function
    (rather than inlined in `_extract`) so tests can monkeypatch
    `feed.stages.normalize._fetch_remote_text` and guarantee the fallback
    path never touches the network -- see tests/test_normalize.py, which
    also carries an autouse fixture that makes the underlying
    trafilatura.fetch_url call raise if it is ever reached in a test,
    as a second line of defense.
    """
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    return trafilatura.extract(downloaded) or ""


def _fetch_og_image(url: str) -> str | None:
    """Fetch solely for the og:image/twitter:image fallback -- used when
    item.summary already gave usable text (so `_fetch_remote_text`/
    `_fetch_remote_page` are never called) and only the image is still
    missing. A distinct, independently monkeypatchable network seam, same
    contract as `_fetch_remote_text` above.
    """
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    try:
        return _extract_og_image(downloaded, url)
    except Exception:
        return None


def _html_to_text(html: str) -> str:
    """Strip markup from an RSS/Atom summary using trafilatura, not a regex.

    I4 fix: `_extract` used to only whitespace-collapse item.summary, so a
    real RSS description like
    '<p>OpenAI <a href="...">announced</a> GPT-6...</p>' was stored
    verbatim in item.text, polluting content_hash, the embedding text, and
    entity extraction (a stray hostname or CSS class value inside an
    attribute reads as a spurious capitalised "entity"). A hand-rolled
    regex HTML stripper is its own bug farm (nested tags, entities,
    malformed markup), so this reuses trafilatura -- already a project
    dependency for the network-fallback path below -- instead.

    trafilatura.extract() silently returns None for a bare fragment with no
    <html>/<body> wrapper, regardless of content length or quality (verified
    empirically), which would otherwise make every RSS summary look like
    "no usable text". Wrapping guarantees it sees something that looks like
    a real document without changing the content itself. favor_recall=True
    keeps short, low-boilerplate fragments (typical of a one-paragraph
    summary) from being discarded as insufficiently substantial.

    Falls back to the previous whitespace-collapse behaviour if trafilatura
    still returns nothing (e.g. a summary that is pure junk) -- degrading to
    the old behaviour is better than losing the text entirely.
    """
    import trafilatura

    extracted = trafilatura.extract(f"<html><body>{html}</body></html>",
                                    favor_recall=True)
    return _WS.sub(" ", extracted if extracted else html).strip()


def _extract(item: Item) -> str:
    """Prefer the summary the source gave us; fall back to fetching the page.

    arXiv abstracts and most RSS descriptions are already the best available
    text, and refetching adds latency, failure modes, and load on the source
    for no gain. An arXiv abstract URL short-circuits to "" without ever
    considering the fetch fallback -- refetching arXiv HTML adds nothing.
    """
    if item.summary and len(item.summary.strip()) >= MIN_TEXT_CHARS:
        return _html_to_text(item.summary)
    if item.url.startswith("https://arxiv.org/abs/"):
        return ""
    return _WS.sub(" ", _fetch_remote_text(item.url)).strip()


def _resolve_image(item: Item) -> str | None:
    """spec D0's fallback chain, step 2: the article page's og:image /
    twitter:image meta tag, tried only when the source feed didn't already
    supply item.image_url (step 1, set at collect time from RawItem.image_url
    -- see feed.sources.base.extract_feed_image).

    Best-effort and never raises: a missing/broken lead image is cosmetic,
    never a reason to fail the item's normalization. This also means any
    real network failure (or, in tests, the suite-wide `_block_real_network`
    guard tripping on `trafilatura.fetch_url`) degrades identically to
    "no image found" -- there is no scenario where fetching an image is
    worth crashing the pipeline over. Skipped for arXiv abstract pages for
    the same "refetching arXiv adds nothing" reason `_extract` already
    applies to full text.
    """
    if item.image_url:
        # Defense in depth: even a feed-native image (media:content/
        # thumbnail/enclosure, set at collect time) is rejected if it
        # resolves to a known-unreliable host -- the denylist is about the
        # image URL itself, not which extraction path produced it.
        if _is_unreliable_image_host(item.image_url):
            log.debug("normalize: discarding feed-supplied image on a "
                     "known-unreliable host: %s", item.image_url)
        else:
            return item.image_url
    if item.url.startswith("https://arxiv.org/abs/"):
        return None
    try:
        return _fetch_og_image(item.url)
    except Exception as exc:
        log.debug("normalize: og:image fallback failed for %s: %s", item.url, exc)
        return None


def normalize_item(session: Session, item: Item) -> None:
    text = _extract(item)
    if len(text) < MIN_TEXT_CHARS:
        raise ValueError(f"no text extracted for {item.url}")
    digest = content_hash(text)
    clash = session.scalar(
        select(Item.id).where(Item.content_hash == digest, Item.id != item.id)
    )
    if clash:
        raise DuplicateContent(f"duplicate content of item {clash}")
    item.text = text
    item.content_hash = digest
    item.image_url = _resolve_image(item)


def normalize(session: Session, limit: int = 100) -> StageResult:
    return run_stage(
        session, name="normalize", claim_stage=Stage.COLLECTED,
        next_stage=Stage.NORMALIZED, handler=normalize_item, limit=limit,
    )
