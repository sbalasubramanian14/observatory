from __future__ import annotations
import hashlib
import logging
import re
from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from feed.imaging import (
    DEFAULT_HOST_DELAY,
    DEFAULT_MAX_WORKERS,
    extract_meta_image,
    is_unreliable_image_host,
    resolve_images,
)
from feed.models import Item, Stage
from feed.stages.base import StageResult, run_stage

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
MIN_TEXT_CHARS = 20

# Backward-compatible aliases: this logic moved to feed.imaging (Phase
# D-images) so the live per-item path here and the concurrent bulk path
# (feed.imaging.resolve_images, used by this stage's post-step below AND
# by `feed backfill-images`) share one implementation instead of two that
# could silently drift apart. Kept as module-level names because
# tests/test_normalize.py's og:image unit tests call them directly.
_extract_og_image = extract_meta_image
_is_unreliable_image_host = is_unreliable_image_host


class DuplicateContent(Exception):
    pass


def content_hash(text: str) -> str:
    collapsed = _WS.sub(" ", text or "").strip().lower()
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()


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
    """spec D0's fallback chain, step 1 ONLY: keep the source feed's own
    image (media:content/thumbnail/enclosure, set at collect time from
    RawItem.image_url -- see feed.sources.base.extract_feed_image), unless
    it resolves to a known-unreliable host.

    Step 2 -- fetching the article page's og:image/twitter:image meta tag
    when the feed carried none -- is deliberately NOT done here. It used to
    be: one synchronous network round-trip inlined into this per-item
    handler, which run_stage calls once per row in a plain serial loop.
    That is exactly the "large backlog takes hours" failure mode this
    project already hit once with `_extract`'s remote-text fallback above
    -- at ~400 items/day it means ~400 extra serial HTTP requests through
    normalize() alone. Step 2 now runs as a separate, concurrent,
    politeness-throttled batch pass (feed.imaging.resolve_images) over
    whatever this call just normalized -- see normalize() below, and
    `feed backfill-images` for the same pass run over the existing corpus.
    """
    if item.image_url and is_unreliable_image_host(item.image_url):
        # Defense in depth: even a feed-native image is rejected if it
        # resolves to a known-unreliable host -- the denylist is about the
        # image URL itself, not which extraction path produced it.
        log.debug("normalize: discarding feed-supplied image on a "
                 "known-unreliable host: %s", item.image_url)
        return None
    return item.image_url


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


def normalize(
    session: Session,
    limit: int = 100,
    *,
    image_max_workers: int = DEFAULT_MAX_WORKERS,
    image_host_delay: float = DEFAULT_HOST_DELAY,
) -> StageResult:
    """Run the per-item claim/normalize/advance loop, then a concurrent
    og:image-fallback pass (spec D0 step 2) over whatever this call just
    moved to Stage.NORMALIZED and still has no image.

    Scoped to `Stage.NORMALIZED` (not the whole table) so this stays cheap
    on every call -- an item that already has an image, or was already
    tried (feed.imaging.needs_image_fetch), is excluded before any network
    call happens, so items from a PRIOR call to normalize() in the same
    drain() loop (already checked) are not re-queried here; only this
    call's freshly-normalized, still-uncertain rows are. The historical
    backlog -- items normalized before this fallback existed, already past
    Stage.NORMALIZED -- is handled separately by `feed backfill-images`,
    which runs the identical feed.imaging.resolve_images over the whole
    corpus rather than "bolting on a second implementation" of the fetch
    itself.
    """
    result = run_stage(
        session, name="normalize", claim_stage=Stage.COLLECTED,
        next_stage=Stage.NORMALIZED, handler=normalize_item, limit=limit,
    )
    if result.processed:
        pending = session.scalars(
            select(Item).where(Item.stage == Stage.NORMALIZED,
                               or_(Item.image_url.is_(None), Item.image_url == ""),
                               Item.image_checked_at.is_(None))
        ).all()
        if pending:
            ir = resolve_images(session, pending, max_workers=image_max_workers,
                               host_delay=image_host_delay)
            log.info("normalize: image fallback attempted=%d gained=%d",
                    sum(sum(c.values()) for c in ir.by_source.values()), ir.gained)
    return result
