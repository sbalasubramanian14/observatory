from __future__ import annotations
import hashlib
import re
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.models import Item, Stage
from feed.stages.base import StageResult, run_stage

_WS = re.compile(r"\s+")
MIN_TEXT_CHARS = 20


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


def normalize(session: Session, limit: int = 100) -> StageResult:
    return run_stage(
        session, name="normalize", claim_stage=Stage.COLLECTED,
        next_stage=Stage.NORMALIZED, handler=normalize_item, limit=limit,
    )
