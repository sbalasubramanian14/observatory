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


def _extract(item: Item) -> str:
    """Prefer the summary the source gave us; fall back to fetching the page.

    arXiv abstracts and most RSS descriptions are already the best available
    text, and refetching adds latency, failure modes, and load on the source
    for no gain. An arXiv abstract URL short-circuits to "" without ever
    considering the fetch fallback -- refetching arXiv HTML adds nothing.
    """
    if item.summary and len(item.summary.strip()) >= MIN_TEXT_CHARS:
        return _WS.sub(" ", item.summary).strip()
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
