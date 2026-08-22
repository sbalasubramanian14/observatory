from __future__ import annotations
import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.bundle_schema import (
    FeedPage, FeedPageStory, Manifest, ManifestPage, SourceHealth,
    SourcesReport, StoryDetail, StoryEvidence,
)
from feed.config import PublishConfig
from feed.embedding.base import unpack
from feed.models import Item, Source, Story

log = logging.getLogger(__name__)

BUNDLE_VERSION = 1


@dataclass
class PublishResult:
    published: bool = False
    story_count: int = 0
    page_count: int = 0
    pruned: int = 0
    out_dir: Path | None = None
    error: str | None = None


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _hash_json(obj: dict) -> tuple[str, bytes]:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _hash_bytes(payload), payload


def _evidence_for(story: Story) -> list[dict]:
    """Titles, canonical links, and metadata ONLY -- spec 4.2's hard
    exclusion. Deliberately never touches item.text (the full scraped
    article body) or item.summary (an unvetted publisher blurb) -- see
    StoryEvidence in feed/bundle_schema.py, which enforces this as a
    schema-level safety net in case a future edit here regresses it.
    """
    items = sorted(story.items, key=lambda it: it.id)
    return [
        {
            "id": it.id,
            "title": it.title,
            "url": it.url,
            "source_id": it.source_id,
            "published_at": _iso(it.published_at),
        }
        for it in items
    ]


def _story_detail_dict(story: Story) -> dict:
    return {
        "id": story.id,
        "title": story.title,
        "kind": story.kind,
        "category": story.category,
        "summary": story.summary,
        "analysis": story.analysis,
        "analysis_provider": story.analysis_provider,
        "score": story.score,
        "score_breakdown": story.score_breakdown,
        "first_seen": _iso(story.first_seen),
        "updated_at": _iso(story.updated_at),
        "item_count": story.item_count,
        "outlet_count": story.outlet_count,
        "evidence": _evidence_for(story),
    }


def _feed_page_story_dict(story: Story, *, detail_path: str, detail_hash: str) -> dict:
    return {
        "id": story.id,
        "title": story.title,
        "kind": story.kind,
        "category": story.category,
        "summary": story.summary,
        "score": story.score,
        "item_count": story.item_count,
        "outlet_count": story.outlet_count,
        "updated_at": _iso(story.updated_at),
        "detail_path": detail_path,
        "detail_hash": detail_hash,
    }


def _dominant_embedding_model_id(session: Session, story_ids: list[int]) -> str | None:
    if not story_ids:
        return None
    ids = session.scalars(
        select(Item.embedding_model_id).where(
            Item.story_id.in_(story_ids), Item.embedding_model_id.is_not(None)
        )
    ).all()
    if not ids:
        return None
    return Counter(ids).most_common(1)[0][0]


def _sources_report_dict(session: Session, *, now: datetime) -> dict:
    sources = []
    for src in session.scalars(select(Source).order_by(Source.id)):
        error = src.last_error
        if error and len(error) > 300:
            error = error[:300] + "..."
        sources.append({
            "id": src.id,
            "plugin": src.plugin,
            "enabled": src.enabled,
            "cadence_minutes": src.cadence_minutes,
            "last_run_at": _iso(src.last_run_at),
            "consecutive_failures": src.consecutive_failures,
            "last_error": error,
        })
    return {"generated_at": _iso(now), "sources": sources}


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _prune_dir(directory: Path, keep: set[str]) -> int:
    if not directory.exists():
        return 0
    pruned = 0
    for f in directory.iterdir():
        if f.is_file() and f.name not in keep:
            f.unlink()
            pruned += 1
    return pruned


def publish(session: Session, cfg: PublishConfig, out_dir: str | Path,
           *, now: datetime | None = None) -> PublishResult:
    """Write the static bundle (spec 4.1) to `out_dir` and prune anything
    outside the retention window (spec 4.4).

    Every payload is validated against feed/bundle_schema.py BEFORE any
    file is written -- a schema failure aborts the whole publish with
    nothing written or overwritten, per spec 6 ("the Publish stage
    validates against the schema before committing, and refuses to publish
    on failure"). This also structurally enforces spec 4.2: item.text can
    never reach disk because StoryEvidence's schema (extra="forbid") has no
    field for it, and this stage never puts it in a payload to begin with.
    """
    now = now or datetime.now(timezone.utc)
    out = Path(out_dir)
    cutoff = now - timedelta(days=cfg.retention_days)
    result = PublishResult(out_dir=out)

    stories = list(session.scalars(
        select(Story)
        .where(Story.score.is_not(None), Story.updated_at >= cutoff)
        .order_by(Story.score.desc(), Story.id)
    ))

    try:
        # --- build + validate everything in memory first ---------------
        story_files: dict[int, tuple[str, bytes, str]] = {}  # id -> (rel_path, bytes, hash)
        for story in stories:
            detail = StoryDetail.model_validate(_story_detail_dict(story))
            h, payload = _hash_json(detail.model_dump(mode="json"))
            rel = f"story/{story.id}-{h}.json"
            story_files[story.id] = (rel, payload, h)

        page_stories = [
            _feed_page_story_dict(s, detail_path=story_files[s.id][0],
                                  detail_hash=story_files[s.id][2])
            for s in stories
        ]
        page_size = cfg.page_size
        page_count = max(1, (len(page_stories) + page_size - 1) // page_size) \
            if page_stories else 1

        page_files: list[tuple[str, bytes, str, int]] = []  # rel, bytes, hash, count
        for n in range(page_count):
            chunk = page_stories[n * page_size:(n + 1) * page_size]
            page = FeedPage.model_validate({
                "page": n, "page_count": page_count, "stories": chunk,
            })
            h, payload = _hash_json(page.model_dump(mode="json"))
            rel = f"feed/page-{n}-{h}.json"
            page_files.append((rel, payload, h, len(chunk)))

        embedding_model_id = _dominant_embedding_model_id(session, [s.id for s in stories])
        vectors: list[np.ndarray] = []
        embeddings_index: list[int] = []
        dims: int | None = None
        for s in stories:
            if s.centroid is None:
                continue
            vec = unpack(s.centroid)
            if dims is None:
                dims = len(vec)
            elif len(vec) != dims:
                log.warning("publish: story=%s centroid dim mismatch, skipping "
                           "from embeddings", s.id)
                continue
            vectors.append(vec)
            embeddings_index.append(s.id)

        embeddings_rel: str | None = None
        embeddings_hash: str | None = None
        embeddings_bytes: bytes | None = None
        if vectors:
            embeddings_bytes = np.asarray(vectors, dtype=np.float32).tobytes()
            embeddings_hash = _hash_bytes(embeddings_bytes)
            window = now.strftime("%Y%m%d")
            embeddings_rel = f"embeddings/{window}-{embeddings_hash}.bin"

        sources_dict = _sources_report_dict(session, now=now)
        SourcesReport.model_validate(sources_dict)  # validated, written verbatim (no hash in name)
        sources_bytes = json.dumps(sources_dict, sort_keys=True, indent=2).encode("utf-8")

        manifest = Manifest.model_validate({
            "version": BUNDLE_VERSION,
            "generated_at": _iso(now),
            "embedding_model_id": embedding_model_id,
            "embedding_dimensions": dims,
            "story_count": len(stories),
            "pages": [
                {"page": n, "path": rel, "hash": h, "count": count}
                for n, (rel, _payload, h, count) in enumerate(page_files)
            ],
            "embeddings_path": embeddings_rel,
            "embeddings_hash": embeddings_hash,
            "embeddings_index": embeddings_index,
            "sources_path": "sources.json",
            "retention_days": cfg.retention_days,
        })
        manifest_bytes = json.dumps(manifest.model_dump(mode="json"), sort_keys=True,
                                    indent=2).encode("utf-8")
    except ValidationError as exc:
        result.published = False
        result.error = f"bundle schema validation failed, nothing published: {exc}"
        log.error(result.error)
        return result

    # --- everything validated; now write ---------------------------------
    for rel, payload, _h in story_files.values():
        _write(out / rel, payload)
    for rel, payload, _h, _count in page_files:
        _write(out / rel, payload)
    if embeddings_rel is not None:
        _write(out / embeddings_rel, embeddings_bytes)
    _write(out / "sources.json", sources_bytes)
    _write(out / "manifest.json", manifest_bytes)

    # --- prune anything content-addressed that this run didn't (re)write,
    # per spec 4.4's rolling window -- see module docstring for the
    # "unchanged content keeps its old hash and file" / "changed content
    # gets a new hash and orphans the old file" trade-off this implies.
    pruned = 0
    pruned += _prune_dir(out / "story", {rel.split("/", 1)[1] for rel, *_ in story_files.values()})
    pruned += _prune_dir(out / "feed", {rel.split("/", 1)[1] for rel, *_ in page_files})
    if embeddings_rel is not None:
        pruned += _prune_dir(out / "embeddings", {embeddings_rel.split("/", 1)[1]})
    else:
        pruned += _prune_dir(out / "embeddings", set())

    result.published = True
    result.story_count = len(stories)
    result.page_count = page_count
    result.pruned = pruned
    return result
