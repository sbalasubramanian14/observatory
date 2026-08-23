import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from feed.config import PublishConfig
from feed.embedding.base import pack
from feed.models import Item, Source, Story, StoryStatus
from feed.stages.publish import publish


def _seed_story(session, *, story_id_hint="s1", score=0.7, titles=("A outlet", "B outlet"),
                updated_at=None, centroid_dim=4, analysis=None, summary=None,
                extra_item_kwargs=None) -> Story:
    now = datetime.now(timezone.utc)
    updated_at = updated_at or now
    if session.get(Source, "src") is None:
        session.add(Source(id="src", plugin="rss", config={}, cadence_minutes=30))
        session.flush()
    story = Story(title=f"Story {story_id_hint}", first_seen=now, updated_at=updated_at,
                 item_count=len(titles), outlet_count=len(titles), score=score,
                 score_breakdown={"authority": 0.5}, summary=summary, analysis=analysis,
                 status=StoryStatus.ANALYZED if analysis else StoryStatus.NEW,
                 centroid=pack(np.ones(centroid_dim, dtype=np.float32)))
    session.add(story)
    session.flush()
    extra = extra_item_kwargs or {}
    for i, t in enumerate(titles):
        session.add(Item(source_id="src", url=f"http://x/{story_id_hint}/{i}",
                         url_hash=f"h-{story_id_hint}-{i}", title=t, story_id=story.id,
                         published_at=now, embedding_model_id="test-model/v1", **extra))
    session.commit()
    return story


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def test_publish_writes_a_valid_bundle(session, tmp_path):
    _seed_story(session, score=0.9)
    result = publish(session, PublishConfig(page_size=50), tmp_path)

    assert result.published is True
    assert result.story_count == 1
    manifest = _read_json(tmp_path / "manifest.json")
    assert manifest["story_count"] == 1
    assert manifest["embedding_model_id"] == "test-model/v1"
    assert len(manifest["pages"]) == 1

    page = _read_json(tmp_path / manifest["pages"][0]["path"])
    assert len(page["stories"]) == 1
    story_row = page["stories"][0]

    detail = _read_json(tmp_path / story_row["detail_path"])
    assert detail["id"] == story_row["id"]
    assert len(detail["evidence"]) == 2
    assert (tmp_path / "sources.json").exists()
    assert manifest["embeddings_path"] is not None
    assert (tmp_path / manifest["embeddings_path"]).exists()


def test_publish_never_writes_item_text_anywhere_on_disk(session, tmp_path):
    marker = "SECRET_FULL_ARTICLE_BODY_MUST_NEVER_BE_PUBLISHED"
    _seed_story(session, score=0.9, extra_item_kwargs={"text": marker})

    result = publish(session, PublishConfig(), tmp_path)
    assert result.published is True

    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert marker not in f.read_text(encoding="utf-8", errors="ignore")


def test_publish_never_writes_item_summary_field(session, tmp_path):
    """Only the story-level generated summary belongs in the bundle -- a
    raw RSS blurb on the item must not leak into evidence entries."""
    marker = "RAW_UNVETTED_RSS_BLURB"
    _seed_story(session, score=0.9, extra_item_kwargs={"summary": marker})

    result = publish(session, PublishConfig(), tmp_path)
    assert result.published is True

    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert marker not in f.read_text(encoding="utf-8", errors="ignore")


def test_publish_includes_story_generated_summary_and_analysis(session, tmp_path):
    _seed_story(session, score=0.9, summary="Generated two-sentence summary.",
               analysis="Deep why-this-matters analysis.")
    result = publish(session, PublishConfig(), tmp_path)
    manifest = _read_json(tmp_path / "manifest.json")
    page = _read_json(tmp_path / manifest["pages"][0]["path"])
    detail = _read_json(tmp_path / page["stories"][0]["detail_path"])
    assert detail["summary"] == "Generated two-sentence summary."
    assert detail["analysis"] == "Deep why-this-matters analysis."


def test_publish_excludes_stories_without_a_score(session, tmp_path):
    _seed_story(session, story_id_hint="unscored", score=None)
    result = publish(session, PublishConfig(), tmp_path)
    assert result.story_count == 0


def test_publish_excludes_stories_outside_the_retention_window(session, tmp_path):
    old = datetime.now(timezone.utc) - timedelta(days=200)
    _seed_story(session, story_id_hint="old", score=0.5, updated_at=old)
    result = publish(session, PublishConfig(retention_days=90), tmp_path)
    assert result.story_count == 0


def test_publish_ranks_feed_pages_by_score_descending(session, tmp_path):
    _seed_story(session, story_id_hint="low", score=0.2, titles=("low",))
    _seed_story(session, story_id_hint="high", score=0.95, titles=("high",))
    result = publish(session, PublishConfig(page_size=50), tmp_path)
    manifest = _read_json(tmp_path / "manifest.json")
    page = _read_json(tmp_path / manifest["pages"][0]["path"])
    scores = [s["score"] for s in page["stories"]]
    assert scores == sorted(scores, reverse=True)


def test_publish_paginates_by_page_size(session, tmp_path):
    for i in range(5):
        _seed_story(session, story_id_hint=f"s{i}", score=0.5 + i * 0.01, titles=(f"t{i}",))
    result = publish(session, PublishConfig(page_size=2), tmp_path)
    assert result.page_count == 3
    manifest = _read_json(tmp_path / "manifest.json")
    assert len(manifest["pages"]) == 3
    total = 0
    for p in manifest["pages"]:
        page = _read_json(tmp_path / p["path"])
        total += len(page["stories"])
    assert total == 5


def test_publish_filenames_are_content_addressed_and_stable_for_unchanged_content(session, tmp_path):
    _seed_story(session, score=0.7)
    r1 = publish(session, PublishConfig(), tmp_path)
    manifest1 = _read_json(tmp_path / "manifest.json")

    r2 = publish(session, PublishConfig(), tmp_path)
    manifest2 = _read_json(tmp_path / "manifest.json")

    assert manifest1["pages"][0]["hash"] == manifest2["pages"][0]["hash"]
    assert manifest1["pages"][0]["path"] == manifest2["pages"][0]["path"]


def test_publish_prunes_files_for_stories_that_fall_out_of_the_window(session, tmp_path):
    story = _seed_story(session, story_id_hint="fading", score=0.7)
    publish(session, PublishConfig(retention_days=90), tmp_path)
    story_files_before = list((tmp_path / "story").iterdir())
    assert len(story_files_before) == 1

    # Push the story's updated_at outside the window, then republish.
    story.updated_at = datetime.now(timezone.utc) - timedelta(days=200)
    session.commit()

    result = publish(session, PublishConfig(retention_days=90), tmp_path)
    assert result.story_count == 0
    assert result.pruned >= 1
    assert list((tmp_path / "story").iterdir()) == []


def test_publish_prunes_the_old_hash_when_story_content_changes(session, tmp_path):
    story = _seed_story(session, story_id_hint="evolving", score=0.7)
    publish(session, PublishConfig(), tmp_path)
    first_files = {f.name for f in (tmp_path / "story").iterdir()}
    assert len(first_files) == 1

    story.summary = "Now it has a summary, so its content hash changes."
    session.commit()
    publish(session, PublishConfig(), tmp_path)
    second_files = {f.name for f in (tmp_path / "story").iterdir()}

    assert len(second_files) == 1
    assert second_files != first_files  # old hash pruned, new hash written


def test_publish_includes_sources_health(session, tmp_path):
    session.add(Source(id="broken", plugin="rss", config={}, cadence_minutes=30,
                       consecutive_failures=3, last_error="boom"))
    session.commit()
    publish(session, PublishConfig(), tmp_path)
    sources = _read_json(tmp_path / "sources.json")
    ids = {s["id"] for s in sources["sources"]}
    assert "broken" in ids
    broken = next(s for s in sources["sources"] if s["id"] == "broken")
    assert broken["consecutive_failures"] == 3


# --- D0: lead image selection --------------------------------------------

def test_publish_omits_lead_image_when_no_item_has_one(session, tmp_path):
    _seed_story(session, story_id_hint="noimg", score=0.7)
    publish(session, PublishConfig(), tmp_path)
    manifest = _read_json(tmp_path / "manifest.json")
    page = _read_json(tmp_path / manifest["pages"][0]["path"])
    assert page["stories"][0]["lead_image_url"] is None
    detail = _read_json(tmp_path / page["stories"][0]["detail_path"])
    assert detail["lead_image_url"] is None


def test_publish_picks_lead_image_from_the_highest_authority_item(session, tmp_path):
    now = datetime.now(timezone.utc)
    session.add(Source(id="low_authority", plugin="rss", config={}, cadence_minutes=30,
                       authority=0.2))
    session.add(Source(id="high_authority", plugin="rss", config={}, cadence_minutes=30,
                       authority=0.9))
    session.flush()
    story = Story(title="Multi-source story", first_seen=now, updated_at=now,
                 item_count=2, outlet_count=2, score=0.8, score_breakdown={},
                 centroid=pack(np.ones(4, dtype=np.float32)))
    session.add(story)
    session.flush()
    session.add(Item(source_id="low_authority", url="http://x/low", url_hash="lead-low",
                     title="low", story_id=story.id, published_at=now,
                     embedding_model_id="test-model/v1",
                     image_url="https://img.example.com/low-authority.jpg"))
    session.add(Item(source_id="high_authority", url="http://x/high", url_hash="lead-high",
                     title="high", story_id=story.id, published_at=now,
                     embedding_model_id="test-model/v1",
                     image_url="https://img.example.com/high-authority.jpg"))
    session.commit()

    publish(session, PublishConfig(), tmp_path)
    manifest = _read_json(tmp_path / "manifest.json")
    page = _read_json(tmp_path / manifest["pages"][0]["path"])
    assert page["stories"][0]["lead_image_url"] == "https://img.example.com/high-authority.jpg"
    detail = _read_json(tmp_path / page["stories"][0]["detail_path"])
    assert detail["lead_image_url"] == "https://img.example.com/high-authority.jpg"


def test_publish_falls_back_to_the_only_item_with_an_image_regardless_of_authority(session, tmp_path):
    now = datetime.now(timezone.utc)
    session.add(Source(id="low2", plugin="rss", config={}, cadence_minutes=30, authority=0.1))
    session.add(Source(id="high2", plugin="rss", config={}, cadence_minutes=30, authority=0.9))
    session.flush()
    story = Story(title="Only one has an image", first_seen=now, updated_at=now,
                 item_count=2, outlet_count=2, score=0.8, score_breakdown={},
                 centroid=pack(np.ones(4, dtype=np.float32)))
    session.add(story)
    session.flush()
    session.add(Item(source_id="high2", url="http://x/high2", url_hash="lead-high2",
                     title="high, no image", story_id=story.id, published_at=now,
                     embedding_model_id="test-model/v1"))
    session.add(Item(source_id="low2", url="http://x/low2", url_hash="lead-low2",
                     title="low, has image", story_id=story.id, published_at=now,
                     embedding_model_id="test-model/v1",
                     image_url="https://img.example.com/only-one.jpg"))
    session.commit()

    publish(session, PublishConfig(), tmp_path)
    manifest = _read_json(tmp_path / "manifest.json")
    page = _read_json(tmp_path / manifest["pages"][0]["path"])
    assert page["stories"][0]["lead_image_url"] == "https://img.example.com/only-one.jpg"


def test_publish_refuses_to_write_anything_when_schema_validation_fails(session, tmp_path, monkeypatch):
    """Reproduces spec 6's requirement directly: if a payload smuggles in a
    disallowed field (simulating a future regression that tries to leak
    item.text), the whole publish must abort with nothing written -- not a
    half-written bundle and not a silent pass-through.
    """
    _seed_story(session, score=0.9)

    def _bad_evidence(story):
        return [{"id": 1, "title": "T", "url": "http://x", "source_id": "s",
                 "published_at": None, "text": "leaked full article body"}]

    monkeypatch.setattr("feed.stages.publish._evidence_for", _bad_evidence)

    result = publish(session, PublishConfig(), tmp_path)

    assert result.published is False
    assert result.error is not None
    assert not (tmp_path / "manifest.json").exists()
