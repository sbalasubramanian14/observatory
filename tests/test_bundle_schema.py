import pytest
from pydantic import ValidationError
from feed.bundle_schema import Manifest, StoryDetail, StoryEvidence

_EVIDENCE = {"id": 1, "title": "T", "url": "https://x/1", "source_id": "s",
            "published_at": None}


def test_story_evidence_accepts_the_allowed_fields():
    ev = StoryEvidence.model_validate(_EVIDENCE)
    assert ev.title == "T"


def test_story_evidence_rejects_full_article_text():
    """Spec 4.2 hard requirement: full article text must never enter the
    bundle. The schema is the safety net -- extra="forbid" means a `text`
    key anywhere in an evidence payload is a hard validation failure, not
    a silently-dropped extra field."""
    with pytest.raises(ValidationError):
        StoryEvidence.model_validate({**_EVIDENCE, "text": "full scraped article body"})


def test_story_evidence_rejects_item_summary_field():
    """item.summary (an unvetted RSS blurb) is also excluded -- only our
    own generated story-level summary is allowed to reach the bundle."""
    with pytest.raises(ValidationError):
        StoryEvidence.model_validate({**_EVIDENCE, "summary": "raw publisher blurb"})


def test_story_detail_rejects_reader_behaviour_fields():
    """Spec 4.2: reader behaviour (opens, dwell, dismissals, interest
    profile) must never enter the bundle."""
    base = {
        "id": 1, "title": "T", "first_seen": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00", "item_count": 1,
        "outlet_count": 1, "evidence": [],
    }
    StoryDetail.model_validate(base)  # sanity: valid without extras
    with pytest.raises(ValidationError):
        StoryDetail.model_validate({**base, "read_at": "2026-01-01T00:00:00+00:00"})


def test_manifest_requires_core_fields():
    with pytest.raises(ValidationError):
        Manifest.model_validate({})
