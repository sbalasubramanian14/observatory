"""Issue 3: the off-topic relevance gate (feed/stages/relevance.py).

Offline tests use a FakeEmbedder with a hand-controlled reference centroid
(no model download, no network -- the suite-wide default). A handful of
@pytest.mark.slow tests at the bottom load the real shipped CPU embedder
(ONNX + MiniLM) and check it against the actual text that prompted this
fix: the Verge film review ("'We're All Going to the World's Fair' Debuts
as Intimate Coming-of-Age Horror Film", categorised OTHER, score 48) must
be rejected, and genuine AI stories from the same source class must pass.
"""
from __future__ import annotations
import numpy as np
import pytest
from feed.config import RelevanceConfig
from feed.embedding.base import pack, unpack
from feed.models import Item, Source, Stage
from feed.stages.relevance import (
    AI_REFERENCE_TEXTS, gate_relevance, is_on_topic, keyword_hits,
)


class FakeEmbedder:
    """encode() returns a fixed reference vector for every one of the 5
    AI_REFERENCE_TEXTS (so the resulting centroid is exactly [1, 0, 0, 0]),
    and otherwise a caller-supplied default -- gate_relevance never encodes
    anything else in the offline path (items' embeddings are read directly
    from the stored column, not re-encoded), so that default is never
    actually hit in these tests.
    """
    model_id = "fake/relevance-v1"
    dimensions = 4

    def __init__(self):
        self.encode_calls = []

    def encode(self, texts):
        self.encode_calls.append(list(texts))
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))


class ExplodingReferenceEmbedder(FakeEmbedder):
    def encode(self, texts):
        raise RuntimeError("embedder backend unavailable")


def _seed_item(session, *, vec, title="headline", text="body", stage=Stage.EMBEDDED, source_id="s"):
    if session.get(Source, source_id) is None:
        session.add(Source(id=source_id, plugin="rss", config={}, cadence_minutes=30))
    item = Item(source_id=source_id, url=f"http://x/{title}", url_hash=f"h-{title}",
               title=title, text=text, embedding=pack(vec),
               embedding_model_id="fake/relevance-v1", stage=stage)
    session.add(item)
    session.commit()
    return item


# --- keyword_hits() -----------------------------------------------------

def test_keyword_hits_matches_ai_vocabulary_case_insensitively():
    assert keyword_hits("A new Machine Learning model was released") >= 1
    assert keyword_hits("OpenAI shipped a new model") >= 1


def test_keyword_hits_matches_bare_uppercase_ai_only():
    assert keyword_hits("A new AI model shipped") >= 1
    # lower-case "ai" as a bare token (not a real English word) must not
    # match -- only the phrases in KEYWORD_PHRASES or the upper-case
    # acronym count.
    assert keyword_hits("the committee will ai the proposal") == 0


def test_keyword_hits_does_not_match_ai_as_a_substring():
    # \b...\b must not fire on ordinary words that merely contain "ai".
    assert keyword_hits("We met again to discuss the detail of the plan") == 0


def test_keyword_hits_zero_for_unrelated_text():
    assert keyword_hits("This simple pasta recipe uses garlic and basil") == 0


def test_keyword_hits_handles_empty_and_none():
    assert keyword_hits("") == 0


# --- is_on_topic(): bias-to-keep OR logic --------------------------------

def test_is_on_topic_high_cosine_keeps_regardless_of_keywords():
    cfg = RelevanceConfig(cosine_threshold=0.12, min_keyword_hits=1)
    assert is_on_topic(0.5, 0, cfg, threshold=0.12) is True


def test_is_on_topic_keyword_hit_keeps_regardless_of_low_cosine():
    cfg = RelevanceConfig(cosine_threshold=0.12, min_keyword_hits=1)
    assert is_on_topic(-0.5, 1, cfg, threshold=0.12) is True


def test_is_on_topic_rejects_only_when_both_signals_fail():
    cfg = RelevanceConfig(cosine_threshold=0.12, min_keyword_hits=1)
    assert is_on_topic(0.05, 0, cfg, threshold=0.12) is False


# --- gate_relevance(): DB-level stage transitions ------------------------

def test_on_topic_item_is_kept_at_embedded(session):
    # Aligned with the fake centroid [1,0,0,0] -> cosine 1.0.
    item = _seed_item(session, vec=np.array([1, 0, 0, 0], dtype=np.float32),
                      title="ai-story", text="no ai words here at all")
    res = gate_relevance(session, RelevanceConfig(), FakeEmbedder())
    assert res.processed == 1
    assert res.rejected == 0
    refreshed = session.get(Item, item.id)
    assert refreshed.stage is Stage.EMBEDDED
    assert refreshed.reject_reason is None


def test_off_topic_item_is_rejected_with_a_reason(session):
    # Orthogonal to the fake centroid -> cosine 0.0, well below threshold,
    # and no AI keywords in title/text either.
    item = _seed_item(session, vec=np.array([0, 1, 0, 0], dtype=np.float32),
                      title="film review",
                      text="an intimate coming of age horror film")
    res = gate_relevance(session, RelevanceConfig(cosine_threshold=0.12), FakeEmbedder())
    assert res.processed == 1
    assert res.rejected == 1
    refreshed = session.get(Item, item.id)
    assert refreshed.stage is Stage.REJECTED
    assert refreshed.reject_reason is not None
    assert refreshed.reject_reason.startswith("off-topic:")


def test_keyword_signal_rescues_a_low_cosine_ai_item(session):
    # Same orthogonal, low-cosine vector as the rejected case above, but
    # the text explicitly says "artificial intelligence" -- the OR-on-keep
    # design must save it.
    item = _seed_item(session, vec=np.array([0, 1, 0, 0], dtype=np.float32),
                      title="Startup ships new artificial intelligence camera",
                      text="a surveillance camera firm's AI-driven monitoring")
    res = gate_relevance(session, RelevanceConfig(cosine_threshold=0.12), FakeEmbedder())
    assert res.rejected == 0
    assert session.get(Item, item.id).stage is Stage.EMBEDDED


def test_only_claims_embedded_items(session):
    item = _seed_item(session, vec=np.array([0, 1, 0, 0], dtype=np.float32),
                      title="not yet embedded", stage=Stage.NORMALIZED)
    res = gate_relevance(session, RelevanceConfig(), FakeEmbedder())
    assert res.processed == 0
    assert session.get(Item, item.id).stage is Stage.NORMALIZED


def test_disabled_gate_is_a_true_noop(session):
    item = _seed_item(session, vec=np.array([0, 1, 0, 0], dtype=np.float32),
                      title="off topic but gate disabled")
    res = gate_relevance(session, RelevanceConfig(enabled=False), FakeEmbedder())
    assert res.processed == 0
    assert res.rejected == 0
    assert session.get(Item, item.id).stage is Stage.EMBEDDED


def test_reference_centroid_failure_fails_open(session):
    """A broken embedder must not strand the EMBEDDED backlog -- items are
    left exactly where they were, not silently dropped or wrongly
    rejected.
    """
    item = _seed_item(session, vec=np.array([1, 0, 0, 0], dtype=np.float32),
                      title="whatever")
    res = gate_relevance(session, RelevanceConfig(), ExplodingReferenceEmbedder())
    assert res.processed == 0
    assert res.rejected == 0
    assert len(res.errors) == 1
    assert session.get(Item, item.id).stage is Stage.EMBEDDED


def test_reference_centroid_is_built_from_the_five_ai_topic_sentences(session):
    emb = FakeEmbedder()
    _seed_item(session, vec=np.array([1, 0, 0, 0], dtype=np.float32), title="x")
    gate_relevance(session, RelevanceConfig(), emb)
    assert emb.encode_calls == [AI_REFERENCE_TEXTS]


def test_rejected_items_are_excluded_from_the_next_cluster_stage_claim(session):
    """Mutation-adjacent check: a rejected item must actually leave the
    EMBEDDED claim pool cluster() reads from (Stage.EMBEDDED), not just
    carry a reason nobody enforces. Verified directly against the DB
    query cluster() uses, not by importing cluster() itself (keeps this
    test focused on the gate's own contract).
    """
    from sqlalchemy import select
    off_topic = _seed_item(session, vec=np.array([0, 1, 0, 0], dtype=np.float32),
                           title="film review", text="a horror film")
    on_topic = _seed_item(session, vec=np.array([1, 0, 0, 0], dtype=np.float32),
                          title="ai story", text="")
    gate_relevance(session, RelevanceConfig(cosine_threshold=0.12), FakeEmbedder())
    still_embedded = set(session.scalars(
        select(Item.id).where(Item.stage == Stage.EMBEDDED)
    ))
    assert still_embedded == {on_topic.id}
    assert off_topic.id not in still_embedded


# --- Real examples (slow: loads the shipped CPU embedder) ----------------

@pytest.mark.slow
def test_rejects_the_verge_film_review(session):
    """The actual story from the live-site bug report: 'We're All Going to
    the World's Fair' Debuts as Intimate Coming-of-Age Horror Film,
    theverge, categorised OTHER, score 48 -- exactly the item that should
    never have reached publish.
    """
    from feed.config import EmbeddingConfig
    from feed.embedding import build_embedder

    embedder = build_embedder(EmbeddingConfig(
        backend="onnx", model="sentence-transformers/all-MiniLM-L6-v2", device="cpu",
    ))
    title = "'We're All Going to the World's Fair' Debuts as Intimate Coming-of-Age Horror Film"
    text = (
        "The new movie 'We're All Going to the World's Fair' blends horror "
        "with a personal coming-of-age story, following a teenage "
        "protagonist navigating identity and digital spaces. Critics note "
        "its intimate tone and unsettling atmosphere."
    )
    vec = embedder.encode([f"{title}\n\n{text}"])[0]
    item = _seed_item(session, vec=vec, title=title, text=text, source_id="theverge")
    item.embedding_model_id = embedder.model_id
    session.commit()

    res = gate_relevance(session, RelevanceConfig(), embedder)
    assert res.rejected == 1
    assert session.get(Item, item.id).stage is Stage.REJECTED


@pytest.mark.slow
def test_keeps_genuine_ai_stories_from_the_same_source(session):
    from feed.config import EmbeddingConfig
    from feed.embedding import build_embedder

    embedder = build_embedder(EmbeddingConfig(
        backend="onnx", model="sentence-transformers/all-MiniLM-L6-v2", device="cpu",
    ))
    genuine = [
        ("OpenAI releases GPT-6 with major reasoning gains",
         "OpenAI's new model outperforms prior versions on coding and "
         "reasoning benchmarks, the company said."),
        ("Anthropic ships Claude Opus 5",
         "The new frontier model brings improved agentic tool use and "
         "longer context windows, Anthropic said in a blog post."),
        ("Nvidia unveils next-generation AI training chip",
         "The chipmaker says the new accelerator roughly doubles "
         "large-language-model training throughput over its predecessor."),
    ]
    ids = []
    for title, text in genuine:
        vec = embedder.encode([f"{title}\n\n{text}"])[0]
        item = _seed_item(session, vec=vec, title=title, text=text, source_id="theverge")
        item.embedding_model_id = embedder.model_id
        session.commit()
        ids.append(item.id)

    res = gate_relevance(session, RelevanceConfig(), embedder)
    assert res.rejected == 0
    for item_id in ids:
        assert session.get(Item, item_id).stage is Stage.EMBEDDED


# --- sweep_existing_corpus(): retroactive corpus cleanup -----------------

from feed.stages.relevance import sweep_existing_corpus  # noqa: E402


def _seed_clustered_story(session, *, vec, title, text="", source_id="s",
                          category="OTHER", score=48.0, item_stage=Stage.SCORED):
    from feed.models import Story
    from datetime import datetime, timezone
    if session.get(Source, source_id) is None:
        session.add(Source(id=source_id, plugin="rss", config={}, cadence_minutes=30))
    story = Story(title=title, first_seen=datetime.now(timezone.utc),
                  updated_at=datetime.now(timezone.utc), item_count=1, outlet_count=1,
                  category=category, score=score)
    session.add(story)
    session.flush()
    item = Item(source_id=source_id, url=f"http://x/{title}", url_hash=f"h-{title}",
               title=title, text=text, embedding=pack(vec),
               embedding_model_id="fake/relevance-v1", stage=item_stage, story_id=story.id)
    session.add(item)
    session.commit()
    return item, story


def test_sweep_dry_run_reports_without_mutating(session):
    item, story = _seed_clustered_story(
        session, vec=np.array([0, 1, 0, 0], dtype=np.float32),
        title="film review", text="an intimate coming of age horror film",
    )
    res = sweep_existing_corpus(session, RelevanceConfig(cosine_threshold=0.12),
                                FakeEmbedder(), apply=False)
    assert res.scanned == 1
    assert len(res.findings) == 1
    assert res.findings[0].item_id == item.id
    assert res.findings[0].story_title == "film review"
    assert res.stories_deleted == 0
    # Nothing touched.
    refreshed = session.get(Item, item.id)
    assert refreshed.stage is Stage.SCORED
    assert refreshed.story_id == story.id
    assert refreshed.reject_reason is None


def test_sweep_apply_deletes_now_empty_story(session):
    from feed.models import Story
    item, story = _seed_clustered_story(
        session, vec=np.array([0, 1, 0, 0], dtype=np.float32),
        title="film review", text="an intimate coming of age horror film",
    )
    story_id = story.id
    res = sweep_existing_corpus(session, RelevanceConfig(cosine_threshold=0.12),
                                FakeEmbedder(), apply=True)
    assert len(res.findings) == 1
    assert res.stories_deleted == 1
    refreshed = session.get(Item, item.id)
    assert refreshed.stage is Stage.REJECTED
    assert refreshed.reject_reason is not None
    assert refreshed.story_id is None
    assert session.get(Story, story_id) is None


def test_sweep_apply_keeps_story_alive_when_other_items_remain_on_topic(session):
    from feed.models import Story
    off, story = _seed_clustered_story(
        session, vec=np.array([0, 1, 0, 0], dtype=np.float32),
        title="off-topic sibling", text="unrelated content",
    )
    on = Item(source_id="s", url="http://x/on-topic", url_hash="h-on-topic",
             title="on-topic sibling", text="",
             embedding=pack(np.array([1, 0, 0, 0], dtype=np.float32)),
             embedding_model_id="fake/relevance-v1", stage=Stage.SCORED, story_id=story.id)
    session.add(on)
    session.commit()

    res = sweep_existing_corpus(session, RelevanceConfig(cosine_threshold=0.12),
                                FakeEmbedder(), apply=True)
    assert len(res.findings) == 1
    assert res.stories_deleted == 0
    refreshed_story = session.get(Story, story.id)
    assert refreshed_story is not None
    assert refreshed_story.item_count == 1
    assert refreshed_story.outlet_count == 1
    assert session.get(Item, off.id).stage is Stage.REJECTED
    assert session.get(Item, on.id).stage is Stage.SCORED


def test_sweep_is_safely_rerunnable(session):
    _seed_clustered_story(
        session, vec=np.array([0, 1, 0, 0], dtype=np.float32),
        title="film review", text="an intimate coming of age horror film",
    )
    sweep_existing_corpus(session, RelevanceConfig(cosine_threshold=0.12),
                          FakeEmbedder(), apply=True)
    second = sweep_existing_corpus(session, RelevanceConfig(cosine_threshold=0.12),
                                   FakeEmbedder(), apply=True)
    assert second.scanned == 0
    assert second.findings == []


def test_sweep_does_not_flag_on_topic_stories(session):
    _seed_clustered_story(
        session, vec=np.array([1, 0, 0, 0], dtype=np.float32),
        title="OpenAI ships new model", text="a frontier model release",
    )
    res = sweep_existing_corpus(session, RelevanceConfig(cosine_threshold=0.12),
                                FakeEmbedder(), apply=True)
    assert res.findings == []
    assert res.stories_deleted == 0


def test_sweep_can_be_scoped_to_specific_sources(session):
    off_scoped, _ = _seed_clustered_story(
        session, vec=np.array([0, 1, 0, 0], dtype=np.float32),
        title="film review", text="a horror film", source_id="theverge",
    )
    off_unscoped, _ = _seed_clustered_story(
        session, vec=np.array([0, 1, 0, 0], dtype=np.float32),
        title="arxiv paper", text="some paper", source_id="arxiv:ai",
    )
    res = sweep_existing_corpus(session, RelevanceConfig(cosine_threshold=0.12),
                                FakeEmbedder(), apply=False, source_ids=["theverge"])
    assert res.scanned == 1
    assert [f.item_id for f in res.findings] == [off_scoped.id]
    assert session.get(Item, off_unscoped.id).stage is Stage.SCORED
