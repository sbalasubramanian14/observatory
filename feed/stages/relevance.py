from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from feed.config import RelevanceConfig
from feed.embedding.base import Embedder, pack, unpack
from feed.models import Item, Stage, Story
from feed.stages.base import StageResult

log = logging.getLogger(__name__)

# Reference sentences spanning the AI-news beat this project exists to
# cover (README / feed.toml's own framing: frontier model releases, the
# labs and chip makers behind them, and AI policy) -- deliberately a
# handful of short, generic sentences describing the *topic* rather than
# the vocabulary of any single story, so the resulting centroid represents
# "AI news" in general, not one event. Recomputed once per gate_relevance()
# call (five short strings through an already-loaded embedder -- negligible
# next to Tier 0's own per-item cost) rather than cached across runs, so it
# always matches whichever embedding model is active.
AI_REFERENCE_TEXTS = [
    "Artificial intelligence and machine learning research, large language "
    "models, neural networks, and AI systems.",
    "A new AI model from OpenAI, Anthropic, Google DeepMind, or Meta AI is "
    "announced, trained on large datasets using deep learning and "
    "transformers.",
    "AI chatbots, generative AI, GPT, LLM agents, and machine learning "
    "algorithms powering artificial intelligence products.",
    "Nvidia GPUs, AI chips, and the infrastructure powering artificial "
    "intelligence training and inference at scale.",
    "AI policy, regulation, safety, and the societal impact of artificial "
    "intelligence and automation.",
]

# Cheap, predictable keyword/entity signal -- deliberately broad (see
# RelevanceConfig's docstring on bias-to-keep: a false-positive keyword
# hit just means an item skips the embedding check, not that an off-topic
# item gets published outright, since publish still depends on the item
# clearing clustering/scoring on its own merits). Matched case-insensitively.
KEYWORD_PHRASES = [
    "artificial intelligence", "machine learning", "deep learning",
    "neural network", "large language model", "generative ai", "genai",
    "chatgpt", "chatbot", "openai", "anthropic", "deepmind", "claude",
    "gemini", "copilot", "llm", "gpt", "transformer model",
    "foundation model", "diffusion model", "reinforcement learning",
    "computer vision", "natural language processing", "nlp",
    "nvidia", "ai chip", "ai model", "ai-driven", "ai-powered",
]
_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in KEYWORD_PHRASES) + r")\b",
    re.IGNORECASE,
)
# The bare two-letter acronym "AI" is matched separately and
# case-SENSITIVE (upper-case only): matching it case-insensitively would
# trip on ordinary words that merely contain the letters (regex \b...\b
# already stops it matching inside "again" or "detail", but would still
# match standalone lower-case "ai", which isn't a real word in English
# text and would only ever be a false negative-avoidance measure with no
# real benefit -- while genuinely risking odd casing in transcribed
# quotes). A story that only ever refers to "artificial intelligence" in
# lower case still has that full phrase in KEYWORD_PHRASES above.
_BARE_AI_RE = re.compile(r"\bAI\b")


def keyword_hits(text: str) -> int:
    """Count of AI-vocabulary matches in `text`. >=1 is a strong,
    human-auditable signal that an item is on-topic -- see
    RelevanceConfig.min_keyword_hits.
    """
    if not text:
        return 0
    return len(_KEYWORD_RE.findall(text)) + len(_BARE_AI_RE.findall(text))


def relevance_text_for(item: Item) -> str:
    return f"{item.title}\n\n{item.text or ''}".strip()


def build_reference_centroid(embedder: Embedder) -> np.ndarray:
    """Encode AI_REFERENCE_TEXTS with `embedder` and return the unit-norm
    mean vector -- the "AI topic" point every item's embedding is compared
    against via cosine similarity.
    """
    vecs = np.asarray(embedder.encode(AI_REFERENCE_TEXTS), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    centroid = vecs.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return centroid
    return centroid / centroid_norm


def cosine_to_centroid(vec: np.ndarray, centroid: np.ndarray) -> float:
    norm = np.linalg.norm(vec)
    if norm == 0 or not np.isfinite(vec).all():
        return 0.0
    return float(np.dot(vec, centroid) / norm)


def is_on_topic(cos: float, hits: int, cfg: RelevanceConfig, *, threshold: float) -> bool:
    """Bias-to-keep: on-topic if EITHER signal says so. Rejecting requires
    BOTH the embedding and the keyword signal to agree the item is
    off-topic. See RelevanceConfig's docstring for the full rationale.
    """
    return cos >= threshold or hits >= cfg.min_keyword_hits


def gate_relevance(session: Session, cfg: RelevanceConfig, embedder: Embedder, *,
                   limit: int | None = None) -> StageResult:
    """Claim items at Stage.EMBEDDED; leave on-topic ones there (unchanged,
    so feed.stages.cluster's own Stage.EMBEDDED claim query picks them up
    next), route off-topic ones to the terminal Stage.REJECTED with an
    auditable Item.reject_reason instead of silently dropping them.

    Deliberately called ONCE per `feed run`, not looped through
    feed.stages.base.drain() the way normalize/embed/cluster/score are.
    drain() stops when a round makes zero progress, on the assumption that
    every claimed item LEAVES the stage it was claimed at -- true for an
    off-topic item (-> Stage.REJECTED) but NOT for an on-topic one, which
    stays at Stage.EMBEDDED by design (so cluster's own claim query still
    finds it). Looping that through drain() would just re-examine the same
    kept items every round until the round-cap fired, doing nothing but
    burning cycles. `limit=None` (the default) matches that "one pass over
    everything currently pending" model: cosine + keyword matching against
    an embedding Tier 0 already computed is cheap enough that, unlike
    embed's own batch_size, there is no real benefit to capping this call
    the way a compute-heavy encode() batch needs to be.

    Disabled via cfg.enabled=False is a true no-op (returns immediately,
    claims nothing) -- e.g. for a deployment that would rather rely on
    curated feeds alone, or while tuning the threshold offline.
    """
    result = StageResult(name="relevance")
    if not cfg.enabled:
        return result

    stmt = select(Item).where(Item.stage == Stage.EMBEDDED).order_by(Item.id)
    if limit is not None:
        stmt = stmt.limit(limit)
    items = list(session.scalars(stmt))
    if not items:
        return result

    try:
        centroid = build_reference_centroid(embedder)
    except Exception as exc:
        # Fail open: a broken reference-centroid build (e.g. a transient
        # embedder hiccup) must not strand the whole EMBEDDED backlog
        # behind a gate that can never pass judgment. Every claimed item
        # is left exactly where it was (still Stage.EMBEDDED) so the next
        # run's cluster stage still reaches it -- consistent with "bias
        # toward keeping" at the level of the whole gate, not just one
        # item's threshold.
        log.warning("relevance gate: failed to build reference centroid, "
                    "skipping this round (items left at Stage.EMBEDDED): %s", exc)
        result.errors.append((0, f"reference centroid build failed: {exc}"))
        return result

    threshold = cfg.threshold_for(embedder.model_id)

    for item in items:
        try:
            vec = unpack(item.embedding)
            cos = cosine_to_centroid(vec, centroid)
            hits = keyword_hits(relevance_text_for(item))
            if is_on_topic(cos, hits, cfg, threshold=threshold):
                item.reject_reason = None
            else:
                item.stage = Stage.REJECTED
                item.reject_reason = (
                    f"off-topic: cosine={cos:.3f} (threshold {threshold:.3f}), "
                    f"keyword_hits={hits}"
                )
                result.rejected += 1
            session.commit()
            result.processed += 1
        except Exception as exc:
            session.rollback()
            fresh = session.get(Item, item.id)
            if fresh is not None:
                fresh.stage = Stage.FAILED
                fresh.error = f"relevance gate failed: {type(exc).__name__}: {exc}"
                session.commit()
            result.failed += 1
            result.errors.append((item.id, str(exc)))
            log.warning("relevance gate item=%s failed: %s", item.id, exc)

    return result


# ---------------------------------------------------------------------------
# Corpus sweep: retroactively apply the same signal to items that already
# advanced past Stage.EMBEDDED before this gate existed (Issue 3's "clean
# the existing corpus" ask).
# ---------------------------------------------------------------------------

@dataclass
class SweepFinding:
    item_id: int
    title: str
    source_id: str
    story_id: int | None
    story_title: str | None
    story_category: str | None
    story_score: float | None
    reason: str


@dataclass
class SweepResult:
    scanned: int = 0
    findings: list[SweepFinding] = field(default_factory=list)
    stories_deleted: int = 0
    applied: bool = False


def sweep_existing_corpus(session: Session, cfg: RelevanceConfig, embedder: Embedder, *,
                          apply: bool = False,
                          source_ids: list[str] | None = None) -> SweepResult:
    """One-off (but safely re-runnable) retroactive pass of the exact same
    signal gate_relevance() uses, applied to items that already advanced
    PAST Stage.EMBEDDED and into a story before this gate existed -- e.g.
    the Verge film review ("'We're All Going to the World's Fair' Debuts
    as Intimate Coming-of-Age Horror Film") that reached publish as an
    OTHER-category story, score 48, off the whole-site (not AI-specific)
    feed that predated the sources.catalogue.toml fix shipped alongside
    this gate.

    Dry-run by default (apply=False): every offending item is reported in
    `result.findings` without touching the database at all -- "report how
    many there are before removing them" is exactly what a dry-run answers.
    apply=True then detaches each offending item from its story (Stage ->
    REJECTED, reject_reason set, story_id cleared) and recomputes that
    story's aggregates from its remaining members, or deletes the story
    outright if it becomes empty -- the common case, since most single-
    source drift like the film review is a single-item, single-outlet
    story to begin with. Safely re-runnable: an item already at
    Stage.REJECTED is never re-scanned (the candidate query only looks at
    Stage.CLUSTERED/SCORED, i.e. items still actually in a story).

    `source_ids`, if given, scopes the scan to just those sources -- use
    this to keep a cleanup surgical. A blanket sweep with the default
    threshold over EVERY source is not always the right call: sources
    whose catalogue entry was already AI-scoped correctly (e.g. an arXiv
    category feed, or a GitHub releases feed for an inference-engine repo)
    can carry short, jargon-heavy titles that this gate's cheap signals
    were never tuned against and would flag as false positives -- exactly
    the "wrongly dropping a real story" failure this project biases
    against. Passing the specific source(s) known to have carried the bug
    (e.g. the two whose catalogue URL this same change fixed) keeps the
    cleanup scoped to the actual root cause instead of relitigating every
    other source's editorial judgment.
    """
    result = SweepResult(applied=apply)
    centroid = build_reference_centroid(embedder)

    conditions = [
        Item.stage.in_([Stage.CLUSTERED, Stage.SCORED]),
        Item.embedding.is_not(None),
        Item.story_id.is_not(None),
    ]
    if source_ids:
        conditions.append(Item.source_id.in_(source_ids))
    candidates = list(session.scalars(
        select(Item).where(*conditions).order_by(Item.id)
    ))
    result.scanned = len(candidates)

    affected_story_ids: set[int] = set()
    for item in candidates:
        threshold = cfg.threshold_for(item.embedding_model_id)
        vec = unpack(item.embedding)
        cos = cosine_to_centroid(vec, centroid)
        hits = keyword_hits(relevance_text_for(item))
        if is_on_topic(cos, hits, cfg, threshold=threshold):
            continue

        story = item.story
        reason = (f"off-topic: cosine={cos:.3f} (threshold {threshold:.3f}), "
                  f"keyword_hits={hits}")
        result.findings.append(SweepFinding(
            item_id=item.id, title=item.title, source_id=item.source_id,
            story_id=story.id if story else None,
            story_title=story.title if story else None,
            story_category=story.category if story else None,
            story_score=story.score if story else None,
            reason=reason,
        ))
        if apply:
            story_id = item.story_id
            item.stage = Stage.REJECTED
            item.reject_reason = reason
            item.story_id = None
            if story_id is not None:
                affected_story_ids.add(story_id)

    if apply and affected_story_ids:
        session.flush()
        for story_id in affected_story_ids:
            story = session.get(Story, story_id)
            if story is None:
                continue
            members = list(session.scalars(
                select(Item).where(Item.story_id == story.id)
            ))
            if not members:
                session.delete(story)
                result.stories_deleted += 1
                continue
            story.item_count = len(members)
            story.outlet_count = len({m.source_id for m in members})
            dated = [m.published_at for m in members if m.published_at]
            if dated:
                story.updated_at = max(dated)
            vecs = [unpack(m.embedding) for m in members if m.embedding]
            if vecs:
                story.centroid = pack(np.mean(np.array(vecs, dtype=np.float32), axis=0))
        session.commit()
    elif apply:
        session.commit()

    return result
