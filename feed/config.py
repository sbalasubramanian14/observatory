# feed/config.py
from __future__ import annotations
import tomllib
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///feed.db"


class EmbeddingConfig(BaseModel):
    backend: Literal["auto", "torch", "onnx"] = "auto"
    model: str = "BAAI/bge-small-en-v1.5"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    batch_size: int = Field(default=256, gt=0, le=1024)


class CollectConfig(BaseModel):
    """Spec A4 (phaseA-report): a global cap on how far back a source's
    first fetch (or a fetch after a long gap) is allowed to reach.

    Without this, a machine off for three weeks asks every source for three
    weeks of history in one go (arXiv/HN volume makes that expensive and,
    for HN's search API, still incomplete), and a brand-new source drags in
    its entire archive (OpenAI's RSS feed returned 1,143 items back to 2015
    the first time it was added). The owner's instruction: "if the gap is
    large, at least it should pull 2 days data" -- so the default is 2, not
    0 or unbounded. See feed.stages.collect._effective_since, which applies
    this uniformly to every source, and Source.max_backfill_days, which
    lets one source override it individually.
    """
    max_backfill_days: int = Field(default=2, gt=0)


class ClusteringConfig(BaseModel):
    window_hours: int = Field(default=48, gt=0)
    cosine_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    entity_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    merge_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    # Keyed by embedding model id. A single merge_threshold cannot serve two
    # embedding models: the two models this project uses produce different
    # similarity scales (measured: bge-small same-story minimum 0.695,
    # MiniLM 0.412). merge_threshold remains the fallback when a model id is
    # absent from this map (or when no model id is given).
    #
    # The pydantic-level default here stays an empty dict (see
    # tests/test_config.py::test_merge_thresholds_defaults_to_empty_dict) --
    # the actual production value for
    # sentence-transformers/all-MiniLM-L6-v2 (the CPU default) lives in
    # feed.toml's [clustering.merge_thresholds] table: 0.35, the midpoint of
    # the safe threshold band measured by tests/golden/test_golden.py on the
    # 22-item golden corpus with the 0.6*cosine + 0.4*entity_overlap blend --
    # band observed as low=0.30 high=0.40 width=0.10 (>= MIN_BAND_WIDTH
    # 0.06). Rerun that test and update feed.toml's value plus this comment
    # if the corpus, weights, or model change.
    merge_thresholds: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "ClusteringConfig":
        total = self.cosine_weight + self.entity_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"cosine_weight + entity_weight must sum to 1.0, got {total}"
            )
        return self

    def threshold_for(self, model_id: str | None) -> float:
        if model_id is not None and model_id in self.merge_thresholds:
            return self.merge_thresholds[model_id]
        return self.merge_threshold


class RelevanceConfig(BaseModel):
    """The off-topic gate (Issue 3): a cheap, pre-LLM filter that runs once
    per item, right after Tier 0 embedding and before clustering, so a
    general-purpose feed accidentally added to the catalogue (or a niche
    off-topic item slipping through an otherwise AI-labelled feed) doesn't
    silently pollute the AI news feed -- e.g. a Verge film review that
    reached publish as an OTHER-category story, score 48.

    Design (see feed/stages/relevance.py for the implementation):
    combines two independent, cheap signals rather than one:

    1. Embedding cosine similarity to a small, fixed "AI topic" reference
       centroid -- reuses the embedding Tier 0 already computed (spec: "no
       per-item LLM call is acceptable" at ~1,700 items/day). Catches
       topical drift a keyword list can't ("a piece about a surveillance
       camera startup's AI-driven monitoring" has no exact keyword hit but
       sits close to the AI centroid).
    2. A curated keyword/entity match against the item's own text --
       cheap, deterministic, and immune to embedding-model drift or a
       reference centroid that ages poorly. Catches the inverse case: a
       short, generic-sounding item that happens to be squarely about AI
       (mentions "GPT" or "Anthropic" by name) but whose embedding sits in
       an ambiguous region of the space.

    An item is rejected ONLY if BOTH signals say off-topic (cosine below
    threshold AND zero keyword hits) -- deliberately an AND on the reject
    condition (an OR on the "keep" condition), per the owner's explicit
    instruction to bias toward keeping: wrongly dropping a real AI story
    is invisible and unrecoverable (a reader never sees what they never
    got), wrongly keeping an off-topic one is merely annoying and visible
    on the page. cosine_threshold defaults low (0.12) for the same reason
    -- tuned against real examples (see tests/test_relevance_stage.py and
    the live sweep in the Issue 3 report): the offending Verge film review
    measured 0.089 against the shipped MiniLM reference centroid, while a
    borderline-but-genuine AI/surveillance story measured 0.166 and every
    clearly-AI story measured 0.27-0.65. 0.12 sits in the gap, closer to
    the reject side than a naive midpoint would, so a story that's even
    loosely AI-adjacent clears it.
    """
    enabled: bool = True
    cosine_threshold: float = Field(default=0.12, ge=-1.0, le=1.0)
    # Keyed by embedding model id, mirroring ClusteringConfig.merge_thresholds
    # -- different embedding models produce different similarity scales, so
    # one global cosine_threshold cannot be assumed to serve every model.
    # Falls back to cosine_threshold when the active model id is absent.
    cosine_thresholds: dict[str, float] = Field(default_factory=dict)
    # A single hit on the curated AI-vocabulary list is enough to call an
    # item on-topic regardless of its embedding similarity -- see the
    # class docstring's "OR on keep" rationale.
    min_keyword_hits: int = Field(default=1, ge=1)

    def threshold_for(self, model_id: str | None) -> float:
        if model_id is not None and model_id in self.cosine_thresholds:
            return self.cosine_thresholds[model_id]
        return self.cosine_threshold


class ScoringConfig(BaseModel):
    # entity is pinned to 0.0, not 0.15, DELIBERATELY: nothing in Phase 1
    # populates the Entity/StoryEntity tables (feed.scoring.signals.
    # entity_weight() always falls back to its 0.5 default), so a nonzero
    # weight here would bake a constant into every story's score. combine()
    # divides by the sum of these weights, so with entity at 0.0 the
    # remaining three signals' denominator is 0.85 and the achievable score
    # range is exactly [0, 1] -- not the [0.075, 0.925] a nonzero constant
    # signal would compress it to. Phase 1's whole point is deriving
    # absolute thresholds from this distribution, so that compression is
    # not acceptable. Re-enable this weight in one line once something
    # actually writes Entity/StoryEntity rows -- do not restore it before
    # that, or the compression comes back silently.
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "authority": 0.25,
            "velocity": 0.40,
            "novelty": 0.20,
            "entity": 0.0,
        }
    )


class BulkProviderConfig(BaseModel):
    """One entry in the BULK (Tier 1) failover chain -- priority is simply
    list order in feed.toml's [[providers.bulk]] array. Model names, base
    URLs, and env var names all live here rather than hardcoded in
    provider classes: stale model names broke this project three separate
    times in one day (a dead gemini-2.0-flash, a Cerebras model that does
    not exist, two OpenRouter model names that 404'd), and a config value
    is a one-line fix where a hardcoded constant is a code change.
    """
    name: str
    kind: Literal["openai_compatible", "gemini"]
    model: str
    # Required for kind="openai_compatible" (feed.providers.openai_compatible.
    # OpenAICompatibleProvider needs it); unused for kind="gemini", which
    # has its endpoint baked into feed.providers.gemini.ENDPOINT.
    base_url: str | None = None
    env_var: str
    enabled: bool = True
    timeout: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def base_url_required_for_openai_compatible(self) -> "BulkProviderConfig":
        if self.kind == "openai_compatible" and not self.base_url:
            raise ValueError(
                f"providers.bulk[{self.name!r}]: base_url is required for "
                "kind='openai_compatible'"
            )
        return self


class ProvidersConfig(BaseModel):
    """Spec 3.5 LLM tiering, extended with multi-provider BULK failover.
    BULK (Tier 1, once per story) tries `bulk` in priority order with
    automatic failover (see feed.providers.failover.FailoverProvider);
    Claude Code remains the single DEEP provider (Tier 2, budgeted). No
    section carries a secret -- every provider's key comes from the
    environment / .env via its own `env_var`, never from feed.toml, so
    this config is safe to commit.
    """
    # Pydantic-level default is deliberately an empty list, matching the
    # existing convention for clustering.merge_thresholds: the real chain
    # -- Groq, Mistral, OpenRouter, Gemini, and a disabled-by-default
    # Cerebras -- lives in feed.toml's [[providers.bulk]] tables, not here.
    bulk: list[BulkProviderConfig] = Field(default_factory=list)
    # Bounded retry-with-backoff (requirement 5) applied inside each
    # provider's own complete() before the failover chain advances.
    max_retries: int = Field(default=2, ge=0)
    backoff_base: float = Field(default=0.5, gt=0)
    # Requirement 2: consecutive 429s before a provider is skipped for the
    # rest of the UTC day. A single 402 always disables immediately,
    # regardless of this value.
    rate_limit_disable_threshold: int = Field(default=3, gt=0)
    # Retained for backward compatibility (existing tests / a bare feed.toml
    # with no [[providers.bulk]] entries); _build_router in feed/cli.py no
    # longer reads these when `bulk` is populated -- Gemini's model/timeout
    # then come from its own BulkProviderConfig entry instead.
    gemini_model: str = "gemini-flash-latest"
    gemini_timeout: float = Field(default=30.0, gt=0)
    claude_code_timeout: float = Field(default=120.0, gt=0)
    # Tier 2 eligibility: a story's importance score (0..1, see scoring)
    # must clear this cut to be a Tier 2 candidate at all. Spec 9.6 says
    # this "cannot be guessed and must come from a live score distribution"
    # -- 0.6 is a placeholder starting point, tune after observing real
    # scores.
    tier2_score_cut: float = Field(default=0.6, ge=0.0, le=1.0)
    # Spec 3.5 / 9.2: "Tier 2 daily budget -- starts at 20 stories/day, to
    # be tuned against observed Claude Code rate limits." Budgeted by call
    # count, not tokens, because Claude Code is subscription-billed.
    daily_budget: int = Field(default=20, gt=0)


class PublishConfig(BaseModel):
    """Spec 4.1/4.4: the static bundle the Publish stage writes."""
    out_dir: str = "public"
    # Spec 4.4: "prunes files outside a rolling window (default 90 days)".
    retention_days: int = Field(default=90, gt=0)
    page_size: int = Field(default=50, gt=0)


class Config(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    collect: CollectConfig = Field(default_factory=CollectConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    relevance: RelevanceConfig = Field(default_factory=RelevanceConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    publish: PublishConfig = Field(default_factory=PublishConfig)


def load_config(path: Path | None = None) -> Config:
    path = Path(path) if path is not None else Path("feed.toml")
    if not path.exists():
        return Config()
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return Config.model_validate(raw)
