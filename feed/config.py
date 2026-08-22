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


class ProvidersConfig(BaseModel):
    """Spec 3.5 LLM tiering. Gemini is the BULK provider (Tier 1, once per
    story); Claude Code is the DEEP provider (Tier 2, budgeted). Neither
    section carries a secret -- the Gemini key comes from the environment /
    .env, never from feed.toml, so this config is safe to commit.
    """
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
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
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
