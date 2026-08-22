from __future__ import annotations
import enum
from typing import Callable, Protocol


class Verdict(enum.Enum):
    SAME = "same"
    DIFFERENT = "different"
    AMBIGUOUS = "ambiguous"


class Adjudicator(Protocol):
    def decide(self, pair_score: float, left, right) -> Verdict: ...


class ThresholdAdjudicator:
    """Blended-score thresholding with an explicit uncertainty band.

    The band exists because the spike measured a *negative* separation margin
    for cosine alone: no single threshold cleanly divides same-story from
    different-story pairs. Scores inside the band are honestly reported as
    AMBIGUOUS rather than forced to a guess.

    `threshold_for`, when given, is consulted as `threshold_for(left.embedding_model_id)`
    to resolve the effective threshold per pair -- this is how a caller wires
    in ClusteringConfig.threshold_for, so that model-scoped policy lives here
    (the one object whose job is thresholds) rather than being smuggled in by
    pre-shifting `pair_score` before it reaches `decide()`. `pair_score` is
    always the raw, uninterpreted blended similarity: the Adjudicator protocol
    is the seam for a future LLM adjudicator, and a value that has been
    silently shifted by an unrelated caller is not a score that seam, or
    anything else, could interpret, log, or calibrate against.

    Falls back to `self.merge_threshold` when no `threshold_for` was given, or
    when `left` is `None` (as in the unit tests below, which call `decide`
    directly with synthetic floats and no items).
    """

    def __init__(self, merge_threshold: float = 0.50, ambiguous_band: float = 0.06,
                 *, threshold_for: Callable[[str | None], float] | None = None):
        self.merge_threshold = merge_threshold
        self.ambiguous_band = ambiguous_band
        self.threshold_for = threshold_for

    def _effective_threshold(self, left) -> float:
        if self.threshold_for is not None and left is not None:
            return self.threshold_for(getattr(left, "embedding_model_id", None))
        return self.merge_threshold

    def decide(self, pair_score: float, left=None, right=None) -> Verdict:
        threshold = self._effective_threshold(left)
        low = threshold - self.ambiguous_band / 2
        high = threshold + self.ambiguous_band / 2
        if pair_score >= high:
            return Verdict.SAME
        if pair_score <= low:
            return Verdict.DIFFERENT
        return Verdict.AMBIGUOUS


class NullAdjudicator:
    """Phase 1 seam. Resolves AMBIGUOUS to DIFFERENT.

    Splitting a story wrongly produces two visible rows the reader can see and
    reconcile. Merging wrongly silently hides an event, which violates success
    criterion 1. When uncertain, split. Phase 2 replaces this with an LLM
    adjudicator that actually answers the question.
    """

    def __init__(self, inner: Adjudicator):
        self.inner = inner

    def decide(self, pair_score: float, left=None, right=None) -> Verdict:
        verdict = self.inner.decide(pair_score, left, right)
        return Verdict.DIFFERENT if verdict is Verdict.AMBIGUOUS else verdict
