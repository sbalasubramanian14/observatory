from __future__ import annotations
import enum
from typing import Protocol


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
    """

    def __init__(self, merge_threshold: float = 0.50, ambiguous_band: float = 0.06):
        self.merge_threshold = merge_threshold
        self.ambiguous_band = ambiguous_band

    def decide(self, pair_score: float, left=None, right=None) -> Verdict:
        low = self.merge_threshold - self.ambiguous_band / 2
        high = self.merge_threshold + self.ambiguous_band / 2
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
