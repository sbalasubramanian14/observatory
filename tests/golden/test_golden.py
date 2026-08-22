from pathlib import Path

import numpy as np
import pytest
from feed.clustering.entities import extract_entities
from feed.clustering.signals import blend, cosine, entity_overlap
from feed.config import EmbeddingConfig
from feed.embedding import build_embedder
from tests.golden.corpus import CORPUS

# Absolute sanity floor only -- NOT the primary guard. See
# test_blend_recovers_ground_truth_over_a_usable_band for why: an absolute
# constant here was found (Task 12 review) to be model-specific and
# non-self-calibrating. The 0.10 figure below is this golden corpus measured
# against sentence-transformers/all-MiniLM-L6-v2 (the model this test pins).
# A prior version of this comment claimed "0.02 for cosine alone" -- that
# figure came from a bge-small spike run, a DIFFERENT model on a different
# similarity scale, and does not apply here. The actual MiniLM cosine-only
# width, measured on this corpus, is recorded next to the comparative
# assertion below.
MIN_BAND_WIDTH = 0.06   # measured 0.10 for the MiniLM blend on this corpus

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# The repo-root feed.toml, resolved from this file's location so the test is
# independent of the pytest invocation's cwd.
FEED_TOML = Path(__file__).resolve().parents[2] / "feed.toml"


def _union_find_clusters(scores, n, threshold):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if scores[i][j] >= threshold:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return {frozenset(v) for v in groups.values()}


def safe_band_width(scores, labels) -> tuple[float, float, float]:
    """Sweep merge thresholds and return (low, high, width) of the widest
    CONTIGUOUS range of thresholds that all reproduce the ground-truth
    clusters exactly.

    Contiguity is enforced, not just claimed: thresholds are swept in fixed
    0.01 steps, the passing ones are grouped into consecutive runs, and the
    widest run is returned. This matters because min/max over all passing
    thresholds would silently overstate the band if a gap exists (e.g.
    passes at 0.30-0.34, fails at 0.35, passes again at 0.40) -- a caller
    reading (low=0.30, high=0.40, width=0.10) would wrongly believe every
    threshold in between is safe, when 0.35 is not. That number is now
    load-bearing for the comparative assertion below, so an overstated band
    would make the test give false confidence in exactly the failure mode
    it exists to catch.

    This measures fragility, not just correctness: a signal that only
    recovers the right clusters at one razor-thin threshold is unusable in
    production, where the "right" threshold drifts with corpus composition,
    embedding model, and minor wording differences between outlets covering
    the same story. The band width is the actual margin for error.
    """
    n = len(labels)
    truth: dict[str, list[int]] = {}
    for i, l in enumerate(labels):
        truth.setdefault(l, []).append(i)
    truth_sets = {frozenset(v) for v in truth.values()}

    step = 0.01
    sweep = [round(t, 2) for t in np.arange(0.20, 0.95, step)]
    working = [t for t in sweep if _union_find_clusters(scores, n, t) == truth_sets]
    if not working:
        return (0.0, 0.0, 0.0)

    # Group into contiguous runs (consecutive multiples of `step`, tolerant
    # of float error) and keep the widest run.
    runs: list[list[float]] = [[working[0]]]
    for t in working[1:]:
        if round(t - runs[-1][-1], 2) == step:
            runs[-1].append(t)
        else:
            runs.append([t])
    best = max(runs, key=lambda run: run[-1] - run[0])
    low, high = best[0], best[-1]
    return (low, high, round(high - low, 2))


@pytest.fixture(scope="module")
def scores():
    labels = [c[0] for c in CORPUS]
    texts = [c[2] for c in CORPUS]
    cfg = EmbeddingConfig(backend="onnx", model=MODEL_ID, device="cpu", batch_size=32)
    V = build_embedder(cfg).encode(texts)
    ents = [extract_entities(t) for t in texts]
    n = len(texts)

    M_blend = [[0.0] * n for _ in range(n)]
    M_cosine_only = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cos_ij = cosine(V[i], V[j])
            ent_ij = entity_overlap(ents[i], ents[j])
            M_blend[i][j] = blend(cos_ij, ent_ij, cosine_weight=0.6, entity_weight=0.4)
            # cosine-only counterfactual: what this same corpus/model would
            # measure if the entity signal contributed nothing at all. Used
            # by test_blend_recovers_ground_truth_over_a_usable_band as the
            # baseline the blend must beat -- this is what makes that test
            # catch a regression that zeroes out the entity signal, which an
            # absolute-width-only gate would not.
            M_cosine_only[i][j] = blend(cos_ij, ent_ij, cosine_weight=1.0, entity_weight=0.0)
    return labels, M_blend, M_cosine_only, n


@pytest.mark.slow
def test_blend_recovers_ground_truth_over_a_usable_band(scores):
    labels, M_blend, M_cosine_only, n = scores

    blend_low, blend_high, blend_width = safe_band_width(M_blend, labels)
    cos_low, cos_high, cos_width = safe_band_width(M_cosine_only, labels)
    print(
        f"\nmeasured safe threshold band (blend):       "
        f"low={blend_low:.2f} high={blend_high:.2f} width={blend_width:.2f}"
    )
    print(
        f"measured safe threshold band (cosine-only):  "
        f"low={cos_low:.2f} high={cos_high:.2f} width={cos_width:.2f}"
    )

    assert blend_width > 0.0, (
        "No threshold recovers the labelled stories. The clustering signals "
        "have regressed below usability."
    )
    assert blend_width >= MIN_BAND_WIDTH, (
        f"Safe threshold band narrowed to {blend_width:.2f} (band "
        f"{blend_low:.2f}-{blend_high:.2f}); minimum is {MIN_BAND_WIDTH}. "
        f"Clustering is now fragile even though it still passes at some "
        f"threshold."
    )

    # PRIMARY GUARD. This is comparative, not absolute, and is the actual
    # point of this test file (see module docstring in corpus.py and the
    # Task 12 brief): the invariant we care about is "the entity signal
    # contributes materially to separability", not "the width exceeds some
    # constant". An absolute constant does not self-calibrate when the
    # model or corpus changes -- e.g. a Task-12-review finding: on THIS
    # corpus with THIS entity extractor, cosine-only alone already measures
    # width 0.07, comfortably above a naively-chosen MIN_BAND_WIDTH of 0.06.
    # A regression that dropped the entity signal to zero would have shipped
    # green under an absolute-only gate. The comparative assertion below
    # would catch it: blend_width would collapse to equal cos_width (margin
    # 0.0), which fails the >= margin check.
    #
    # Measured on this corpus/model (see task-12-report.md fix-report
    # section for the run that produced these numbers): blend width 0.10,
    # cosine-only width 0.07, margin 0.03. MARGIN below is set to 0.02 --
    # below the measured 0.03 to leave headroom for run-to-run measurement
    # noise (rounding to the nearest 0.01 threshold step), while still well
    # above 0.0 so a total collapse of the entity signal's contribution is
    # caught. This is not a round number picked in the abstract; it is
    # 2/3 of the one real measurement taken, deliberately conservative.
    MARGIN = 0.02
    assert blend_width - cos_width >= MARGIN, (
        f"Blend band (width {blend_width:.2f}) is not meaningfully wider than "
        f"the cosine-only band (width {cos_width:.2f}); margin "
        f"{blend_width - cos_width:.2f} < required {MARGIN}. The entity "
        f"signal is no longer contributing materially to separability -- "
        f"this is the fragile-clustering regression this test exists to "
        f"catch, even though the blend alone still clears MIN_BAND_WIDTH."
    )


@pytest.mark.slow
def test_configured_threshold_sits_inside_the_working_band(scores):
    from feed.config import load_config

    labels, M_blend, M_cosine_only, n = scores
    truth: dict[str, list[int]] = {}
    for i, l in enumerate(labels):
        truth.setdefault(l, []).append(i)
    truth_sets = {frozenset(v) for v in truth.values()}

    # Read the real, deployed feed.toml via load_config()/threshold_for(),
    # NOT a bare ClusteringConfig().merge_threshold. Two reasons:
    #
    # 1. threshold_for(model_id) over merge_threshold: this golden test pins
    #    the CPU model (MiniLM), and merge_thresholds exists specifically
    #    because bge-small (GPU default) and MiniLM (CPU default) sit on
    #    different similarity scales (see feed/config.py). Reading
    #    .merge_threshold directly would check the GPU-tuned global fallback
    #    against a CPU-model score matrix -- the wrong comparison -- and, if
    #    "fixed" by overwriting merge_threshold to fit MiniLM, would
    #    silently break clustering for the GPU/bge-small path instead.
    # 2. load_config(feed.toml) over the bare pydantic default: the
    #    ClusteringConfig.merge_thresholds default stays an empty dict (see
    #    tests/test_config.py::test_merge_thresholds_defaults_to_empty_dict,
    #    an existing Task-3 test this task must not break); the reconciled
    #    0.35 value for MiniLM lives only in feed.toml, the actual deployed
    #    config. So the "configured threshold" this test must check against
    #    is what load_config() returns for the real feed.toml, not what the
    #    bare in-code default returns.
    cfg = load_config(FEED_TOML)
    configured = cfg.clustering.threshold_for(MODEL_ID)
    assert _union_find_clusters(M_blend, n, configured) == truth_sets, (
        f"configured threshold_for({MODEL_ID!r})={configured} (from {FEED_TOML}) "
        f"does not recover ground truth"
    )
