from pathlib import Path

import numpy as np
import pytest
from feed.clustering.entities import extract_entities
from feed.clustering.signals import blend, cosine, entity_overlap
from feed.config import EmbeddingConfig
from feed.embedding import build_embedder
from tests.golden.corpus import CORPUS

MIN_BAND_WIDTH = 0.06   # measured 0.10 for the blend; 0.02 for cosine alone

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
    contiguous range that reproduces the ground-truth clusters exactly.

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

    working = [
        round(t, 2)
        for t in np.arange(0.20, 0.95, 0.01)
        if _union_find_clusters(scores, n, round(t, 2)) == truth_sets
    ]
    if not working:
        return (0.0, 0.0, 0.0)
    low, high = min(working), max(working)
    return (low, high, high - low)


@pytest.fixture(scope="module")
def scores():
    labels = [c[0] for c in CORPUS]
    texts = [c[2] for c in CORPUS]
    cfg = EmbeddingConfig(backend="onnx", model=MODEL_ID, device="cpu", batch_size=32)
    V = build_embedder(cfg).encode(texts)
    ents = [extract_entities(t) for t in texts]
    n = len(texts)
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = blend(
                cosine(V[i], V[j]), entity_overlap(ents[i], ents[j]),
                cosine_weight=0.6, entity_weight=0.4,
            )
    return labels, M, n


@pytest.mark.slow
def test_blend_recovers_ground_truth_over_a_usable_band(scores):
    labels, M, n = scores

    low, high, width = safe_band_width(M, labels)
    print(f"\nmeasured safe threshold band: low={low:.2f} high={high:.2f} width={width:.2f}")

    assert width > 0.0, (
        "No threshold recovers the labelled stories. The clustering signals "
        "have regressed below usability."
    )
    assert width >= MIN_BAND_WIDTH, (
        f"Safe threshold band narrowed to {width:.2f} (band {low:.2f}-{high:.2f}); "
        f"minimum is {MIN_BAND_WIDTH}. Clustering is now fragile even though it "
        f"still passes at some threshold."
    )


@pytest.mark.slow
def test_configured_threshold_sits_inside_the_working_band(scores):
    from feed.config import load_config

    labels, M, n = scores
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
    assert _union_find_clusters(M, n, configured) == truth_sets, (
        f"configured threshold_for({MODEL_ID!r})={configured} (from {FEED_TOML}) "
        f"does not recover ground truth"
    )
