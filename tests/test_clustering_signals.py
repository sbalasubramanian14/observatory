from datetime import datetime, timedelta, timezone
import numpy as np
from feed.clustering.entities import extract_entities
from feed.clustering.signals import (blend, cosine, entity_overlap, link_overlap,
                                     time_proximity)

def test_extract_entities_finds_orgs_and_model_names():
    ents = extract_entities("DeepSeek releases V4, an open-weights MoE model.")
    assert "deepseek" in ents
    assert "v4" in ents

def test_extract_entities_drops_leading_stopwords():
    ents = extract_entities("The Commission postponed the deadline.")
    assert "the" not in ents
    assert "commission" in ents

def test_extract_entities_is_case_insensitive_in_output():
    assert extract_entities("NVIDIA beats estimates") == extract_entities("Nvidia beats estimates")

def test_cosine_of_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine(v, v) == 1.0

def test_cosine_handles_zero_vector_without_dividing_by_zero():
    z = np.zeros(3, dtype=np.float32)
    assert cosine(z, np.array([1.0, 0, 0], dtype=np.float32)) == 0.0

def test_entity_overlap_is_jaccard():
    assert entity_overlap({"a", "b"}, {"a", "b"}) == 1.0
    assert entity_overlap({"a", "b"}, {"b", "c"}) == 1 / 3
    assert entity_overlap(set(), set()) == 0.0

def test_link_overlap_detects_shared_source_document():
    a = ["https://huggingface.co/deepseek/v4", "https://x.com/a"]
    b = ["https://huggingface.co/deepseek/v4"]
    assert link_overlap(a, b) > 0.0
    assert link_overlap(a, ["https://unrelated.example"]) == 0.0

def test_time_proximity_decays_to_zero_at_window_edge():
    t = datetime(2026, 8, 19, 12, tzinfo=timezone.utc)
    assert time_proximity(t, t, window_hours=48) == 1.0
    assert time_proximity(t, t + timedelta(hours=48), window_hours=48) == 0.0
    mid = time_proximity(t, t + timedelta(hours=24), window_hours=48)
    assert 0.4 < mid < 0.6

def test_blend_matches_the_measured_weights():
    # spec Appendix A: 0.6*cosine + 0.4*entities
    assert blend(1.0, 0.0, cosine_weight=0.6, entity_weight=0.4) == 0.6
    assert blend(0.0, 1.0, cosine_weight=0.6, entity_weight=0.4) == 0.4

def test_cosine_returns_zero_never_one_for_nan_or_inf_contaminated_vectors():
    # A raw nan comparison (nan >= threshold) fails SAFE: always False, so
    # nothing merges. A clamp built on Python's min/max would launder that
    # nan into a fabricated 1.0 (min(1.0, nan) == 1.0), which fails UNSAFE:
    # it would force a merge of unrelated stories on corrupted input. Every
    # case below must be 0.0, never 1.0.
    nan_vec = np.array([np.nan, 1.0, 2.0], dtype=np.float32)
    other_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    zero_vec = np.zeros(3, dtype=np.float32)
    inf_vec = np.array([np.inf, 1.0, 2.0], dtype=np.float32)
    single_nan = np.array([1.0, np.nan, 3.0], dtype=np.float32)

    assert cosine(nan_vec, nan_vec) == 0.0
    assert cosine(nan_vec, other_vec) == 0.0
    assert cosine(nan_vec, zero_vec) == 0.0
    assert cosine(inf_vec, inf_vec) == 0.0
    assert cosine(single_nan, other_vec) == 0.0
    # zero-vector case (no nan/inf involved) must still be 0.0, not broken
    # by the finiteness check.
    assert cosine(zero_vec, other_vec) == 0.0
