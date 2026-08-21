from pathlib import Path
import pytest
from feed.config import load_config


def test_loads_defaults_when_file_absent(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.embedding.device == "auto"
    assert cfg.embedding.batch_size == 256
    # merge_threshold is an empirically derived constant that Task 12 may
    # revise from the golden-set band. Assert it is sane, not exact, so the
    # two tasks are not coupled.
    assert 0.30 <= cfg.clustering.merge_threshold <= 0.70


def test_file_overrides_defaults(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text(
        "[embedding]\ndevice = \"cpu\"\nbatch_size = 64\n"
        "[clustering]\nmerge_threshold = 0.62\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.embedding.device == "cpu"
    assert cfg.embedding.batch_size == 64
    assert cfg.clustering.merge_threshold == 0.62
    assert cfg.embedding.model == "BAAI/bge-small-en-v1.5"  # untouched default


def test_rejects_unknown_device(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text("[embedding]\ndevice = \"tpu\"\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_clustering_weights_must_sum_to_one(tmp_path):
    p = tmp_path / "feed.toml"
    p.write_text(
        "[clustering]\ncosine_weight = 0.9\nentity_weight = 0.4\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_config(p)


def test_threshold_for_hit(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    cfg.clustering.merge_thresholds = {"BAAI/bge-small-en-v1.5": 0.695, "MiniLM": 0.412}
    assert cfg.clustering.threshold_for("BAAI/bge-small-en-v1.5") == 0.695
    assert cfg.clustering.threshold_for("MiniLM") == 0.412


def test_threshold_for_miss_falls_back_to_merge_threshold(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    cfg.clustering.merge_thresholds = {"MiniLM": 0.412}
    assert cfg.clustering.threshold_for("some-other-model") == cfg.clustering.merge_threshold


def test_threshold_for_none_falls_back_to_merge_threshold(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    cfg.clustering.merge_thresholds = {"MiniLM": 0.412}
    assert cfg.clustering.threshold_for(None) == cfg.clustering.merge_threshold


def test_merge_thresholds_defaults_to_empty_dict(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.clustering.merge_thresholds == {}
