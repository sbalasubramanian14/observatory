from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from feed.config import Config, EmbeddingConfig, load_config
from feed.embedding.base import pack, unpack
from feed.embedding.resolve import resolve
from feed.embedding import build_embedder

# The repo-root feed.toml, resolved from this file's location so the test is
# independent of the pytest invocation's cwd (see tests/golden/test_golden.py
# for the same pattern).
FEED_TOML = Path(__file__).resolve().parents[1] / "feed.toml"


def test_pack_roundtrips_exactly():
    v = np.random.rand(384).astype(np.float32)
    assert np.array_equal(unpack(pack(v)), v)


def test_unpack_returns_a_writable_array():
    # np.frombuffer hands back a read-only view over the bytes object; any
    # downstream code that tries to mutate a vector in place (e.g. an
    # in-place normalisation `vec /= norm`) would get a confusing
    # ValueError: assignment destination is read-only. unpack() copies so
    # callers get an array they can safely write to.
    v = np.random.rand(384).astype(np.float32)
    out = unpack(pack(v))
    assert out.flags.writeable
    out[0] = 123.0  # must not raise
    assert out[0] == 123.0


def test_explicit_settings_are_passed_through():
    cfg = EmbeddingConfig(backend="torch", model="BAAI/bge-small-en-v1.5", device="cpu")
    assert resolve(cfg) == ("torch", "BAAI/bge-small-en-v1.5", "cpu")


def test_auto_without_gpu_picks_onnx_minilm_cpu(monkeypatch):
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: False)
    cfg = EmbeddingConfig(backend="auto", device="auto")
    assert resolve(cfg) == ("onnx", "sentence-transformers/all-MiniLM-L6-v2", "cpu")


def test_auto_with_gpu_picks_torch_bge_cuda(monkeypatch):
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: True)
    cfg = EmbeddingConfig(backend="auto", device="auto")
    assert resolve(cfg) == ("torch", "BAAI/bge-small-en-v1.5", "cuda")


def test_explicit_cuda_without_gpu_raises(monkeypatch):
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda requested"):
        resolve(EmbeddingConfig(device="cuda", backend="torch"))


def test_onnx_backend_with_cuda_device_raises(monkeypatch):
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: True)
    with pytest.raises(RuntimeError, match="onnx"):
        resolve(EmbeddingConfig(backend="onnx", device="cuda"))


def test_unset_model_with_onnx_backend_swaps_to_minilm(monkeypatch):
    # model left unset -> the bge ONNX export is anomalously slow, so an
    # unset model should be swapped to the fast CPU default.
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: False)
    cfg = EmbeddingConfig(backend="onnx")
    assert resolve(cfg) == ("onnx", "sentence-transformers/all-MiniLM-L6-v2", "cpu")


def test_explicit_bge_model_with_onnx_backend_is_honoured(monkeypatch):
    # An explicitly-chosen model must never be silently overridden, even
    # though it is the slow-but-honoured bge-on-onnx combination.
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: False)
    cfg = EmbeddingConfig(model="BAAI/bge-small-en-v1.5", backend="onnx", device="cpu")
    assert resolve(cfg) == ("onnx", "BAAI/bge-small-en-v1.5", "cpu")


def test_load_config_unset_model_with_onnx_backend_swaps_to_minilm(tmp_path, monkeypatch):
    # Exercise the real production path (load_config -> Config.model_validate)
    # rather than only direct EmbeddingConfig(...) construction, since that is
    # the path that matters for model_fields_set correctness.
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: False)
    toml_path = tmp_path / "feed.toml"
    toml_path.write_text('[embedding]\nbackend = "onnx"\n')
    cfg = load_config(toml_path)
    assert "model" not in cfg.embedding.model_fields_set
    assert resolve(cfg.embedding) == (
        "onnx",
        "sentence-transformers/all-MiniLM-L6-v2",
        "cpu",
    )


def test_load_config_explicit_model_with_onnx_backend_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: False)
    toml_path = tmp_path / "feed.toml"
    toml_path.write_text(
        '[embedding]\nbackend = "onnx"\nmodel = "BAAI/bge-small-en-v1.5"\n'
    )
    cfg = load_config(toml_path)
    assert "model" in cfg.embedding.model_fields_set
    assert resolve(cfg.embedding) == ("onnx", "BAAI/bge-small-en-v1.5", "cpu")


@pytest.mark.slow
def test_onnx_embedder_produces_normalisable_vectors():
    cfg = EmbeddingConfig(
        backend="onnx",
        model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        batch_size=8,
    )
    emb = build_embedder(cfg)
    V = emb.encode(["DeepSeek releases V4", "EU delays the AI Act"])
    assert V.shape == (2, emb.dimensions)
    assert emb.model_id.endswith("all-MiniLM-L6-v2")
    sim = float(V[0] @ V[1] / (np.linalg.norm(V[0]) * np.linalg.norm(V[1])))
    assert -1.0 <= sim <= 1.0


@pytest.mark.slow
def test_torch_embedder_produces_correct_shape_and_handles_empty():
    # TorchEmbedder had never been executed (only its selection path via a
    # mocked resolve() was covered). This exercises the real backend
    # end-to-end on CPU: construction, encode() shape/model_id/dimensions,
    # and the empty-input branch that returns shape (0, dimensions) rather
    # than raising.
    from feed.embedding.torch_backend import TorchEmbedder

    emb = TorchEmbedder(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu", batch_size=8
    )
    V = emb.encode(["DeepSeek releases V4", "EU delays the AI Act"])
    assert V.shape == (2, emb.dimensions)
    assert emb.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert emb.dimensions > 0

    empty = emb.encode([])
    assert empty.shape == (0, emb.dimensions)


def test_shipped_feed_toml_resolves_to_the_intended_cpu_profile(monkeypatch):
    """C2: feed.toml previously pinned `model = "BAAI/bge-small-en-v1.5"`
    explicitly. Because resolve() only swaps an unset model, that explicit
    choice survived onto every CPU machine and forced the (onnx, bge-small,
    cpu) combination -- measured at 2.8 docs/s, ~6 minutes per 1000 items,
    32x slower than the intended (onnx, MiniLM, cpu) profile at 89.6 docs/s.
    It also meant threshold_for('BAAI/bge-small-en-v1.5') returned the
    global 0.50 fallback instead of the 0.35 actually measured for MiniLM
    on the golden corpus, since that key never matched.

    This pins what the REAL, DEPLOYED feed.toml resolves to on a CPU
    machine (no CUDA) end to end through load_config() -> resolve() ->
    threshold_for(), so a future re-introduction of an explicit `model =`
    line in feed.toml fails this test immediately instead of silently
    stranding the measured threshold again.
    """
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: False)
    cfg = load_config(FEED_TOML)

    assert "model" not in cfg.embedding.model_fields_set, (
        "feed.toml must not pin an explicit embedding model -- doing so "
        "defeats resolve()'s CPU/GPU auto-selection (see C2)"
    )

    backend, model, device = resolve(cfg.embedding)
    assert (backend, model, device) == (
        "onnx",
        "sentence-transformers/all-MiniLM-L6-v2",
        "cpu",
    )
    assert cfg.clustering.threshold_for(model) == 0.35


def test_shipped_feed_toml_global_threshold_sits_inside_bges_measured_safe_band():
    """C2: the global merge_threshold (used whenever a model id has no
    entry in merge_thresholds -- i.e. the GPU/bge-small path, since only
    MiniLM has an override) must sit inside the safe band the spike run
    actually measured for bge-small: 0.46-0.56 (see
    .superpowers/sdd/2026-08-22-ai-news-feed-phase1/progress.md, the F3
    pre-flight ruling). feed.toml ships 0.50, which is inside that band.
    This does not re-run the (expensive, bge-specific) spike measurement --
    there is no committed bge golden corpus to re-derive it from -- but it
    pins the shipped constant against the recorded measurement so a future
    edit to either the toml value or this test's asserted band cannot drift
    apart silently.
    """
    cfg = load_config(FEED_TOML)
    BGE_SAFE_BAND_LOW, BGE_SAFE_BAND_HIGH = 0.46, 0.56
    assert BGE_SAFE_BAND_LOW <= cfg.clustering.merge_threshold <= BGE_SAFE_BAND_HIGH
