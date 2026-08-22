from __future__ import annotations
from feed.config import EmbeddingConfig
from feed.embedding.base import Embedder, pack, unpack  # noqa: F401

# Alias on import: binding the name `resolve` here (matching the submodule's
# own name) would rebind the `feed.embedding.resolve` package attribute from
# the submodule to this function, breaking anything that does
# `monkeypatch.setattr("feed.embedding.resolve.cuda_available", ...)` or
# otherwise reaches the module via attribute access (e.g. pytest's
# monkeypatch dotted-path resolution). Import under a different local name
# so the submodule reference on the package stays intact.
from feed.embedding.resolve import resolve as _resolve


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    backend, model, device = _resolve(cfg)
    if backend == "onnx":
        from feed.embedding.onnx_backend import OnnxEmbedder
        return OnnxEmbedder(model, batch_size=cfg.batch_size)
    from feed.embedding.torch_backend import TorchEmbedder
    return TorchEmbedder(model, device=device, batch_size=cfg.batch_size)
