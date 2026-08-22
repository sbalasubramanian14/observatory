from __future__ import annotations
from feed.config import EmbeddingConfig

CPU_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GPU_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve(cfg: EmbeddingConfig) -> tuple[str, str, str]:
    """Return (backend, model, device).

    Measured rationale (spec Appendix A): on CPU, ONNX + MiniLM is the fastest
    configuration at ~90 docs/s. With a GPU, torch + bge-small reaches ~203
    docs/s while also giving a 512-token window instead of 256, so the GPU
    buys the better model rather than only speed.
    """
    has_gpu = cuda_available()

    device = cfg.device
    if device == "auto":
        device = "cuda" if has_gpu else "cpu"
    elif device == "cuda" and not has_gpu:
        raise RuntimeError("cuda requested but no CUDA device is available")

    backend = cfg.backend
    if backend == "auto":
        backend = "torch" if device == "cuda" else "onnx"
    if backend == "onnx" and device == "cuda":
        raise RuntimeError("onnx backend is CPU-only in this project; use backend=torch")

    model = cfg.model
    # Only swap the model when the caller left it unset. Comparing against
    # the default *value* can't tell "unset" apart from "explicitly chose
    # the default", so it would silently override an explicit choice of
    # bge-small + onnx. model_fields_set records which fields were actually
    # provided, so use that instead.
    if "model" not in cfg.model_fields_set and backend == "onnx":
        # the bge ONNX export is anomalously slow (spec Appendix A); prefer MiniLM
        model = CPU_DEFAULT_MODEL
    return backend, model, device
