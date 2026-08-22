from __future__ import annotations
from typing import Protocol
import numpy as np


class Embedder(Protocol):
    model_id: str
    dimensions: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    # np.frombuffer returns a read-only view over `blob`. Copy so callers
    # get an array they can safely mutate (e.g. in-place normalisation)
    # without hitting a confusing "assignment destination is read-only"
    # error; the copy is cheap for the small vectors this project stores.
    return np.frombuffer(blob, dtype=np.float32).copy()
