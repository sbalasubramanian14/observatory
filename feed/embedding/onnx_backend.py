from __future__ import annotations
import numpy as np


class OnnxEmbedder:
    def __init__(self, model: str, batch_size: int = 256):
        from fastembed import TextEmbedding
        self.model_id = model
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model)
        self.dimensions = int(next(iter(self._model.embed(["probe"]))).shape[0])

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.asarray(
            list(self._model.embed(texts, batch_size=self.batch_size)), dtype=np.float32
        )
