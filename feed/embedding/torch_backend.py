from __future__ import annotations
import numpy as np


class TorchEmbedder:
    def __init__(self, model: str, device: str = "cpu", batch_size: int = 256):
        from sentence_transformers import SentenceTransformer
        self.model_id = model
        self.batch_size = batch_size
        self._model = SentenceTransformer(model, device=device)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.asarray(
            self._model.encode(texts, batch_size=self.batch_size,
                               show_progress_bar=False),
            dtype=np.float32,
        )
