from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    """Thin wrapper for normalized multilingual E5 embeddings."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        # Deterministic single-thread inference avoids a native OpenMP crash seen
        # on Apple Silicon; clean-machine users may explicitly override it.
        torch.set_num_threads(max(1, int(os.getenv("EMBEDDING_THREADS", "1"))))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode_documents(
        self, texts: Sequence[str], batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        prefixed = [f"passage: {text}" for text in texts]
        return self._encode(prefixed, batch_size, show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([f"query: {query}"], batch_size=1, show_progress=False)

    def _encode(
        self, texts: Sequence[str], batch_size: int, show_progress: bool
    ) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")
