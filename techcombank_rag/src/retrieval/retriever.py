from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import faiss

from src.config import Settings, get_settings
from src.retrieval.embedder import Embedder


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing index metadata: {path}. Run scripts/ingest.py.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class DenseRetriever:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        index_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.index_dir = Path(index_dir or self.settings.index_dir)
        index_path = self.index_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}. Run scripts/ingest.py.")
        metadata_path = self.index_dir / "metadata.json"
        indexed_model: str | None = None
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            indexed_model = metadata.get("embedding_model")
            if (
                indexed_model
                and self.settings.embedding_model
                and indexed_model != self.settings.embedding_model
            ):
                raise ValueError(
                    f"Index uses {indexed_model}, but EMBEDDING_MODEL is "
                    f"{self.settings.embedding_model}. Re-ingest or fix configuration."
                )
        self.index = faiss.read_index(str(index_path))
        self.chunks = load_jsonl(self.index_dir / "chunks.jsonl")
        if self.index.ntotal != len(self.chunks):
            raise ValueError("FAISS vector count does not match chunks.jsonl")
        if embedder is not None:
            self.embedder = embedder
        else:
            embedding_model = self.settings.embedding_model or indexed_model
            if not embedding_model:
                raise ValueError(
                    "EMBEDDING_MODEL is required when index metadata does not declare it."
                )
            self.embedder = Embedder(embedding_model, self.settings.embedding_device)
        self._query_cache: OrderedDict[str, Any] = OrderedDict()
        self._query_cache_size = max(0, self.settings.query_embedding_cache_size)

    def _query_vector(self, query: str):
        key = " ".join(query.casefold().split())
        if self._query_cache_size and key in self._query_cache:
            vector = self._query_cache.pop(key)
            self._query_cache[key] = vector
            return vector
        vector = self.embedder.encode_query(query)
        if self._query_cache_size:
            self._query_cache[key] = vector
            while len(self._query_cache) > self._query_cache_size:
                self._query_cache.popitem(last=False)
        return vector

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        return self.retrieve_by_vector(self._query_vector(query), top_k)

    def retrieve_by_vector(
        self, vector: Any, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Search with a caller-provided normalized vector.

        This keeps HyDE modular: the hypothetical passage is embedded with the
        same encoder as indexed passages, while FAISS/chunk loading stays here.
        """
        requested_k = top_k or self.settings.top_k
        k = min(requested_k, len(self.chunks))
        if k <= 0:
            return []
        scores, indices = self.index.search(vector, k)
        results: list[dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            result = dict(self.chunks[int(index)])
            result["score"] = float(score)
            results.append(result)
        return results
