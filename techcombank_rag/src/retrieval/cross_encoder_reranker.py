from __future__ import annotations

import math
import time
from typing import Any


class CrossEncoderReranker:
    def __init__(
        self,
        retriever: Any,
        *,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        candidate_k: int = 20,
        threads: int = 8,
        model: Any | None = None,
    ) -> None:
        import torch

        torch.set_num_threads(max(1, int(threads)))
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(
                model_name,
                device=device,
                max_length=max_length,
                trust_remote_code=True,
            )
        self.base = retriever
        self.model = model
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.candidate_k = int(candidate_k)
        self.threads = max(1, int(threads))
        self.last_trace: dict[str, Any] | None = None
        self.last_retrieval_calls: list[Any] = []
        self.last_router_call = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        started = time.perf_counter()
        requested_k = int(top_k or self.candidate_k)
        rows = self.base.retrieve(query, max(requested_k, self.candidate_k))
        if not rows:
            return []
        scores = self.model.predict(
            [(query, str(row.get("text", ""))) for row in rows],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scored: list[dict[str, Any]] = []
        for row, raw_score in zip(rows, scores):
            value = float(raw_score)
            item = dict(row)
            item["cross_encoder_score"] = value
            scored.append(item)
        scored.sort(key=lambda row: (-row["cross_encoder_score"], str(row["chunk_id"])))
        highest = scored[0]["cross_encoder_score"]
        lowest = scored[-1]["cross_encoder_score"]
        span = highest - lowest
        for rank, row in enumerate(scored, 1):
            normalized = (
                (row["cross_encoder_score"] - lowest) / span if span > 1e-12 else 1.0
            )
            row["cross_encoder_rank"] = rank
            value = row["cross_encoder_score"]
            sigmoid = (
                1.0 / (1.0 + math.exp(-value))
                if value >= 0
                else math.exp(value) / (1.0 + math.exp(value))
            )
            row["score"] = float(max(normalized, sigmoid))
        base_trace = (getattr(self.base, "last_trace", None) or {}).get("b4", {})
        output = scored[:requested_k]
        self.last_trace = {
            "b4": {
                **base_trace,
                "stage": "cross_encoder_rerank",
                "cross_encoder_model": self.model_name,
                "cross_encoder_threads": self.threads,
                "cross_encoder_candidates": len(scored),
                "returned_candidates": len(output),
                "final_pages": [int(row["printed_page"]) for row in output],
                "total_latency_seconds": round(time.perf_counter() - started, 6),
            }
        }
        return output
