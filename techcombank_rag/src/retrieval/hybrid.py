from __future__ import annotations

import time
from typing import Any


class HybridRRFRetriever:
    def __init__(
        self,
        dense_retriever: Any,
        bm25_retriever: Any,
        *,
        candidate_k: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        if candidate_k <= 0 or rrf_k <= 0:
            raise ValueError("candidate_k and rrf_k must be positive")
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.candidate_k = int(candidate_k)
        self.rrf_k = int(rrf_k)
        self.dense_weight = float(dense_weight)
        self.bm25_weight = float(bm25_weight)
        self.last_trace: dict[str, Any] | None = None
        self.last_retrieval_calls: list[Any] = []
        self.last_router_call = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        started = time.perf_counter()
        requested_k = int(top_k or self.candidate_k)
        dense_rows = self.dense.retrieve(query, self.candidate_k)
        bm25_rows = self.bm25.retrieve(query, self.candidate_k)
        fused: dict[str, dict[str, Any]] = {}
        for source, rows, weight in (
            ("dense", dense_rows, self.dense_weight),
            ("bm25", bm25_rows, self.bm25_weight),
        ):
            for rank, row in enumerate(rows, 1):
                key = str(row["chunk_id"])
                entry = fused.setdefault(
                    key,
                    {
                        "row": dict(row),
                        "fusion_score": 0.0,
                        "retrieval_sources": [],
                        "source_scores": {},
                    },
                )
                entry["fusion_score"] += weight / (self.rrf_k + rank)
                entry["retrieval_sources"].append({"source": source, "rank": rank})
                entry["source_scores"][source] = float(row.get("score", 0.0))
                if source == "dense":
                    entry["row"] = dict(row)
        ordered = sorted(
            fused.values(),
            key=lambda item: (-item["fusion_score"], str(item["row"]["chunk_id"])),
        )[:requested_k]
        best_fusion = ordered[0]["fusion_score"] if ordered else 1.0
        results: list[dict[str, Any]] = []
        for rank, entry in enumerate(ordered, 1):
            row = dict(entry["row"])
            normalized_fusion = entry["fusion_score"] / best_fusion
            row.update(
                {
                    "score": float(max(normalized_fusion, *entry["source_scores"].values())),
                    "fusion_score": float(entry["fusion_score"]),
                    "fusion_score_normalized": float(normalized_fusion),
                    "fusion_rank": rank,
                    "retrieval_sources": entry["retrieval_sources"],
                    "source_scores": entry["source_scores"],
                    "retrieval_source": "dense_bm25_rrf",
                }
            )
            results.append(row)
        self.last_trace = {
            "b4": {
                "stage": "hybrid_rrf",
                "query": query,
                "dense_candidates": len(dense_rows),
                "bm25_candidates": len(bm25_rows),
                "union_candidates": len(fused),
                "returned_candidates": len(results),
                "rrf_k": self.rrf_k,
                "dense_weight": self.dense_weight,
                "bm25_weight": self.bm25_weight,
                "final_pages": [int(row["printed_page"]) for row in results],
                "latency_seconds": round(time.perf_counter() - started, 6),
            }
        }
        return results
