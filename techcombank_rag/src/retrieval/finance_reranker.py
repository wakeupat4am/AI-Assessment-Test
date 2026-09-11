from __future__ import annotations

import re
import time
import unicodedata
from typing import Any


YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
ACRONYM_RE = re.compile(r"(?<!\w)[A-ZĐ]{2,}(?!\w)")


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class FinanceAwareReranker:
    def __init__(
        self, retriever: Any, config: dict[str, Any], *, candidate_k: int = 20
    ) -> None:
        self.base = retriever
        self.config = config
        self.candidate_k = int(candidate_k)
        self.last_trace: dict[str, Any] | None = None
        self.last_retrieval_calls: list[Any] = []
        self.last_router_call = None

    def _features(self, query: str) -> dict[str, Any]:
        normalized = normalize_text(query)
        years = sorted(set(YEAR_RE.findall(query)))
        numbers = sorted(set(NUMBER_RE.findall(query)) - set(years))
        acronyms = sorted(set(ACRONYM_RE.findall(query)))
        metrics = [
            term
            for term in self.config["financial_terms"]
            if normalize_text(term) in normalized
        ]
        explanatory = any(
            marker in normalized for marker in self.config["explanatory_markers"]
        )
        return {
            "years": years,
            "numbers": numbers,
            "acronyms": acronyms,
            "metrics": metrics,
            "explanatory": explanatory,
        }

    @staticmethod
    def _recall(required: list[str], text: str) -> float:
        if not required:
            return 1.0
        return sum(normalize_text(value) in text for value in required) / len(required)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        started = time.perf_counter()
        requested_k = int(top_k or self.candidate_k)
        rows = self.base.retrieve(query, max(requested_k, self.candidate_k))
        features = self._features(query)
        base_weight = float(self.config["base_weight"])
        constraint_weight = float(self.config["constraint_weight"])
        component_weights = {
            "year": float(self.config["year_weight"]),
            "acronym": float(self.config["acronym_weight"]),
            "metric": float(self.config["metric_weight"]),
            "number": float(self.config["number_weight"]),
            "granularity": float(self.config["granularity_weight"]),
        }
        scored: list[dict[str, Any]] = []
        for row in rows:
            text = normalize_text(str(row.get("text", "")))
            granularities = {"metric", "row"}
            if features["explanatory"]:
                granularity_score = float(row.get("granularity") in {"block", "page"})
            else:
                granularity_score = float(row.get("granularity") in granularities)
            components = {
                "year": self._recall(features["years"], text),
                "acronym": self._recall(features["acronyms"], text),
                "metric": self._recall(features["metrics"], text),
                "number": self._recall(features["numbers"], text),
                "granularity": granularity_score,
            }
            constraint_score = sum(
                components[name] * component_weights[name] for name in component_weights
            )
            base_score = float(row.get("fusion_score_normalized", row.get("score", 0.0)))
            final_score = base_weight * base_score + constraint_weight * constraint_score
            item = dict(row)
            item.update(
                {
                    "finance_rerank_score": float(final_score),
                    "finance_constraint_score": float(constraint_score),
                    "finance_constraint_components": components,
                }
            )
            scored.append(item)
        scored.sort(
            key=lambda row: (-row["finance_rerank_score"], str(row["chunk_id"]))
        )
        if scored:
            best = scored[0]["finance_rerank_score"] or 1.0
            for rank, row in enumerate(scored, 1):
                row["finance_rerank_rank"] = rank
                row["score"] = float(row["finance_rerank_score"] / best)
        base_trace = (getattr(self.base, "last_trace", None) or {}).get("b4", {})
        output = scored[:requested_k]
        self.last_trace = {
            "b4": {
                **base_trace,
                "stage": "finance_aware_rerank",
                "query_features": features,
                "reranked_candidates": len(scored),
                "returned_candidates": len(output),
                "final_pages": [int(row["printed_page"]) for row in output],
                "total_latency_seconds": round(time.perf_counter() - started, 6),
            }
        }
        return output
