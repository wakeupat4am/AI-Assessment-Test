from __future__ import annotations

import time
from typing import Any

from src.retrieval.metric_identity import (
    candidate_matches_constraints,
    extract_metric_label,
    normalize_search_text,
    normalized_terms,
    query_constraints,
)


class MetricAwareRetriever:
    """Rerank a bounded hybrid pool using label, qualifier, year and statement identity."""

    def __init__(self, base: Any, config: dict[str, Any], *, candidate_k: int = 40) -> None:
        self.base = base
        self.config = config
        self.candidate_k = int(candidate_k)
        self.last_trace = None
        self.last_retrieval_calls: list[Any] = []
        self.last_router_call = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        started = time.perf_counter()
        requested_k = int(top_k or self.candidate_k)
        rows = self.base.retrieve(normalize_search_text(query), self.candidate_k)
        constraints = query_constraints(query)
        query_terms = set(constraints["terms"])
        scored: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            label = str(item.get("metric_label") or extract_metric_label(item))
            label_terms = set(normalized_terms(label))
            union = query_terms | label_terms
            label_overlap = len(query_terms & label_terms) / len(union) if union else 0.0
            exact_label = float(bool(label and normalize_search_text(label).casefold() in constraints["normalized"].casefold()))
            candidate_text = normalize_search_text(str(item.get("search_text") or item.get("text", ""))).casefold()
            query_qualifiers = constraints["qualifiers"]
            qualifier_coverage = (
                sum(qualifier in candidate_text for qualifier in query_qualifiers)
                / len(query_qualifiers)
                if query_qualifiers
                else 1.0
            )
            years = constraints["years"]
            year_coverage = sum(year in candidate_text for year in years) / len(years) if years else 1.0
            requested_type = constraints["statement_type"]
            candidate_type = str(item.get("statement_type", "unspecified"))
            statement_match = float(
                requested_type == "unspecified"
                or candidate_type in {requested_type, "unspecified"}
            )
            base_score = float(item.get("fusion_score_normalized", item.get("score", 0.0)))
            score = (
                float(self.config.get("base_weight", 0.48)) * base_score
                + float(self.config.get("label_weight", 0.22)) * label_overlap
                + float(self.config.get("exact_label_weight", 0.10)) * exact_label
                + float(self.config.get("qualifier_weight", 0.14)) * qualifier_coverage
                + float(self.config.get("year_weight", 0.03)) * year_coverage
                + float(self.config.get("statement_weight", 0.03)) * statement_match
            )
            constraint_pass = candidate_matches_constraints(query, item)
            if query_qualifiers and not constraint_pass:
                score -= float(self.config.get("missing_qualifier_penalty", 0.35))
            item.update(
                {
                    "score": score,
                    "metric_aware_score": score,
                    "metric_constraint_pass": constraint_pass,
                    "metric_score_components": {
                        "base": base_score,
                        "label_overlap": label_overlap,
                        "exact_label": exact_label,
                        "qualifier_coverage": qualifier_coverage,
                        "year_coverage": year_coverage,
                        "statement_match": statement_match,
                    },
                    "retrieval_source": "dense_bm25_rrf_metric_aware",
                }
            )
            scored.append(item)
        scored.sort(key=lambda row: (-float(row["metric_aware_score"]), str(row["chunk_id"])))
        results = scored[:requested_k]
        self.last_trace = {
            "b4": {
                "stage": "metric_aware",
                "query": query,
                "constraints": constraints,
                "candidate_count": len(rows),
                "returned_candidates": len(results),
                "final_pages": [int(row["printed_page"]) for row in results],
                "latency_seconds": round(time.perf_counter() - started, 6),
                "base_trace": getattr(self.base, "last_trace", None),
            }
        }
        return results


class SelectiveMetricAwareRetriever:
    """Keep frozen B4b for ordinary queries and route only ambiguity-risk queries."""

    def __init__(self, frozen: Any, metric_aware: Any, triggers: list[str]) -> None:
        self.frozen = frozen
        self.metric_aware = metric_aware
        self.triggers = [normalize_search_text(item).casefold() for item in triggers]
        self.last_trace = None
        self.last_retrieval_calls: list[Any] = []
        self.last_router_call = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        normalized = normalize_search_text(query).casefold()
        matched = [trigger for trigger in self.triggers if trigger in normalized]
        selected = self.metric_aware if matched else self.frozen
        rows = selected.retrieve(query, top_k)
        self.last_trace = {
            "b4": {
                "stage": "selective_metric_aware",
                "selected_path": "A3.2+B4e" if matched else "A3.1+B4b_frozen",
                "matched_triggers": matched,
                "base_trace": getattr(selected, "last_trace", None),
            }
        }
        self.last_retrieval_calls = list(
            getattr(selected, "last_retrieval_calls", []) or []
        )
        self.last_router_call = getattr(selected, "last_router_call", None)
        return rows
