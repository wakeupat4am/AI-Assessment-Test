from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import DenseRetriever
from src.routing.llm_xrouter import LLMAgentXRouter
from src.routing.xrouter import ROUTES, RuleBasedXRouter


class RoutedRetriever:
    """Dispatch retrieval to A2/A3 representations using a B1 router."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        router: Any | None = None,
        retrievers: dict[str, Any] | None = None,
        route_indexes: dict[str, list[str]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if router is not None:
            self.router = router
        elif self.settings.xrouter_type.casefold() == "llm":
            self.router = LLMAgentXRouter(
                self.settings.xrouter_config_path,
                settings=self.settings,
                confidence_threshold=self.settings.xrouter_confidence_threshold,
                log_path=self.settings.xrouter_log_path,
            )
        elif self.settings.xrouter_type.casefold() in {"rule", "rule_based", "b1a"}:
            self.router = RuleBasedXRouter(
                self.settings.xrouter_config_path,
                confidence_threshold=self.settings.xrouter_confidence_threshold,
                log_path=self.settings.xrouter_log_path,
            )
        else:
            raise ValueError(
                f"Unsupported XROUTER_TYPE={self.settings.xrouter_type!r}; "
                "expected rule_based or llm"
            )
        selected_route_indexes = route_indexes or self.router.config["retrieval"][
            "route_indexes"
        ]
        if set(selected_route_indexes) != ROUTES:
            raise ValueError("route_indexes must define every X-Router route")
        self.route_indexes: dict[str, list[str]] = {
            route: [str(name) for name in names]
            for route, names in selected_route_indexes.items()
        }
        if retrievers is None:
            embedder = Embedder(
                self.settings.embedding_model, self.settings.embedding_device
            )
            index_dirs = {
                "narrative": self.settings.narrative_index_dir,
                "structured": self.settings.structured_index_dir,
                "multi_repr": self.settings.multi_repr_index_dir,
            }
            self.retrievers = {
                name: DenseRetriever(
                    self.settings, embedder=embedder, index_dir=Path(index_dir)
                )
                for name, index_dir in index_dirs.items()
            }
        else:
            self.retrievers = dict(retrievers)
        referenced = {
            name for names in self.route_indexes.values() for name in names
        }
        missing = sorted(referenced - set(self.retrievers))
        if missing:
            raise ValueError(f"Missing routed retrievers: {missing}")
        self.rrf_k = int(self.router.config["retrieval"].get("rrf_k", 60))
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.last_trace: dict[str, Any] | None = None
        self.last_router_call: Any | None = None

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        return self.retrieve_routed(query, top_k=top_k)

    def retrieve_routed(
        self,
        query: str,
        top_k: int | None = None,
        *,
        conversation_context: str | None = None,
        router_query: str | None = None,
    ) -> list[dict[str, Any]]:
        requested_k = top_k or self.settings.top_k
        route_started = time.perf_counter()
        decision = self.router.route(router_query or query, conversation_context)
        self.last_router_call = getattr(self.router, "last_call", None)
        router_seconds = time.perf_counter() - route_started
        selected_indexes = self.route_indexes[decision["route"]]

        retrieval_started = time.perf_counter()
        per_index: dict[str, list[dict[str, Any]]] = {}
        for name in selected_indexes:
            rows = self.retrievers[name].retrieve(query, requested_k)
            per_index[name] = [
                {**row, "retrieval_source": name, "source_rank": rank}
                for rank, row in enumerate(rows, 1)
            ]
        raw_count = sum(len(rows) for rows in per_index.values())
        if len(per_index) == 1:
            results = next(iter(per_index.values()))[:requested_k]
        else:
            results = self._reciprocal_rank_fusion(per_index, requested_k)
        retrieval_seconds = time.perf_counter() - retrieval_started
        self.last_trace = {
            **decision,
            "router_latency_seconds": round(router_seconds, 9),
            "retrieval_latency_seconds": round(retrieval_seconds, 9),
            "selected_indexes": selected_indexes,
            "raw_candidate_count": raw_count,
            "returned_candidate_count": len(results),
        }
        router_metadata = getattr(self.router, "last_metadata", None)
        if router_metadata:
            self.last_trace["router_metadata"] = dict(router_metadata)
        if self.last_router_call is not None:
            self.last_trace["router_llm"] = self.last_router_call.to_dict()
        return results

    def _reciprocal_rank_fusion(
        self, per_index: dict[str, list[dict[str, Any]]], top_k: int
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for source, rows in per_index.items():
            for rank, row in enumerate(rows, 1):
                # Identical A2 chunks can occur in both the full structured
                # index and its narrative projection. Fuse them once rather
                # than spending two top-k positions on duplicate evidence.
                key = str(row["chunk_id"])
                item = fused.setdefault(
                    key,
                    {
                        **row,
                        "retrieval_sources": [],
                        "source_ranks": {},
                        "fusion_score": 0.0,
                    },
                )
                item["retrieval_sources"].append(source)
                item["source_ranks"][source] = rank
                item["fusion_score"] += 1.0 / (self.rrf_k + rank)
                # Keep the dense similarity in `score`: Chatbot's frozen
                # refusal threshold is defined on this field.
                item["score"] = max(float(item["score"]), float(row["score"]))
        ordered = sorted(
            fused.values(),
            key=lambda row: (
                -float(row["fusion_score"]),
                -float(row["score"]),
                str(row["chunk_id"]),
            ),
        )
        return ordered[:top_k]
