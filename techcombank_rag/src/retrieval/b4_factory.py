from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import Settings
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.finance_reranker import FinanceAwareReranker
from src.retrieval.hybrid import HybridRRFRetriever


MODES = {"bm25", "hybrid", "finance_rerank", "cross_encoder", "metric_aware"}


def load_b4_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("tokenizer", "bm25", "fusion", "finance_reranker"):
        if key not in config:
            raise ValueError(f"B4 configuration is missing {key}")
    return config


def create_b4_retriever(
    dense_retriever: Any,
    settings: Settings,
    *,
    mode: str | None = None,
    config: dict[str, Any] | None = None,
    cross_encoder_model: Any | None = None,
) -> Any:
    selected_mode = mode or settings.b4_retrieval_mode
    if selected_mode not in MODES:
        raise ValueError(f"B4_RETRIEVAL_MODE must be one of {sorted(MODES)}")
    resolved_config = config or load_b4_config(settings.b4_config_path)
    bm25_config = resolved_config["bm25"]
    fusion_config = resolved_config["fusion"]
    bm25 = BM25Retriever(
        settings.index_dir / "chunks.jsonl",
        settings.b4_bm25_artifact_path,
        candidate_k=int(bm25_config["candidate_k"]),
    )
    if selected_mode == "bm25":
        return bm25
    hybrid = HybridRRFRetriever(
        dense_retriever,
        bm25,
        candidate_k=int(fusion_config["candidate_k"]),
        rrf_k=int(fusion_config["rrf_k"]),
        dense_weight=float(fusion_config["dense_weight"]),
        bm25_weight=float(fusion_config["bm25_weight"]),
    )
    if selected_mode == "hybrid":
        return hybrid
    if selected_mode == "metric_aware":
        from src.retrieval.metric_aware import (
            MetricAwareRetriever,
            SelectiveMetricAwareRetriever,
        )
        from src.retrieval.retriever import DenseRetriever

        if "metric_aware" not in resolved_config:
            raise ValueError("B4e configuration is missing metric_aware")
        metric_aware = MetricAwareRetriever(
            hybrid,
            resolved_config["metric_aware"],
            candidate_k=int(fusion_config["candidate_k"]),
        )
        frozen_config = load_b4_config(settings.b4_frozen_config_path)
        frozen_bm25_config = frozen_config["bm25"]
        frozen_fusion_config = frozen_config["fusion"]
        frozen_dense = DenseRetriever(
            settings,
            embedder=dense_retriever.embedder,
            index_dir=settings.b4_frozen_index_dir,
        )
        frozen_bm25 = BM25Retriever(
            settings.b4_frozen_index_dir / "chunks.jsonl",
            settings.b4_frozen_index_dir / "bm25.json.gz",
            candidate_k=int(frozen_bm25_config["candidate_k"]),
        )
        frozen = HybridRRFRetriever(
            frozen_dense,
            frozen_bm25,
            candidate_k=int(frozen_fusion_config["candidate_k"]),
            rrf_k=int(frozen_fusion_config["rrf_k"]),
            dense_weight=float(frozen_fusion_config["dense_weight"]),
            bm25_weight=float(frozen_fusion_config["bm25_weight"]),
        )
        return SelectiveMetricAwareRetriever(
            frozen,
            metric_aware,
            list(resolved_config["metric_aware"].get("selective_triggers", [])),
        )
    if selected_mode == "finance_rerank":
        return FinanceAwareReranker(
            hybrid,
            resolved_config["finance_reranker"],
            candidate_k=int(fusion_config["candidate_k"]),
        )
    from src.retrieval.cross_encoder_reranker import CrossEncoderReranker

    return CrossEncoderReranker(
        hybrid,
        model_name=settings.b4_cross_encoder_model,
        device=settings.b4_cross_encoder_device,
        batch_size=settings.b4_cross_encoder_batch_size,
        max_length=settings.b4_cross_encoder_max_length,
        candidate_k=int(fusion_config["candidate_k"]),
        threads=settings.b4_cross_encoder_threads,
        model=cross_encoder_model,
    )
