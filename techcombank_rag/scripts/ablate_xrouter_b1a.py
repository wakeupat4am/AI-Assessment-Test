#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chat.chatbot import Chatbot
from src.config import get_settings
from src.evaluation.evaluator import evaluate_questions
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import DenseRetriever
from src.retrieval.routed_retriever import RoutedRetriever
from src.routing.xrouter import RuleBasedXRouter


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return round(ordered[low], 9)
    weight = position - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 9)


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 9) if values else None


def compact_metrics(
    summary: dict[str, Any], rows: list[dict[str, Any]], *, routed: bool
) -> dict[str, Any]:
    metrics = summary["metrics"]
    retrieval_latencies = [
        float(row.get("latency_breakdown", {}).get("retrieval_seconds", 0))
        for row in rows
        if not row.get("error")
    ]
    returned_counts = [float(len(row.get("retrieved_chunks", []))) for row in rows]
    raw_counts = [
        float(row["routing"]["raw_candidate_count"])
        if routed and row.get("routing")
        else float(len(row.get("retrieved_chunks", [])))
        for row in rows
    ]
    return {
        "questions": len(rows),
        "runtime_errors": summary["counts"]["runtime_errors"],
        "retrieval_hit_at_1": metrics["retrieval_hit_at_1"],
        "retrieval_hit_at_5": metrics["retrieval_hit_at_5"],
        "retrieval_hit_at_10": metrics["retrieval_hit_at_10"],
        "retrieval_recall_at_1": metrics["retrieval_recall_at_1"],
        "retrieval_recall_at_5": metrics["retrieval_recall_at_5"],
        "retrieval_recall_at_10": metrics["retrieval_recall_at_10"],
        "answer_accuracy": metrics["answer_accuracy"],
        "manual_answer_accuracy": metrics["manual_answer_accuracy"],
        "total_latency_seconds_p50": metrics["latency_seconds_p50"],
        "total_latency_seconds_p95": metrics["latency_seconds_p95"],
        "retrieval_latency_seconds_mean": mean(retrieval_latencies),
        "retrieval_latency_seconds_p50": percentile(retrieval_latencies, 0.50),
        "retrieval_latency_seconds_p95": percentile(retrieval_latencies, 0.95),
        "raw_candidates_mean": mean(raw_counts),
        "returned_candidates_mean": mean(returned_counts),
        "total_input_tokens": metrics["tokens"]["total_input_tokens"],
        "output_tokens": metrics["tokens"]["output_tokens"],
        "llm_calls_total": metrics["llm_calls_total"],
        "cost_projection_same_tokens_usd": metrics[
            "cost_projection_same_tokens_usd"
        ],
    }


def delta(candidate: Any, baseline: Any) -> float | None:
    if candidate is None or baseline is None:
        return None
    if not isinstance(candidate, (int, float)) or not isinstance(
        baseline, (int, float)
    ):
        return None
    return round(float(candidate) - float(baseline), 9)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B1a ablation: always-A2 retrieval versus rule-based X-Router"
    )
    parser.add_argument("questions", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/evaluation/experiments/b1a",
    )
    parser.add_argument(
        "--router-config", type=Path, default=ROOT / "config/xrouter_b1a.json"
    )
    parser.add_argument("--run-label", default="public")
    parser.add_argument(
        "--manual-review",
        type=Path,
        help="Optional shared blind review file applied to both arms",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Regenerate comparison from existing row/summary files without retrieval or LLM calls",
    )
    args = parser.parse_args()

    settings = get_settings()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows_path = args.output_dir / f"b1a-no-router-a2-{args.run_label}.jsonl"
    routed_rows_path = args.output_dir / f"b1a-xrouter-{args.run_label}.jsonl"
    router_log_path = args.output_dir / f"b1a-router-{args.run_label}.jsonl"
    if not args.compare_only:
        router_log_path.write_text("", encoding="utf-8")
        arm_settings = replace(
            settings,
            index_dir=settings.structured_index_dir,
            xrouter_enabled=False,
        )

        # Share one encoder across all paths. This avoids counting model-loading
        # differences as retrieval latency and guarantees the same query vector.
        embedder = Embedder(settings.embedding_model, settings.embedding_device)
        structured = DenseRetriever(
            arm_settings,
            embedder=embedder,
            index_dir=settings.structured_index_dir,
        )
        baseline_bot = Chatbot(arm_settings, retriever=structured)
        evaluate_questions(
            args.questions,
            baseline_rows_path,
            chatbot=baseline_bot,
            run_name=f"b1a-no-router-a2-{args.run_label}",
            manual_reviews_path=args.manual_review,
        )

        routed = RoutedRetriever(
            arm_settings,
            router=RuleBasedXRouter(
                args.router_config,
                confidence_threshold=settings.xrouter_confidence_threshold,
            log_path=router_log_path,
        ),
        retrievers={
            "narrative": DenseRetriever(
                arm_settings,
                embedder=embedder,
                index_dir=settings.narrative_index_dir,
            ),
            "structured": structured,
            "multi_repr": DenseRetriever(
                arm_settings,
                embedder=embedder,
                index_dir=settings.multi_repr_index_dir,
            ),
            },
        )
        routed_bot = Chatbot(arm_settings, retriever=routed)
        evaluate_questions(
            args.questions,
            routed_rows_path,
            chatbot=routed_bot,
            run_name=f"b1a-xrouter-{args.run_label}",
            manual_reviews_path=args.manual_review,
        )

    baseline_summary = load_json(baseline_rows_path.with_suffix(".summary.json"))
    routed_summary = load_json(routed_rows_path.with_suffix(".summary.json"))
    baseline_rows = load_jsonl(baseline_rows_path)
    routed_rows = load_jsonl(routed_rows_path)
    baseline_metrics = compact_metrics(baseline_summary, baseline_rows, routed=False)
    routed_metrics = compact_metrics(routed_summary, routed_rows, routed=True)
    comparable = [
        "retrieval_hit_at_1",
        "retrieval_hit_at_5",
        "retrieval_hit_at_10",
        "retrieval_recall_at_1",
        "retrieval_recall_at_5",
        "retrieval_recall_at_10",
        "answer_accuracy",
        "retrieval_latency_seconds_mean",
        "retrieval_latency_seconds_p50",
        "retrieval_latency_seconds_p95",
        "raw_candidates_mean",
        "returned_candidates_mean",
        "total_input_tokens",
        "output_tokens",
        "llm_calls_total",
    ]
    route_distribution = Counter(
        str(row["routing"]["route"])
        for row in routed_rows
        if row.get("routing")
    )
    comparison = {
        "artifact_schema_version": 1,
        "experiment": "B1a",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "A deterministic query router improves retrieval over always using A2 without changing answer generation.",
        "questions": str(args.questions),
        "questions_sha256": sha256(args.questions),
        "router_config": str(args.router_config),
        "router_config_sha256": sha256(args.router_config),
        "index_mapping": {
            "narrative": str(settings.narrative_index_dir),
            "structured": str(settings.structured_index_dir),
            "multi_repr": str(settings.multi_repr_index_dir),
            "multi_hop": [
                str(settings.structured_index_dir),
                str(settings.narrative_index_dir),
            ],
        },
        "answer_generation_changed": False,
        "baseline": baseline_metrics,
        "xrouter": {
            **routed_metrics,
            "route_distribution": dict(sorted(route_distribution.items())),
            "router_latency_seconds_p50": routed_summary["routing_metrics"][
                "router_latency_seconds_p50"
            ],
            "router_latency_seconds_p95": routed_summary["routing_metrics"][
                "router_latency_seconds_p95"
            ],
        },
        "delta_xrouter_minus_baseline": {
            name: delta(routed_metrics[name], baseline_metrics[name])
            for name in comparable
        },
        "per_question_routes": {
            row["id"]: {
                "route": row["routing"]["route"],
                "confidence": row["routing"]["confidence"],
                "selected_indexes": row["routing"]["selected_indexes"],
                "raw_candidate_count": row["routing"]["raw_candidate_count"],
                "retrieved_pages": row["retrieved_pages"],
                "hit_at_5": row["scores"]["hit_at_5"],
                "hit_at_10": row["scores"]["hit_at_10"],
                "automatic_correct": row["scores"]["automatic_correct"],
                "effective_correct": row["scores"]["effective_correct"],
            }
            for row in routed_rows
        },
    }
    comparison_path = args.output_dir / f"b1a-ablation-{args.run_label}.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"Ablation: {comparison_path}")


if __name__ == "__main__":
    main()
