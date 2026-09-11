#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from src.routing.llm_xrouter import LLMAgentXRouter
from src.routing.xrouter import RuleBasedXRouter


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compact(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = summary["metrics"]
    routed = [row["routing"] for row in rows if row.get("routing")]
    return {
        "questions": len(rows),
        "runtime_errors": summary["counts"]["runtime_errors"],
        "retrieval_hit_at_1": metrics["retrieval_hit_at_1"],
        "retrieval_hit_at_5": metrics["retrieval_hit_at_5"],
        "retrieval_hit_at_10": metrics["retrieval_hit_at_10"],
        "retrieval_recall_at_5": metrics["retrieval_recall_at_5"],
        "retrieval_recall_at_10": metrics["retrieval_recall_at_10"],
        "answer_accuracy_automatic": metrics["answer_accuracy"],
        "manual_answer_accuracy": metrics["manual_answer_accuracy"],
        "citation_precision": metrics["citation_precision"],
        "citation_recall": metrics["citation_recall"],
        "latency_seconds_p50": metrics["latency_seconds_p50"],
        "latency_seconds_p95": metrics["latency_seconds_p95"],
        "router_latency_seconds_p50": summary["routing_metrics"]["router_latency_seconds_p50"],
        "router_latency_seconds_p95": summary["routing_metrics"]["router_latency_seconds_p95"],
        "route_distribution": dict(
            sorted(Counter(str(item["route"]) for item in routed).items())
        ),
        "raw_candidates_mean": summary["routing_metrics"]["raw_candidates_mean"],
        "llm_calls_total": metrics["llm_calls_total"],
        "calls_by_purpose": metrics["calls_by_purpose"],
        "tokens": metrics["tokens"],
        "cost_usd_total": metrics["cost_usd_total"],
        "router_decision_source": dict(
            sorted(
                Counter(
                    str(item.get("router_metadata", {}).get("decision_source", "rules"))
                    for item in routed
                ).items()
            )
        ),
    }


def delta(candidate: Any, baseline: Any) -> float | None:
    if isinstance(candidate, (int, float)) and isinstance(baseline, (int, float)):
        return round(float(candidate) - float(baseline), 6)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B1b ablation: rule-based B1a versus LLM-agent B1b"
    )
    parser.add_argument("questions", type=Path)
    parser.add_argument("--run-label", default="public")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/evaluation/experiments/b1b"
    )
    parser.add_argument(
        "--b1a-config", type=Path, default=ROOT / "config/xrouter_b1a.json"
    )
    parser.add_argument(
        "--b1b-config", type=Path, default=ROOT / "config/xrouter_b1b.json"
    )
    parser.add_argument(
        "--retrieval-config",
        type=Path,
        help="Optional route-to-index override, e.g. controlled A2 mapping",
    )
    parser.add_argument("--manual-review", type=Path)
    args = parser.parse_args()

    settings = replace(get_settings(), xrouter_enabled=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b1a_config = load_json(args.b1a_config)
    route_indexes = (
        load_json(args.retrieval_config)["route_indexes"]
        if args.retrieval_config
        else b1a_config["retrieval"]["route_indexes"]
    )
    referenced_indexes = {
        name for names in route_indexes.values() for name in names
    }
    index_dirs = {
        "narrative": settings.narrative_index_dir,
        "structured": settings.structured_index_dir,
        "multi_repr": settings.multi_repr_index_dir,
    }
    unknown_indexes = sorted(referenced_indexes - set(index_dirs))
    if unknown_indexes:
        raise ValueError(f"Unknown retrieval indexes: {unknown_indexes}")
    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    retrievers = {
        name: DenseRetriever(settings, embedder=embedder, index_dir=index_dirs[name])
        for name in sorted(referenced_indexes)
    }
    experiment_slug = "b1b-controlled-a2" if args.retrieval_config else "b1b"

    arms = {
        "b1a": RuleBasedXRouter(
            args.b1a_config,
            confidence_threshold=settings.xrouter_confidence_threshold,
            log_path=args.output_dir / f"b1a-router-{args.run_label}.jsonl",
        ),
        "b1b": LLMAgentXRouter(
            args.b1b_config,
            settings=settings,
            confidence_threshold=settings.xrouter_confidence_threshold,
            log_path=args.output_dir / f"b1b-router-{args.run_label}.jsonl",
        ),
    }
    metrics: dict[str, dict[str, Any]] = {}
    row_sets: dict[str, list[dict[str, Any]]] = {}
    for name, router in arms.items():
        rows_path = args.output_dir / f"{name}-xrouter-{experiment_slug}-{args.run_label}.jsonl"
        routed = RoutedRetriever(
            settings,
            router=router,
            retrievers=retrievers,
            route_indexes=route_indexes,
        )
        bot = Chatbot(settings, retriever=routed)
        evaluate_questions(
            args.questions,
            rows_path,
            chatbot=bot,
            run_name=f"{name}-xrouter-{args.run_label}",
            manual_reviews_path=args.manual_review,
        )
        rows = load_jsonl(rows_path)
        row_sets[name] = rows
        metrics[name] = compact(load_json(rows_path.with_suffix(".summary.json")), rows)

    comparable = [
        "retrieval_hit_at_1", "retrieval_hit_at_5", "retrieval_hit_at_10",
        "retrieval_recall_at_5", "retrieval_recall_at_10",
        "answer_accuracy_automatic", "citation_precision", "citation_recall",
        "latency_seconds_p50", "latency_seconds_p95",
        "router_latency_seconds_p50", "router_latency_seconds_p95",
        "raw_candidates_mean", "llm_calls_total",
    ]
    payload = {
        "artifact_schema_version": 1,
        "experiment": "B1b",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "controlled_variables": {
            "indexes": (
                "frozen A2 structured and A2-derived narrative only"
                if args.retrieval_config
                else "frozen A2 narrative/structured and A3 multi_repr"
            ),
            "retrieval": "same dense retrieval and RRF",
            "answer_generation": "unchanged",
            "changed_variable": "route decision: B1a rules vs B1b Qwen LLM agent",
        },
        "b1a": metrics["b1a"],
        "b1b": metrics["b1b"],
        "delta_b1b_minus_b1a": {
            key: delta(metrics["b1b"][key], metrics["b1a"][key])
            for key in comparable
        },
        "per_question": {
            b1b_row["id"]: {
                "b1a_route": b1a_row["routing"]["route"],
                "b1b_route": b1b_row["routing"]["route"],
                "b1a_hit_at_5": b1a_row["scores"]["hit_at_5"],
                "b1b_hit_at_5": b1b_row["scores"]["hit_at_5"],
                "b1a_automatic_correct": b1a_row["scores"]["automatic_correct"],
                "b1b_automatic_correct": b1b_row["scores"]["automatic_correct"],
            }
            for b1a_row, b1b_row in zip(row_sets["b1a"], row_sets["b1b"])
        },
        "retrieval_config": str(args.retrieval_config) if args.retrieval_config else None,
        "route_indexes": route_indexes,
    }
    output = args.output_dir / f"{experiment_slug}-ablation-{args.run_label}.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ("b1a", "b1b", "delta_b1b_minus_b1a")}, ensure_ascii=False, indent=2))
    print(f"Ablation: {output}")


if __name__ == "__main__":
    main()
