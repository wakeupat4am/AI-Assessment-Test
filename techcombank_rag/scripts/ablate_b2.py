#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chat.chatbot import Chatbot
from src.config import get_settings
from src.evaluation.evaluator import evaluate_questions
from src.retrieval.auto_search import AdaptiveRetriever
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import DenseRetriever
from src.retrieval.routed_retriever import RoutedRetriever
from src.routing.xrouter import RuleBasedXRouter


VARIANTS = {
    "A_baseline_b1a": (False, False, False),
    "B_page_dedup": (False, True, False),
    "C_dedup_diversity": (False, True, True),
    "D_auto_search": (True, False, False),
    "E_full_b2": (True, True, True),
}


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_base_retriever(settings, embedder: Embedder, router_config: Path):
    structured = DenseRetriever(
        settings, embedder=embedder, index_dir=settings.structured_index_dir
    )
    retrievers = {
        "narrative": DenseRetriever(
            settings, embedder=embedder, index_dir=settings.narrative_index_dir
        ),
        "structured": structured,
        "multi_repr": DenseRetriever(
            settings, embedder=embedder, index_dir=settings.multi_repr_index_dir
        ),
    }
    return RoutedRetriever(
        settings,
        router=RuleBasedXRouter(router_config),
        retrievers=retrievers,
    )


def compact(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = summary["metrics"]
    traces = [
        (row.get("routing") or {})["b2"]
        for row in rows
        if (row.get("routing") or {}).get("b2")
    ]
    rounds = [int(trace["search_rounds"]) for trace in traces]
    retrieved_counts = [len(row.get("retrieved_chunks", [])) for row in rows]
    duplicate_slots = []
    for row in rows:
        pages = [int(item["printed_page"]) for item in row.get("retrieved_chunks", [])]
        duplicate_slots.append(len(pages) - len(set(pages)))
    return {
        "questions": len(rows),
        "runtime_errors": summary["counts"]["runtime_errors"],
        "answer_accuracy_automatic": metrics["answer_accuracy"],
        "manual_answer_accuracy": metrics["manual_answer_accuracy"],
        "retrieval_hit_at_1": metrics["retrieval_hit_at_1"],
        "retrieval_hit_at_5": metrics["retrieval_hit_at_5"],
        "retrieval_hit_at_10": metrics["retrieval_hit_at_10"],
        "retrieval_recall_at_5": metrics["retrieval_recall_at_5"],
        "retrieval_recall_at_10": metrics["retrieval_recall_at_10"],
        "citation_precision": metrics["citation_precision"],
        "citation_recall": metrics["citation_recall"],
        "citation_validator_pass_rate": metrics["citation_validator_pass_rate"],
        "refusal_accuracy": metrics["refusal_accuracy"],
        "end_to_end_latency_p50_seconds": metrics["latency_seconds_p50"],
        "end_to_end_latency_p95_seconds": metrics["latency_seconds_p95"],
        "llm_calls_total": metrics["llm_calls_total"],
        "llm_tokens": metrics["tokens"],
        "cost_usd_total": metrics["cost_usd_total"],
        "average_search_rounds": mean([float(value) for value in rounds]),
        "second_search_trigger_rate": round(
            sum(value == 2 for value in rounds) / len(rounds), 6
        ) if rounds else None,
        "average_initial_candidates": mean(
            [float(trace["initial_candidates"]) for trace in traces]
        ),
        "average_after_page_dedup": mean(
            [float(trace["after_page_dedup"]) for trace in traces]
        ),
        "average_after_diversity": mean(
            [float(trace["after_diversity"]) for trace in traces]
        ),
        "average_retrieved_candidates": mean(
            [float(value) for value in retrieved_counts]
        ),
        "average_final_evidence_chunks": mean(
            [float(trace["final_evidence_count"]) for trace in traces]
        ),
        "average_final_evidence_tokens": mean(
            [float(trace["final_evidence_tokens"]) for trace in traces]
        ),
        "average_duplicate_page_slots": mean(
            [float(value) for value in duplicate_slots]
        ),
        "average_retrieval_latency_ms": mean(
            [float(trace["total_retrieval_latency_ms"]) for trace in traces]
        ),
        "sufficient_rate": round(
            sum(bool(trace["sufficient"]) for trace in traces) / len(traces), 6
        ) if traces else None,
        "stop_reasons": {
            reason: sum(trace["stop_reason"] == reason for trace in traces)
            for reason in sorted({str(trace["stop_reason"]) for trace in traces})
        },
        "failure_taxonomy": summary["failure_taxonomy"],
    }


def markdown_table(payload: dict[str, Any]) -> str:
    lines = [
        f"# B2 ablation — {payload['run_label']}",
        "",
        "| Method | Auto | Dedup | Diversity | Auto accuracy | Hit@5 | Hit@10 | Citation P/R | Avg rounds | Trigger | Evidence tokens | Duplicate slots | Retrieval ms | E2E p50 s | LLM tokens |",
        "|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in payload["variants"].items():
        flags = payload["flags"][name]
        citation = f"{result['citation_precision']:.4f}/{result['citation_recall']:.4f}"
        lines.append(
            "| " + " | ".join(
                [
                    name,
                    "Y" if flags["auto_search"] else "N",
                    "Y" if flags["page_dedup"] else "N",
                    "Y" if flags["diversity"] else "N",
                    f"{result['answer_accuracy_automatic']:.4f}",
                    f"{result['retrieval_hit_at_5']:.4f}",
                    f"{result['retrieval_hit_at_10']:.4f}",
                    citation,
                    f"{result['average_search_rounds']:.3f}",
                    f"{result['second_search_trigger_rate']:.3f}",
                    f"{result['average_final_evidence_tokens']:.1f}",
                    f"{result['average_duplicate_page_slots']:.2f}",
                    f"{result['average_retrieval_latency_ms']:.1f}",
                    f"{result['end_to_end_latency_p50_seconds']:.3f}",
                    str(result["llm_tokens"]["total_tokens"]),
                ]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "> Automatic answer accuracy is diagnostic; no new manual review is applied.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-arm B2 adaptive retrieval ablation")
    parser.add_argument("questions", type=Path)
    parser.add_argument("--run-label", default="public")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/evaluation/experiments/b2"
    )
    parser.add_argument(
        "--router-config", type=Path, default=ROOT / "config/xrouter_b1a.json"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=tuple(VARIANTS),
        help="Rerun selected arms and reuse existing artifacts for the others.",
    )
    args = parser.parse_args()

    settings = replace(
        get_settings(),
        xrouter_enabled=True,
        xrouter_type="rule_based",
        b2_enabled=False,
        enable_auto_search=False,
        enable_page_dedup=False,
        enable_diversity_selection=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    results: dict[str, dict[str, Any]] = {}
    flags: dict[str, dict[str, bool]] = {}

    for name, (auto, dedup, diversity) in VARIANTS.items():
        rows_path = args.output_dir / f"{name}-{args.run_label}.jsonl"
        if args.only and name not in args.only:
            if not rows_path.exists() or not rows_path.with_suffix(".summary.json").exists():
                raise FileNotFoundError(
                    f"Cannot reuse missing artifact for {name}: {rows_path}"
                )
            rows = load_jsonl(rows_path)
            summary = load_json(rows_path.with_suffix(".summary.json"))
            results[name] = compact(summary, rows)
            flags[name] = {
                "auto_search": auto,
                "page_dedup": dedup,
                "diversity": diversity,
            }
            continue
        base = make_base_retriever(settings, embedder, args.router_config)
        log_path = args.output_dir / f"{name}-{args.run_label}-search.jsonl"
        log_path.write_text("", encoding="utf-8")
        adaptive = AdaptiveRetriever(
            base,
            enable_auto_search=auto,
            enable_page_dedup=dedup,
            enable_diversity_selection=diversity,
            initial_top_k=settings.b2_initial_top_k,
            second_round_top_k=settings.b2_second_round_top_k,
            max_search_rounds=settings.b2_max_search_rounds,
            max_per_page=settings.b2_max_per_page,
            similarity_threshold=settings.b2_page_similarity_threshold,
            final_k=settings.top_k,
            max_evidence_tokens=settings.b2_max_evidence_tokens,
            lambda_relevance=settings.b2_diversity_lambda,
            log_path=log_path,
        )
        evaluate_questions(
            args.questions,
            rows_path,
            chatbot=Chatbot(settings, retriever=adaptive),
            run_name=f"b2-{name}-{args.run_label}",
        )
        rows = load_jsonl(rows_path)
        summary = load_json(rows_path.with_suffix(".summary.json"))
        results[name] = compact(summary, rows)
        flags[name] = {
            "auto_search": auto,
            "page_dedup": dedup,
            "diversity": diversity,
        }

    payload = {
        "artifact_schema_version": 1,
        "experiment": "B2 AutoSearch-inspired adaptive retrieval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "questions": str(args.questions),
        "parent_system": "B1a rule router over A2 narrative/structured and A3 multi_repr",
        "answer_generation_changed": False,
        "reranker": None,
        "configuration": {
            "initial_top_k": settings.b2_initial_top_k,
            "second_round_top_k": settings.b2_second_round_top_k,
            "max_search_rounds": settings.b2_max_search_rounds,
            "max_per_page": settings.b2_max_per_page,
            "page_similarity_threshold": settings.b2_page_similarity_threshold,
            "final_k": settings.top_k,
            "max_evidence_tokens": settings.b2_max_evidence_tokens,
            "lambda_relevance": settings.b2_diversity_lambda,
        },
        "flags": flags,
        "variants": results,
    }
    output_json = args.output_dir / f"b2-ablation-{args.run_label}.json"
    output_md = args.output_dir / f"b2-ablation-{args.run_label}.md"
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_md.write_text(markdown_table(payload), encoding="utf-8")
    print(markdown_table(payload))
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()
