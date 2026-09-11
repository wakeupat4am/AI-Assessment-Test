#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from src.retrieval.auto_search import AdaptiveRetriever
from src.retrieval.diversity import estimate_tokens
from src.retrieval.embedder import Embedder
from src.retrieval.parent_expansion import ParentExpandingRetriever
from src.retrieval.retriever import DenseRetriever
from src.retrieval.routed_retriever import RoutedRetriever
from src.routing.llm_xrouter import LLMAgentXRouter
from src.routing.xrouter import RuleBasedXRouter


ARMS = ("B0_one_shot", "B1a_rule", "B1b_llm", "B2_adaptive")
ROUTE_INDEXES = {
    "narrative": ["narrative"],
    "structured": ["structured"],
    "multi_repr": ["multi_repr"],
    "multi_hop": ["structured", "narrative"],
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def make_dense(settings, embedder: Embedder, path: Path, parent_chars: int):
    return ParentExpandingRetriever(
        DenseRetriever(settings, embedder=embedder, index_dir=path),
        max_parent_characters=parent_chars,
    )


def make_routed(
    settings,
    embedder: Embedder,
    index_root: Path,
    router: Any,
    parent_chars: int,
) -> RoutedRetriever:
    return RoutedRetriever(
        settings,
        router=router,
        retrievers={
            "narrative": make_dense(
                settings, embedder, index_root / "narrative", parent_chars
            ),
            "structured": make_dense(
                settings, embedder, index_root / "structured", parent_chars
            ),
            "multi_repr": make_dense(
                settings, embedder, index_root / "all", parent_chars
            ),
        },
        route_indexes=ROUTE_INDEXES,
    )


def compact(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = summary["metrics"]
    retrieval_seconds = [
        float(row.get("latency_breakdown", {}).get("retrieval_seconds", 0))
        for row in rows
        if not row.get("error")
    ]
    duplicate_slots: list[float] = []
    evidence_tokens: list[float] = []
    parent_expanded: list[float] = []
    routes: Counter[str] = Counter()
    rounds: list[float] = []
    for row in rows:
        chunks = row.get("retrieved_chunks", [])
        pages = [int(chunk["printed_page"]) for chunk in chunks]
        duplicate_slots.append(float(len(pages) - len(set(pages))))
        evidence_tokens.append(
            float(sum(estimate_tokens(str(chunk.get("text", ""))) for chunk in chunks[:5]))
        )
        parent_expanded.append(
            float(sum(bool(chunk.get("parent_expanded")) for chunk in chunks[:5]))
        )
        routing = row.get("routing") or {}
        if routing.get("route"):
            routes[str(routing["route"])] += 1
        b2 = routing.get("b2") or {}
        rounds.append(float(b2.get("search_rounds", 1)))
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
        "latency_p50_seconds": metrics["latency_seconds_p50"],
        "latency_p95_seconds": metrics["latency_seconds_p95"],
        "retrieval_latency_mean_seconds": mean(retrieval_seconds),
        "llm_calls_total": metrics["llm_calls_total"],
        "llm_tokens": metrics["tokens"],
        "cost_usd_total": metrics["cost_usd_total"],
        "cost_projection_same_tokens_usd": metrics["cost_projection_same_tokens_usd"],
        "average_search_rounds": mean(rounds),
        "second_search_trigger_rate": round(
            sum(value == 2 for value in rounds) / len(rounds), 6
        ) if rounds else 0.0,
        "average_returned_candidates": mean(
            [float(len(row.get("retrieved_chunks", []))) for row in rows]
        ),
        "average_evidence_tokens": mean(evidence_tokens),
        "average_parent_expanded_chunks": mean(parent_expanded),
        "average_duplicate_page_slots": mean(duplicate_slots),
        "route_distribution": dict(sorted(routes.items())),
        "failure_taxonomy": summary["failure_taxonomy"],
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# A3.1 controller benchmark — {payload['run_label']}",
        "",
        "| Controller | Auto accuracy | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Avg rounds | Trigger | Evidence tokens | Parent-expanded | Duplicate slots | Retrieval ms | E2E p50/p95 s | LLM calls/tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARMS:
        item = payload["controllers"][name]
        lines.append(
            "| " + " | ".join(
                [
                    name,
                    f"{item['answer_accuracy_automatic']:.4f}",
                    f"{item['retrieval_hit_at_1']:.4f}",
                    f"{item['retrieval_hit_at_5']:.4f}",
                    f"{item['retrieval_hit_at_10']:.4f}",
                    f"{item['citation_precision']:.4f}/{item['citation_recall']:.4f}",
                    f"{item['refusal_accuracy']:.4f}",
                    f"{item['average_search_rounds']:.3f}",
                    f"{item['second_search_trigger_rate']:.3f}",
                    f"{item['average_evidence_tokens']:.1f}",
                    f"{item['average_parent_expanded_chunks']:.2f}",
                    f"{item['average_duplicate_page_slots']:.2f}",
                    f"{item['retrieval_latency_mean_seconds'] * 1000:.1f}",
                    f"{item['latency_p50_seconds']:.3f}/{item['latency_p95_seconds']:.3f}",
                    f"{item['llm_calls_total']}/{item['llm_tokens']['total_tokens']}",
                ]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "> A3.1 and answer generation are fixed across arms; only the retrieval controller changes.",
            "> Automatic answer accuracy remains diagnostic until manual review is added.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled A3.1 benchmark over B0, B1a, B1b and B2"
    )
    parser.add_argument("questions", type=Path)
    parser.add_argument("--run-label", default="public")
    parser.add_argument(
        "--index-root",
        type=Path,
        default=ROOT / "data/index/a31_semantic_multirepr",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/evaluation/experiments/a31",
    )
    parser.add_argument(
        "--b1a-config", type=Path, default=ROOT / "config/xrouter_b1a.json"
    )
    parser.add_argument(
        "--b1b-config", type=Path, default=ROOT / "config/xrouter_b1b.json"
    )
    parser.add_argument("--parent-characters", type=int, default=1800)
    parser.add_argument("--only", nargs="+", choices=ARMS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_settings = replace(
        get_settings(),
        index_dir=args.index_root / "all",
        xrouter_enabled=False,
        b2_enabled=False,
        enable_auto_search=False,
        enable_page_dedup=False,
        enable_diversity_selection=False,
    )
    embedder = Embedder(base_settings.embedding_model, base_settings.embedding_device)
    selected = set(args.only or ARMS)
    metrics: dict[str, dict[str, Any]] = {}

    for name in ARMS:
        rows_path = args.output_dir / f"a31-{name}-{args.run_label}.jsonl"
        if name in selected:
            if name == "B0_one_shot":
                retriever = make_dense(
                    base_settings,
                    embedder,
                    args.index_root / "all",
                    args.parent_characters,
                )
            elif name == "B1a_rule":
                retriever = make_routed(
                    base_settings,
                    embedder,
                    args.index_root,
                    RuleBasedXRouter(args.b1a_config),
                    args.parent_characters,
                )
            elif name == "B1b_llm":
                retriever = make_routed(
                    base_settings,
                    embedder,
                    args.index_root,
                    LLMAgentXRouter(args.b1b_config, settings=base_settings),
                    args.parent_characters,
                )
            else:
                routed = make_routed(
                    base_settings,
                    embedder,
                    args.index_root,
                    RuleBasedXRouter(args.b1a_config),
                    args.parent_characters,
                )
                retriever = AdaptiveRetriever(
                    routed,
                    enable_auto_search=True,
                    enable_page_dedup=True,
                    enable_diversity_selection=True,
                    initial_top_k=base_settings.b2_initial_top_k,
                    second_round_top_k=base_settings.b2_second_round_top_k,
                    max_search_rounds=base_settings.b2_max_search_rounds,
                    max_per_page=base_settings.b2_max_per_page,
                    similarity_threshold=base_settings.b2_page_similarity_threshold,
                    final_k=base_settings.top_k,
                    max_evidence_tokens=base_settings.b2_max_evidence_tokens,
                    lambda_relevance=base_settings.b2_diversity_lambda,
                    log_path=args.output_dir / f"a31-B2-search-{args.run_label}.jsonl",
                )
            evaluate_questions(
                args.questions,
                rows_path,
                chatbot=Chatbot(base_settings, retriever=retriever),
                run_name=f"a31-{name}-{args.run_label}",
            )
        rows = load_jsonl(rows_path)
        summary = load_json(rows_path.with_suffix(".summary.json"))
        metrics[name] = compact(summary, rows)

    payload = {
        "artifact_schema_version": 1,
        "experiment": "A3.1 semantic multi-representation x retrieval controller",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "questions": str(args.questions),
        "index_root": str(args.index_root),
        "controlled_variables": {
            "document_representation": "A3.1 for all four controllers",
            "embedding_model": base_settings.embedding_model,
            "answer_generation": "unchanged Qwen endpoint and prompt",
            "top_k_evidence": base_settings.top_k,
            "changed_variable": "B0 one-shot vs B1a rule routing vs B1b LLM routing vs B2 adaptive retrieval",
        },
        "route_indexes": ROUTE_INDEXES,
        "parent_character_limit": args.parent_characters,
        "controllers": metrics,
    }
    json_path = args.output_dir / f"a31-controller-benchmark-{args.run_label}.json"
    md_path = args.output_dir / f"a31-controller-benchmark-{args.run_label}.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
