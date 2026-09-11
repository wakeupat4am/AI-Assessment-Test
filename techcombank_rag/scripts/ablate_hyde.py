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
from src.retrieval.diversity import estimate_tokens
from src.retrieval.embedder import Embedder
from src.retrieval.hyde import HyDERetriever
from src.retrieval.parent_expansion import ParentExpandingRetriever
from src.retrieval.retriever import DenseRetriever


ARMS = (
    "B0_one_shot",
    "B3a_hyde_only",
    "B3b_fusion",
    "B3c_safe_fusion",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def make_retriever(
    name: str,
    settings: Any,
    embedder: Embedder,
    index_dir: Path,
    parent_chars: int,
    log_path: Path,
) -> Any:
    dense = DenseRetriever(settings, embedder=embedder, index_dir=index_dir)
    if name == "B0_one_shot":
        child_retriever: Any = dense
    else:
        mode = "hyde_only" if name == "B3a_hyde_only" else "fusion"
        prompt_variant = (
            "finance_safe" if name == "B3c_safe_fusion" else "paper_faithful"
        )
        child_retriever = HyDERetriever(
            dense,
            settings=settings,
            mode=mode,
            prompt_variant=prompt_variant,
            candidate_k=settings.hyde_candidate_k,
            rrf_k=settings.hyde_rrf_k,
            original_weight=settings.hyde_original_weight,
            hypothetical_weight=settings.hyde_hypothetical_weight,
            log_path=log_path,
        )
    return ParentExpandingRetriever(
        child_retriever, max_parent_characters=parent_chars
    )


def compact(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = summary["metrics"]
    traces = [
        (row.get("routing") or {}).get("hyde")
        for row in rows
        if (row.get("routing") or {}).get("hyde")
    ]
    chunks_by_row = [row.get("retrieved_chunks", []) for row in rows]
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
        "retrieval_latency_mean_seconds": mean(
            [
                float(row.get("latency_breakdown", {}).get("retrieval_seconds", 0))
                for row in rows
                if not row.get("error")
            ]
        ),
        "llm_calls_total": metrics["llm_calls_total"],
        "calls_by_purpose": metrics["calls_by_purpose"],
        "llm_tokens": metrics["tokens"],
        "cost_usd_total": metrics["cost_usd_total"],
        "cost_projection_same_tokens_usd": metrics[
            "cost_projection_same_tokens_usd"
        ],
        "average_returned_candidates": mean(
            [float(len(chunks)) for chunks in chunks_by_row]
        ),
        "average_evidence_tokens": mean(
            [
                float(
                    sum(
                        estimate_tokens(str(chunk.get("text", "")))
                        for chunk in chunks[:5]
                    )
                )
                for chunks in chunks_by_row
            ]
        ),
        "average_duplicate_page_slots": mean(
            [
                float(
                    len(chunks)
                    - len({int(chunk["printed_page"]) for chunk in chunks})
                )
                for chunks in chunks_by_row
            ]
        ),
        "hyde_trigger_rate": round(len(traces) / len(rows), 6) if rows else 0.0,
        "hyde_fallback_rate": (
            round(sum(bool(trace["fallback"]) for trace in traces) / len(traces), 6)
            if traces
            else 0.0
        ),
        "hyde_cache_hit_rate": (
            round(sum(bool(trace["cache_hit"]) for trace in traces) / len(traces), 6)
            if traces
            else 0.0
        ),
        "average_hypothetical_characters": mean(
            [float(len(str(trace["hypothetical_document"]))) for trace in traces]
        ),
        "average_original_candidates": mean(
            [float(trace["original_candidates"]) for trace in traces]
        ),
        "average_hyde_candidates": mean(
            [float(trace["hyde_candidates"]) for trace in traces]
        ),
        "failure_taxonomy": summary["failure_taxonomy"],
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# A3.1 HyDE ablation — {payload['run_label']}",
        "",
        "| Arm | Auto accuracy | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Retrieval ms | E2E p50/p95 s | Calls | Tokens | HyDE fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARMS:
        item = payload["arms"][name]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{item['answer_accuracy_automatic']:.4f}",
                    f"{item['retrieval_hit_at_1']:.4f}",
                    f"{item['retrieval_hit_at_5']:.4f}",
                    f"{item['retrieval_hit_at_10']:.4f}",
                    f"{item['citation_precision']:.4f}/{item['citation_recall']:.4f}",
                    f"{item['refusal_accuracy']:.4f}",
                    f"{item['retrieval_latency_mean_seconds'] * 1000:.1f}",
                    f"{item['latency_p50_seconds']:.3f}/{item['latency_p95_seconds']:.3f}",
                    str(item["llm_calls_total"]),
                    str(item["llm_tokens"]["total_tokens"]),
                    f"{item['hyde_fallback_rate']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "> A3.1, E5, answer prompt, answer model, and top-k are fixed. Only query representation/retrieval changes.",
            "> Hypothetical documents are search pivots only and never enter answer evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate HyDE over fixed A3.1")
    parser.add_argument("questions", type=Path)
    parser.add_argument("--run-label", default="public")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT / "data/index/a31_semantic_multirepr/all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/evaluation/experiments/hyde",
    )
    parser.add_argument("--parent-characters", type=int, default=1800)
    parser.add_argument("--only", nargs="+", choices=ARMS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = replace(
        get_settings(),
        index_dir=args.index_dir,
        xrouter_enabled=False,
        b2_enabled=False,
        enable_auto_search=False,
        enable_page_dedup=False,
        enable_diversity_selection=False,
        hyde_enabled=False,
    )
    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    selected = set(args.only or ARMS)
    metrics: dict[str, dict[str, Any]] = {}
    for name in ARMS:
        rows_path = args.output_dir / f"{name}-{args.run_label}.jsonl"
        if name in selected:
            trace_path = args.output_dir / f"{name}-{args.run_label}-trace.jsonl"
            trace_path.unlink(missing_ok=True)
            retriever = make_retriever(
                name,
                settings,
                embedder,
                args.index_dir,
                args.parent_characters,
                trace_path,
            )
            evaluate_questions(
                args.questions,
                rows_path,
                chatbot=Chatbot(settings, retriever=retriever),
                run_name=f"a31-{name}-{args.run_label}",
            )
        rows = load_jsonl(rows_path)
        summary = load_json(rows_path.with_suffix(".summary.json"))
        metrics[name] = compact(summary, rows)

    payload = {
        "artifact_schema_version": 1,
        "experiment": "A3.1 x HyDE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "questions": str(args.questions),
        "index_dir": str(args.index_dir),
        "controlled_variables": {
            "document_representation": "A3.1 all index",
            "embedding_model": settings.embedding_model,
            "answer_generation": "unchanged provider/model/prompt",
            "top_k_evidence": settings.top_k,
            "changed_variable": "original query vs HyDE-only vs RRF fusion vs finance-safe RRF fusion",
        },
        "hyde_configuration": {
            "candidate_k": settings.hyde_candidate_k,
            "rrf_k": settings.hyde_rrf_k,
            "original_weight": settings.hyde_original_weight,
            "hypothetical_weight": settings.hyde_hypothetical_weight,
            "max_generation_tokens": settings.hyde_llm_max_new_tokens,
            "model": settings.hyde_llm_model,
        },
        "arms": metrics,
    }
    json_path = args.output_dir / f"hyde-ablation-{args.run_label}.json"
    md_path = args.output_dir / f"hyde-ablation-{args.run_label}.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
