#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chat.chatbot import Chatbot
from src.config import get_settings
from src.evaluation.evaluator import evaluate_questions, load_questions
from src.retrieval.b4_factory import create_b4_retriever, load_b4_config
from src.retrieval.diversity import estimate_tokens
from src.retrieval.embedder import Embedder
from src.retrieval.parent_expansion import ParentExpandingRetriever
from src.retrieval.retriever import DenseRetriever


ARMS = ("B0_one_shot", "B4a_bm25", "B4b_hybrid_rrf", "B4c_finance_rerank", "B4d_cross_encoder")
MODES = {
    "B4a_bm25": "bm25",
    "B4b_hybrid_rrf": "hybrid",
    "B4c_finance_rerank": "finance_rerank",
    "B4d_cross_encoder": "cross_encoder",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_retriever(name: str, settings: Any, embedder: Embedder, config: dict[str, Any], parent_chars: int) -> Any:
    dense = DenseRetriever(settings, embedder=embedder, index_dir=settings.index_dir)
    child = dense if name == "B0_one_shot" else create_b4_retriever(
        dense, settings, mode=MODES[name], config=config
    )
    return ParentExpandingRetriever(child, max_parent_characters=parent_chars)


def retrieval_screen(questions_path: Path, retriever: Any, retrieval_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in load_questions(questions_path):
        started = time.perf_counter()
        chunks = retriever.retrieve(item["question"], retrieval_k)
        latency = time.perf_counter() - started
        pages = [int(chunk["printed_page"]) for chunk in chunks]
        gold = set(int(page) for page in item["gold_printed_pages"])
        ranks = [rank for rank, page in enumerate(pages, 1) if page in gold]
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answerable": item["answerable"],
                "gold_printed_pages": item["gold_printed_pages"],
                "retrieved_pages": pages,
                "gold_ranks": ranks,
                "hit_at_1": bool(ranks and min(ranks) <= 1),
                "hit_at_5": bool(ranks and min(ranks) <= 5),
                "hit_at_10": bool(ranks and min(ranks) <= 10),
                "latency_seconds": round(latency, 6),
                "trace": getattr(retriever, "last_trace", None),
            }
        )
    judged = [row for row in rows if row["answerable"] is True]
    latencies = [float(row["latency_seconds"]) for row in rows]
    metrics = {
        "questions": len(rows),
        "answerable": len(judged),
        "hit_at_1": mean([float(row["hit_at_1"]) for row in judged]),
        "hit_at_5": mean([float(row["hit_at_5"]) for row in judged]),
        "hit_at_10": mean([float(row["hit_at_10"]) for row in judged]),
        "mrr_at_10": mean([1.0 / min(row["gold_ranks"]) if row["gold_ranks"] and min(row["gold_ranks"]) <= 10 else 0.0 for row in judged]),
        "latency_mean_ms": round(statistics.mean(latencies) * 1000, 3),
        "latency_p50_ms": round(statistics.median(latencies) * 1000, 3),
    }
    return rows, metrics


def compact_full(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = summary["metrics"]
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
        "retrieval_latency_mean_seconds": mean([float(row.get("latency_breakdown", {}).get("retrieval_seconds", 0.0)) for row in rows if not row.get("error")]),
        "llm_calls_total": metrics["llm_calls_total"],
        "calls_by_purpose": metrics["calls_by_purpose"],
        "llm_tokens": metrics["tokens"],
        "cost_usd_total": metrics["cost_usd_total"],
        "cost_projection_same_tokens_usd": metrics["cost_projection_same_tokens_usd"],
        "average_returned_candidates": mean([float(len(row.get("retrieved_chunks", []))) for row in rows]),
        "average_evidence_tokens": mean([float(sum(estimate_tokens(str(chunk.get("text", ""))) for chunk in row.get("retrieved_chunks", [])[:5])) for row in rows]),
        "failure_taxonomy": summary["failure_taxonomy"],
    }


def markdown(payload: dict[str, Any]) -> str:
    is_screen = payload["run_type"] == "retrieval_only"
    if is_screen:
        lines = [
            f"# B4 retrieval-only ablation — {payload['run_label']}", "",
            "| Arm | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Mean ms | p50 ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in ARMS:
            item = payload["arms"][name]
            lines.append(f"| {name} | {item['hit_at_1']:.4f} | {item['hit_at_5']:.4f} | {item['hit_at_10']:.4f} | {item['mrr_at_10']:.4f} | {item['latency_mean_ms']:.1f} | {item['latency_p50_ms']:.1f} |")
    else:
        lines = [
            f"# B4 end-to-end ablation — {payload['run_label']}", "",
            "| Arm | Auto acc. | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Retrieval ms | E2E p50/p95 s | Calls | Tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name in ARMS:
            item = payload["arms"][name]
            lines.append("| " + " | ".join([
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
            ]) + " |")
    lines.extend(["", "> A3.1 chunks, E5 dense index, answer model/prompt, questions, and evaluation k are fixed.", "> Only the first-stage retrieval/reranking method changes.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled B4 retrieval ablation over frozen A3.1")
    parser.add_argument("questions", type=Path)
    parser.add_argument("--run-label", default="public")
    parser.add_argument("--run-type", choices=("retrieval_only", "full"), default="full")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data/index/a31_semantic_multirepr/all")
    parser.add_argument("--config", type=Path, default=ROOT / "config/b4_retrieval.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evaluation/experiments/b4")
    parser.add_argument("--parent-characters", type=int, default=1800)
    parser.add_argument("--only", nargs="+", choices=ARMS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = replace(
        get_settings(), index_dir=args.index_dir,
        b4_retrieval_mode="off", xrouter_enabled=False, hyde_enabled=False,
        b2_enabled=False, enable_auto_search=False, enable_page_dedup=False,
        enable_diversity_selection=False,
        b4_config_path=args.config,
        b4_bm25_artifact_path=args.index_dir / "bm25.json.gz",
    )
    config = load_b4_config(args.config)
    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    selected = set(args.only or ARMS)
    arms: dict[str, dict[str, Any]] = {}
    suffix = "retrieval" if args.run_type == "retrieval_only" else "full"
    for name in ARMS:
        rows_path = args.output_dir / f"{name}-{args.run_label}-{suffix}.jsonl"
        if name in selected:
            retriever = make_retriever(name, settings, embedder, config, args.parent_characters)
            if args.run_type == "retrieval_only":
                rows, metrics = retrieval_screen(args.questions, retriever, settings.evaluation_retrieval_k)
                rows_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
                rows_path.with_suffix(".summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            else:
                evaluate_questions(args.questions, rows_path, chatbot=Chatbot(settings, retriever=retriever), run_name=f"b4-{name}-{args.run_label}")
        rows = load_jsonl(rows_path)
        summary = load_json(rows_path.with_suffix(".summary.json"))
        arms[name] = summary if args.run_type == "retrieval_only" else compact_full(summary, rows)
    payload = {
        "artifact_schema_version": 1,
        "experiment": "B4 sparse-dense retrieval over frozen A3.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "run_type": args.run_type,
        "questions": str(args.questions),
        "questions_sha256": sha256(args.questions),
        "index_dir": str(args.index_dir),
        "chunks_sha256": sha256(args.index_dir / "chunks.jsonl"),
        "bm25_artifact_sha256": sha256(args.index_dir / "bm25.json.gz"),
        "config": config,
        "config_sha256": sha256(args.config),
        "controlled_variables": {
            "document_representation": "A3.1 all index",
            "dense_embedding_model": settings.embedding_model,
            "answer_generation": "unchanged provider/model/prompt",
            "evaluation_retrieval_k": settings.evaluation_retrieval_k,
            "top_k_evidence": settings.top_k,
            "changed_variable": "one-shot dense vs BM25 vs hybrid RRF vs deterministic finance reranker vs multilingual cross-encoder",
        },
        "arms": arms,
    }
    stem = args.output_dir / f"b4-ablation-{args.run_label}-{suffix}"
    stem.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))
    print(f"JSON: {stem.with_suffix('.json')}")
    print(f"Markdown: {stem.with_suffix('.md')}")


if __name__ == "__main__":
    main()
