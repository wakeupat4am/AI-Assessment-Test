#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chat.chatbot import Chatbot
from src.config import get_settings
from src.evaluation.evaluator import evaluate_questions, load_questions
from src.retrieval.b4_factory import create_b4_retriever, load_b4_config
from src.retrieval.embedder import Embedder
from src.retrieval.parent_expansion import ParentExpandingRetriever
from src.retrieval.retriever import DenseRetriever


SYSTEMS = {
    "A31_B4b_frozen": {
        "index": ROOT / "data/index/a31_semantic_multirepr/all",
        "config": ROOT / "config/b4_retrieval.json",
        "mode": "hybrid",
    },
    "A32_B4e_metric_aware": {
        "index": ROOT / "data/index/a32_metric_aware/all",
        "config": ROOT / "config/b4e_metric_aware.json",
        "mode": "metric_aware",
    },
}


def make_system(spec: dict[str, Any], embedder: Embedder) -> tuple[Any, Any]:
    settings = replace(
        get_settings(),
        index_dir=spec["index"],
        b4_bm25_artifact_path=spec["index"] / "bm25.json.gz",
        b4_config_path=spec["config"],
        b4_frozen_index_dir=SYSTEMS["A31_B4b_frozen"]["index"],
        b4_frozen_config_path=SYSTEMS["A31_B4b_frozen"]["config"],
        b4_retrieval_mode=spec["mode"],
        xrouter_enabled=False,
        hyde_enabled=False,
        b2_enabled=False,
        enable_auto_search=False,
        enable_page_dedup=False,
        enable_diversity_selection=False,
        b5_agent_enabled=False,
    )
    dense = DenseRetriever(settings, embedder=embedder, index_dir=settings.index_dir)
    child = create_b4_retriever(
        dense, settings, mode=spec["mode"], config=load_b4_config(spec["config"])
    )
    return settings, ParentExpandingRetriever(
        child, max_parent_characters=settings.a31_parent_characters
    )


def retrieval_rows(path: Path, retriever: Any, k: int) -> tuple[list[dict], dict]:
    rows = []
    for item in load_questions(path):
        started = time.perf_counter()
        chunks = retriever.retrieve(item["question"], k)
        pages = [int(chunk["printed_page"]) for chunk in chunks]
        gold = set(item["gold_printed_pages"])
        ranks = [rank for rank, page in enumerate(pages, 1) if page in gold]
        rows.append({
            "id": item["id"], "question": item["question"],
            "gold_printed_pages": item["gold_printed_pages"], "retrieved_pages": pages,
            "gold_ranks": ranks, "latency_seconds": time.perf_counter() - started,
            "trace": getattr(retriever, "last_trace", None),
        })
    judged = [row for row in rows if row["gold_printed_pages"]]
    metric = lambda k: sum(bool(row["gold_ranks"] and min(row["gold_ranks"]) <= k) for row in judged) / len(judged)
    summary = {
        "questions": len(rows), "hit_at_1": metric(1), "hit_at_5": metric(5),
        "hit_at_10": metric(10),
        "latency_mean_ms": statistics.mean(row["latency_seconds"] for row in rows) * 1000,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen A3.1+B4b vs A3.2+B4e")
    parser.add_argument("questions", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--run-type", choices=("retrieval_only", "full"), default="retrieval_only")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evaluation/experiments/metric_aware")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = get_settings()
    embedder = Embedder(base.embedding_model, base.embedding_device)
    summaries = {}
    for name, spec in SYSTEMS.items():
        settings, retriever = make_system(spec, embedder)
        path = args.output_dir / f"{name}-{args.run_label}-{args.run_type}.jsonl"
        if args.run_type == "retrieval_only":
            rows, summary = retrieval_rows(args.questions, retriever, settings.evaluation_retrieval_k)
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            evaluate_questions(args.questions, path, chatbot=Chatbot(settings, retriever=retriever), run_name=f"metric-aware-{name}-{args.run_label}")
            summary = json.loads(path.with_suffix(".summary.json").read_text(encoding="utf-8"))
        summaries[name] = summary
    output = args.output_dir / f"comparison-{args.run_label}-{args.run_type}.json"
    output.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
