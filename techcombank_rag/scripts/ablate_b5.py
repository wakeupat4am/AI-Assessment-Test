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
from src.llm.provider import create_llm_client
from src.retrieval.b4_factory import create_b4_retriever, load_b4_config
from src.retrieval.diversity import estimate_tokens
from src.retrieval.embedder import Embedder
from src.retrieval.parent_expansion import ParentExpandingRetriever
from src.retrieval.retriever import DenseRetriever
from src.retrieval.tool_agent import ToolCallingRetriever, load_b5_config


ARMS = ("B4b_hybrid", "B5_tool_agent")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def make_retriever(name: str, settings: Any, embedder: Embedder, b4_config: dict[str, Any], b5_config: dict[str, Any], llm: Any, log_path: Path) -> Any:
    dense = DenseRetriever(settings, embedder=embedder, index_dir=settings.index_dir)
    hybrid = create_b4_retriever(dense, settings, mode="hybrid", config=b4_config)
    child: Any = hybrid
    if name == "B5_tool_agent":
        child = ToolCallingRetriever(hybrid, llm, config=b5_config, log_path=log_path)
    return ParentExpandingRetriever(child, max_parent_characters=settings.a31_parent_characters)


def compact(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = summary["metrics"]
    traces = [
        (row.get("routing") or {}).get("b5")
        for row in rows
        if (row.get("routing") or {}).get("b5")
    ]
    actions = [action for trace in traces for action in trace.get("actions", [])]
    return {
        "questions": len(rows),
        "runtime_errors": summary["counts"]["runtime_errors"],
        "answer_accuracy_automatic": metrics["answer_accuracy"],
        "retrieval_hit_at_1": metrics["retrieval_hit_at_1"],
        "retrieval_hit_at_5": metrics["retrieval_hit_at_5"],
        "retrieval_hit_at_10": metrics["retrieval_hit_at_10"],
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
        "average_evidence_tokens": mean([float(sum(estimate_tokens(str(chunk.get("text", ""))) for chunk in row.get("retrieved_chunks", [])[:5])) for row in rows]),
        "agent_trigger_rate": round(sum(bool(trace.get("triggered")) for trace in traces) / len(rows), 6) if rows else 0.0,
        "planner_call_rate": round(sum(int(trace.get("planner_calls", 0)) for trace in traces) / len(rows), 6) if rows else 0.0,
        "tool_outcomes": {key: sum(action.get("outcome") == key for action in actions) for key in sorted({str(action.get("outcome")) for action in actions})},
        "calculator_successes": sum(action.get("tool") == "calculator" and action.get("outcome") == "ok" for action in actions),
        "second_retrieval_successes": sum(action.get("tool") == "hybrid_retrieve" and action.get("arguments", {}).get("query") != row.get("question") and action.get("outcome") == "ok" for row in rows for action in ((row.get("routing") or {}).get("b5") or {}).get("actions", [])),
        "failure_taxonomy": summary["failure_taxonomy"],
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# B5 tool-calling ablation — {payload['run_label']}", "",
        "| Arm | Auto acc. | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Retrieval ms | E2E p50/p95 s | Calls | Tokens | Agent trigger | Calculator ok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARMS:
        item = payload["arms"][name]
        lines.append("| " + " | ".join([
            name, f"{item['answer_accuracy_automatic']:.4f}", f"{item['retrieval_hit_at_1']:.4f}", f"{item['retrieval_hit_at_5']:.4f}", f"{item['retrieval_hit_at_10']:.4f}",
            f"{item['citation_precision']:.4f}/{item['citation_recall']:.4f}", f"{item['refusal_accuracy']:.4f}",
            f"{item['retrieval_latency_mean_seconds'] * 1000:.1f}", f"{item['latency_p50_seconds']:.3f}/{item['latency_p95_seconds']:.3f}",
            str(item["llm_calls_total"]), str(item["llm_tokens"]["total_tokens"]), f"{item['agent_trigger_rate']:.3f}", str(item["calculator_successes"]),
        ]) + " |")
    lines.extend(["", "> A3.1, B4b hybrid retrieval, answer model/prompt, questions, and top-k are fixed. B5 changes only bounded tool orchestration.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen B4b with bounded B5 tool calling")
    parser.add_argument("questions", type=Path)
    parser.add_argument("--run-label", default="public")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data/index/a31_semantic_multirepr/all")
    parser.add_argument("--b4-config", type=Path, default=ROOT / "config/b4_retrieval.json")
    parser.add_argument("--b5-config", type=Path, default=ROOT / "config/b5_tool_agent.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evaluation/experiments/b5")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = replace(
        get_settings(), index_dir=args.index_dir, b4_retrieval_mode="off", b5_agent_enabled=False,
        xrouter_enabled=False, hyde_enabled=False, b2_enabled=False, enable_auto_search=False,
        enable_page_dedup=False, enable_diversity_selection=False,
        b4_bm25_artifact_path=args.index_dir / "bm25.json.gz",
    )
    b4_config = load_b4_config(args.b4_config)
    b5_config = load_b5_config(args.b5_config)
    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    llm = create_llm_client(settings)
    arms: dict[str, dict[str, Any]] = {}
    for name in ARMS:
        rows_path = args.output_dir / f"{name}-{args.run_label}.jsonl"
        trace_path = args.output_dir / f"{name}-{args.run_label}-trace.jsonl"
        trace_path.unlink(missing_ok=True)
        retriever = make_retriever(name, settings, embedder, b4_config, b5_config, llm, trace_path)
        evaluate_questions(args.questions, rows_path, chatbot=Chatbot(settings, retriever=retriever, llm_client=llm), run_name=f"b5-{name}-{args.run_label}")
        arms[name] = compact(load_json(rows_path.with_suffix(".summary.json")), load_jsonl(rows_path))
    payload = {
        "artifact_schema_version": 1,
        "experiment": "B5 bounded tool calling over frozen A3.1+B4b",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "questions": str(args.questions),
        "index_dir": str(args.index_dir),
        "controlled_variables": {"document_representation": "A3.1", "first_stage_retrieval": "B4b dense+BM25 weighted RRF", "answer_generation": "unchanged", "changed_variable": "bounded B5 tool calling"},
        "b5_config": b5_config,
        "arms": arms,
    }
    stem = args.output_dir / f"b5-ablation-{args.run_label}"
    stem.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))
    print(f"JSON: {stem.with_suffix('.json')}")
    print(f"Markdown: {stem.with_suffix('.md')}")


if __name__ == "__main__":
    main()
