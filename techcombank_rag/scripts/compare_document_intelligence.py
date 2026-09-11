#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = (
    "retrieval_recall_at_1",
    "retrieval_recall_at_5",
    "retrieval_recall_at_10",
    "answer_accuracy",
    "manual_answer_accuracy",
    "citation_precision",
    "citation_recall",
    "citation_validator_pass_rate",
    "refusal_accuracy",
    "latency_seconds_mean",
    "latency_seconds_p50",
    "latency_seconds_p95",
    "llm_calls_total",
    "cost_usd_total",
)


def load_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value.get("metrics"), dict):
        raise ValueError(f"Invalid evaluation summary: {path}")
    return value


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def first_gold_rank(row: dict[str, Any]) -> int | None:
    gold = set(int(page) for page in row.get("gold_printed_pages", []))
    for rank, page in enumerate(row.get("retrieved_pages", []), 1):
        if int(page) in gold:
            return rank
    return None


def question_view(row: dict[str, Any]) -> dict[str, Any]:
    scores = row.get("scores", {})
    return {
        "first_gold_rank": first_gold_rank(row),
        "retrieved_pages": row.get("retrieved_pages", []),
        "citations": row.get("citations", []),
        "citation_recall": scores.get("citation_recall"),
        "correct": scores.get("effective_correct"),
        "failure_category": scores.get("failure_category"),
        "latency_seconds": row.get("latency_seconds"),
    }


def delta(candidate: Any, baseline: Any) -> float | None:
    if not isinstance(candidate, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return round(float(candidate) - float(baseline), 6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare A0, A1a and A1b summaries")
    parser.add_argument("--a0", required=True, type=Path)
    parser.add_argument("--a1a", required=True, type=Path)
    parser.add_argument("--a1b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--a0-rows", type=Path)
    parser.add_argument("--a1a-rows", type=Path)
    parser.add_argument("--a1b-rows", type=Path)
    args = parser.parse_args()

    runs = {
        "A0": load_summary(args.a0),
        "A1a": load_summary(args.a1a),
        "A1b": load_summary(args.a1b),
    }
    a0_metrics = runs["A0"]["metrics"]
    comparison: dict[str, Any] = {}
    for name, run in runs.items():
        values = {metric: run["metrics"].get(metric) for metric in METRICS}
        values["tokens"] = run["metrics"].get("tokens")
        values["cost_projection_same_tokens_usd"] = run["metrics"].get(
            "cost_projection_same_tokens_usd"
        )
        values["failure_taxonomy"] = run.get("failure_taxonomy")
        if name != "A0":
            values["delta_vs_A0"] = {
                metric: delta(values[metric], a0_metrics.get(metric))
                for metric in METRICS
            }
        comparison[name] = values
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "controlled_variables": {
            "chunk_size_characters": 3200,
            "overlap_characters": 480,
            "embedding_model": "intfloat/multilingual-e5-small",
            "retrieval": "normalized dense vectors + FAISS IndexFlatIP",
            "generator": "same configured Qwen model and answer prompt",
        },
        "runs": comparison,
    }
    row_paths = {
        "A0": args.a0_rows,
        "A1a": args.a1a_rows,
        "A1b": args.a1b_rows,
    }
    if any(row_paths.values()) and not all(row_paths.values()):
        raise ValueError("Provide all of --a0-rows, --a1a-rows and --a1b-rows")
    if all(row_paths.values()):
        loaded_rows = {
            name: load_rows(path) for name, path in row_paths.items() if path is not None
        }
        ids = set(loaded_rows["A0"])
        if any(set(rows) != ids for rows in loaded_rows.values()):
            raise ValueError("Evaluation row files do not contain the same question ids")
        output["per_question"] = {
            question_id: {
                name: question_view(rows[question_id])
                for name, rows in loaded_rows.items()
            }
            for question_id in sorted(ids)
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
