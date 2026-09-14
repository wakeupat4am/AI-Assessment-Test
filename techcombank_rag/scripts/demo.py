#!/usr/bin/env python3
"""Run a question file and stream auditable answers for an unedited demo."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import evaluate_questions, load_questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "questions",
        nargs="?",
        type=Path,
        default=ROOT / "data/evaluation/public.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evaluation/results")
    parser.add_argument("--run-name", default="live-demo")
    args = parser.parse_args()

    total = len(load_questions(args.questions))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"{args.run_name}-{timestamp}.jsonl"
    counter = 0

    def display(row: dict[str, Any]) -> None:
        nonlocal counter
        counter += 1
        citations = row.get("citations") or []
        citation_text = ", ".join(str(page) for page in citations) or "none"
        print(f"\n[{counter}/{total}] {row['question']}", flush=True)
        print(f"Answer: {row['answer'] or '[runtime error]'}", flush=True)
        print(
            f"Citations: {citation_text} | Refused: {row['refused']} | "
            f"Latency: {row['latency_seconds']:.3f}s",
            flush=True,
        )
        if row.get("error"):
            print(f"Error: {row['error']}", flush=True)

    evaluate_questions(
        args.questions,
        output,
        summary_path=output.with_suffix(".summary.json"),
        run_name=args.run_name,
        on_result=display,
    )
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    print("\n=== Run summary ===", flush=True)
    for key in (
        "automatic_answer_accuracy",
        "retrieval_hit_at_5",
        "citation_precision",
        "citation_recall",
        "refusal_accuracy",
        "latency_seconds_p50",
        "latency_seconds_p95",
        "llm_calls_total",
    ):
        print(f"{key}: {metrics.get(key)}", flush=True)
    print(f"Raw results: {output}", flush=True)
    print(f"Summary: {output.with_suffix('.summary.json')}", flush=True)


if __name__ == "__main__":
    main()
