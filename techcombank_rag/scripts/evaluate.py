#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import evaluate_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-interactive RAG evaluation")
    parser.add_argument("questions", type=Path)
    parser.add_argument("--output", type=Path, help="Backward-compatible explicit JSONL path")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evaluation/results")
    parser.add_argument("--run-name", default="evaluation")
    parser.add_argument("--manual-review", type=Path)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or args.output_dir / f"{args.run_name}-{timestamp}.jsonl"
    summary = output.with_suffix(".summary.json")
    rows = evaluate_questions(
        args.questions,
        output,
        summary_path=summary,
        run_name=args.run_name,
        manual_reviews_path=args.manual_review,
    )
    with summary.open(encoding="utf-8") as handle:
        metrics = json.load(handle)["metrics"]
    print(f"Wrote {len(rows)} result(s) to {output}")
    print(f"Summary: {summary}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

