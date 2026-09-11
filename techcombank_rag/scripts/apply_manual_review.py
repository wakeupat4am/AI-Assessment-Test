#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.evaluation.evaluator import (
    load_manual_reviews,
    load_questions,
    score_response,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply auditable human labels to a saved run")
    parser.add_argument("results", type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--manual-review", required=True, type=Path)
    parser.add_argument("--run-name", default="baseline-a0-b0")
    args = parser.parse_args()

    questions = {row["id"]: row for row in load_questions(args.questions)}
    reviews = load_manual_reviews(args.manual_review)
    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if set(reviews) != set(questions):
        missing = sorted(set(questions) - set(reviews))
        extra = sorted(set(reviews) - set(questions))
        raise ValueError(f"Manual review IDs must match questions; missing={missing}, extra={extra}")
    for row in rows:
        item = questions[row["id"]]
        row["scores"] = score_response(item, row, reviews[row["id"]])

    summary_path = args.results.with_suffix(".summary.json")
    previous_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    args.results.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = summarize(rows, get_settings(), args.run_name, args.questions)
    for key in ("created_at_utc", "environment", "artifacts"):
        if key in previous_summary:
            summary[key] = previous_summary[key]
    summary["manual_review"] = {
        "path": str(args.manual_review),
        "sha256": hashlib.sha256(args.manual_review.read_bytes()).hexdigest(),
        "reviewed_questions": len(reviews),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "results": str(args.results),
                "summary": str(summary_path),
                "manual_answer_accuracy": summary["metrics"]["manual_answer_accuracy"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
