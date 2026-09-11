#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compare_document_intelligence import (
    METRICS,
    delta,
    load_rows,
    load_summary,
    question_view,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all controlled Track A runs")
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        required=True,
        metavar=("NAME", "SUMMARY", "ROWS"),
        help="Repeat for each run; the first run is the frozen baseline",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    runs: dict[str, dict[str, Any]] = {}
    rows_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for name, summary_path, rows_path in args.run:
        if name in runs:
            raise ValueError(f"Duplicate run name: {name}")
        runs[name] = load_summary(Path(summary_path))
        rows_by_run[name] = load_rows(Path(rows_path))
    baseline_name = next(iter(runs))
    baseline_metrics = runs[baseline_name]["metrics"]

    metrics: dict[str, Any] = {}
    for name, run in runs.items():
        values = {metric: run["metrics"].get(metric) for metric in METRICS}
        values["tokens"] = run["metrics"].get("tokens")
        values["cost_projection_same_tokens_usd"] = run["metrics"].get(
            "cost_projection_same_tokens_usd"
        )
        values["failure_taxonomy"] = run.get("failure_taxonomy")
        if name != baseline_name:
            values["delta_vs_baseline"] = {
                metric: delta(values[metric], baseline_metrics.get(metric))
                for metric in METRICS
            }
        metrics[name] = values

    ids = set(rows_by_run[baseline_name])
    if any(set(rows) != ids for rows in rows_by_run.values()):
        raise ValueError("Evaluation row files do not contain the same question IDs")
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline_name,
        "runs": metrics,
        "per_question": {
            question_id: {
                name: question_view(rows[question_id])
                for name, rows in rows_by_run.items()
            }
            for question_id in sorted(ids)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
