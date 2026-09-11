#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summary_for(rows_path: Path) -> dict[str, Any]:
    return load_json(rows_path.with_suffix(".summary.json"))["metrics"]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def clean(text: Any) -> str:
    return str(text or "").replace("\n", " ").strip()


def build_report(
    paths: dict[str, Path],
    *,
    title: str = "Public answer comparison — B1a vs B1b vs B2",
) -> str:
    rows_by_system = {
        name: {row["id"]: row for row in load_jsonl(path)}
        for name, path in paths.items()
    }
    first = next(iter(rows_by_system.values()))
    common_ids = [row_id for row_id in first if all(row_id in rows for rows in rows_by_system.values())]
    metrics = {name: summary_for(path) for name, path in paths.items()}

    lines = [
        f"# {title}",
        "",
        "> All systems use the same 10 public questions and the unchanged answer-generation layer. Automatic answer accuracy is diagnostic, not exact-match accuracy.",
        "",
        "## Full metrics",
        "",
        "| System | Auto answer accuracy | Hit@1 | Hit@5 | Hit@10 | Recall@5 | Recall@10 | Citation P | Citation R | Citation valid | Refusal | p50 s | p95 s | LLM tokens | Cost USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in metrics.items():
        token_total = (item.get("tokens") or {}).get("total_tokens")
        lines.append(
            "| " + " | ".join(
                [
                    name,
                    fmt(item.get("answer_accuracy")),
                    fmt(item.get("retrieval_hit_at_1")),
                    fmt(item.get("retrieval_hit_at_5")),
                    fmt(item.get("retrieval_hit_at_10")),
                    fmt(item.get("retrieval_recall_at_5")),
                    fmt(item.get("retrieval_recall_at_10")),
                    fmt(item.get("citation_precision")),
                    fmt(item.get("citation_recall")),
                    fmt(item.get("citation_validator_pass_rate")),
                    fmt(item.get("refusal_accuracy")),
                    fmt(item.get("latency_seconds_p50"), 3),
                    fmt(item.get("latency_seconds_p95"), 3),
                    fmt(token_total),
                    fmt(item.get("cost_usd_total"), 6),
                ]
            ) + " |"
        )

    lines.extend(["", "## Outputs for the 10 public questions", ""])
    for row_id in common_ids:
        reference = first[row_id]
        gold_pages = ", ".join(str(value) for value in reference.get("gold_printed_pages", [])) or "none"
        lines.extend(
            [
                f"### {row_id} — {clean(reference.get('question'))}",
                "",
                f"Gold: {clean(reference.get('gold_answer'))}",
                "",
                f"Gold pages: {gold_pages}",
                "",
            ]
        )
        for name, rows in rows_by_system.items():
            row = rows[row_id]
            citations = ", ".join(str(value) for value in row.get("citations", [])) or "none"
            score = (row.get("scores") or {}).get("automatic_correct")
            b2 = (row.get("routing") or {}).get("b2") or {}
            suffix = ""
            if name == "B2" and b2:
                suffix = f"; rounds={b2.get('search_rounds')}; stop={b2.get('stop_reason')}"
            lines.extend(
                [
                    f"- {name}: {clean(row.get('answer'))}",
                    f"  - citations={citations}; automatic_score={fmt(score)}{suffix}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 10 public answers across B1a, B1b and B2")
    parser.add_argument("--b0", type=Path)
    parser.add_argument(
        "--b1a",
        type=Path,
        default=ROOT / "data/evaluation/experiments/b2/A_baseline_b1a-public.jsonl",
    )
    parser.add_argument(
        "--b1b",
        type=Path,
        default=ROOT / "data/evaluation/experiments/b1b/b1b-xrouter-public.jsonl",
    )
    parser.add_argument(
        "--b2",
        type=Path,
        default=ROOT / "data/evaluation/experiments/b2/E_full_b2-public.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/experiments/B1A_B1B_B2_PUBLIC_ANSWERS.md",
    )
    args = parser.parse_args()
    paths = {"B1a": args.b1a, "B1b": args.b1b, "B2": args.b2}
    title = "Public answer comparison — B1a vs B1b vs B2"
    if args.b0:
        paths = {"B0": args.b0, **paths}
        title = "A3.1 public answer comparison — B0 vs B1a vs B1b vs B2"
    report = build_report(paths, title=title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
