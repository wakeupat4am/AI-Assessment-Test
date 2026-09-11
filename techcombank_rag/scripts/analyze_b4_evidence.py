#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation/experiments/b4"
ARMS = ("B0_one_shot", "B4a_bm25", "B4b_hybrid_rrf", "B4c_finance_rerank", "B4d_cross_encoder")
SPLITS = ("public", "dev", "holdout")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_rank(row: dict[str, Any]) -> str:
    ranks = row.get("gold_ranks", [])
    return str(min(ranks)) if ranks else "—"


def main() -> None:
    output = ROOT / "docs/experiments/B4_EVIDENCE_ANALYSIS.md"
    manual = load(DATA / "manual_review_all.json")
    full = {split: load(DATA / f"b4-ablation-{split}-full.json") for split in SPLITS}
    screen_rows = {
        arm: {row["id"]: row for row in load_jsonl(DATA / f"{arm}-dev-retrieval.jsonl")}
        for arm in ARMS
    }
    answerable_ids = [
        row_id for row_id, row in screen_rows["B0_one_shot"].items() if row["answerable"] is True
    ]
    dense_hits = {row_id for row_id in answerable_ids if screen_rows["B0_one_shot"][row_id]["hit_at_10"]}
    bm25_hits = {row_id for row_id in answerable_ids if screen_rows["B4a_bm25"][row_id]["hit_at_10"]}
    lines = [
        "# B4 evidence analysis", "",
        "## End-to-end metrics", "",
        "| Split | Arm | Auto acc. | Manual public | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Retrieval ms | p50/p95 s | Tokens | GPT-5.6 projection |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        for arm in ARMS:
            item = full[split]["arms"][arm]
            manual_value = manual["splits"][split][arm]["accuracy"]
            projection = item["cost_projection_same_tokens_usd"].get("gpt-5.6")
            lines.append(
                f"| {split} | {arm} | {item['answer_accuracy_automatic']:.4f} | "
                f"{manual_value:.2f}"
            )
            lines[-1] += (
                f" | {item['retrieval_hit_at_1']:.4f} | {item['retrieval_hit_at_5']:.4f} | {item['retrieval_hit_at_10']:.4f}"
                f" | {item['citation_precision']:.4f}/{item['citation_recall']:.4f}"
                f" | {item['retrieval_latency_mean_seconds'] * 1000:.1f}"
                f" | {item['latency_p50_seconds']:.3f}/{item['latency_p95_seconds']:.3f}"
                f" | {item['llm_tokens']['total_tokens']} | ${projection:.6f} |"
            )
    lines.extend([
        "", "## Dense/BM25 complementarity on dev top-10 outputs", "",
        f"- Answerable questions: {len(answerable_ids)}",
        f"- Dense Hit@10: {len(dense_hits)}/{len(answerable_ids)}",
        f"- BM25 Hit@10: {len(bm25_hits)}/{len(answerable_ids)}",
        f"- Union of the two independently returned top-10 sets: {len(dense_hits | bm25_hits)}/{len(answerable_ids)}",
        f"- Found only by BM25: {', '.join(sorted(bm25_hits - dense_hits)) or 'none'}",
        f"- Found only by dense: {', '.join(sorted(dense_hits - bm25_hits)) or 'none'}",
        f"- Missed by both: {', '.join(sorted(set(answerable_ids) - dense_hits - bm25_hits)) or 'none'}",
        "", "## Gold-page rank by dev question", "",
        "| ID | Gold pages | B0 | B4a | B4b | B4c | B4d |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row_id in answerable_ids:
        gold = ", ".join(str(value) for value in screen_rows["B0_one_shot"][row_id]["gold_printed_pages"])
        ranks = [first_rank(screen_rows[arm][row_id]) for arm in ARMS]
        lines.append(f"| {row_id} | {gold} | " + " | ".join(ranks) + " |")
    lines.extend([
        "", "Hybrid internally fuses top-20 from each source, so it can recover a gold page that sits below rank 10 in both individual outputs (for example dev-11).", "",
        "## Manual semantic review", "",
        f"> {manual['criteria']}", "",
        "| Split | Arm | Accuracy | Incorrect IDs |", "|---|---|---:|---|",
    ])
    for split in SPLITS:
        for arm in ARMS:
            item = manual["splits"][split][arm]
            incorrect = ", ".join(item["incorrect_ids"]) or "none"
            lines.append(f"| {split} | {arm} | {item['accuracy']:.2f} | {incorrect} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
