#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from compare_b1_b2_answers import ROOT, build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare B4 public answers")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/evaluation/experiments/b4")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/B4_PUBLIC_ANSWERS.md")
    args = parser.parse_args()
    paths = {
        "B0": args.input_dir / "B0_one_shot-public-full.jsonl",
        "B4a BM25": args.input_dir / "B4a_bm25-public-full.jsonl",
        "B4b Hybrid RRF": args.input_dir / "B4b_hybrid_rrf-public-full.jsonl",
        "B4c Finance rerank": args.input_dir / "B4c_finance_rerank-public-full.jsonl",
        "B4d Cross-encoder": args.input_dir / "B4d_cross_encoder-public-full.jsonl",
    }
    report = build_report(paths, title="A3.1 public answer comparison — B4 retrieval ablation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
