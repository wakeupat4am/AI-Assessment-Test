#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from compare_b1_b2_answers import ROOT, build_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the 10 public answers across the fixed-A3.1 HyDE ablation"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data/evaluation/experiments/hyde",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/experiments/B3_HYDE_PUBLIC_ANSWERS.md",
    )
    args = parser.parse_args()
    paths = {
        "B0 (one-shot)": args.input_dir / "B0_one_shot-public.jsonl",
        "B3a (HyDE-only)": args.input_dir / "B3a_hyde_only-public.jsonl",
        "B3b (query + HyDE fusion)": args.input_dir / "B3b_fusion-public.jsonl",
        "B3c (safe fusion)": args.input_dir / "B3c_safe_fusion-public.jsonl",
    }
    report = build_report(
        paths,
        title="A3.1 public answer comparison — B0 vs HyDE variants",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
