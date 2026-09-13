#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chat.chatbot import Chatbot
from src.config import get_settings
from src.evaluation.evaluator import evaluate_questions


DEFAULT_SPLITS = {
    "public": ROOT / "data/evaluation/public.json",
    "dev20": ROOT / "data/evaluation/dev_20_stateless_reviewed.json",
    "holdout": ROOT / "data/evaluation/holdout.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen B6 instance across public/dev/holdout splits"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/evaluation/experiments/metric_aware",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(DEFAULT_SPLITS),
        default=list(DEFAULT_SPLITS),
    )
    parser.add_argument("--run-prefix", default="A32_B4e_B6-final")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.financial_reasoning_enabled:
        raise ValueError("Set FINANCIAL_REASONING_ENABLED=true for this benchmark")
    bot = Chatbot(settings)
    summaries = {}
    for split in args.splits:
        questions = DEFAULT_SPLITS[split]
        output = args.output_dir / f"{args.run_prefix}-{split}.jsonl"
        evaluate_questions(
            questions,
            output,
            chatbot=bot,
            summary_path=output.with_suffix(".summary.json"),
            run_name=f"{args.run_prefix}-{split}",
        )
        summaries[split] = json.loads(
            output.with_suffix(".summary.json").read_text(encoding="utf-8")
        )["metrics"]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
