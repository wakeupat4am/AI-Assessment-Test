#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_metric_aware import SYSTEMS, make_system
from src.chat.chatbot import Chatbot
from src.config import get_settings
from src.retrieval.embedder import Embedder


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")


def normalized_numbers(text: str) -> set[str]:
    return {value.replace(".", "").replace(",", ".") for value in NUMBER_RE.findall(text)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen/new metric multi-turn behavior")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data/evaluation/multi_turn_metric.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/evaluation/experiments/metric_aware/multiturn-comparison.json",
    )
    args = parser.parse_args()
    conversations = json.loads(args.questions.read_text(encoding="utf-8"))
    base = get_settings()
    embedder = Embedder(base.embedding_model, base.embedding_device)
    payload = {"questions": str(args.questions), "systems": {}}
    for name, spec in SYSTEMS.items():
        settings, retriever = make_system(spec, embedder)
        bot = Chatbot(settings, retriever=retriever)
        system_rows = []
        for conversation in conversations:
            history = []
            turns = []
            for turn in conversation["turns"]:
                response = bot.ask(turn["question"], history)
                expected = normalized_numbers(turn["gold_answer"])
                actual = normalized_numbers(response["answer"])
                gold_pages = set(turn["gold_printed_pages"])
                row = {
                    "question": turn["question"],
                    "standalone_query": response["standalone_query"],
                    "gold_answer": turn["gold_answer"],
                    "answer": response["answer"],
                    "gold_printed_pages": turn["gold_printed_pages"],
                    "citations": response["citations"],
                    "numeric_coverage": expected.issubset(actual),
                    "citation_overlap": bool(gold_pages & set(response["citations"])),
                    "citation_coverage": gold_pages.issubset(set(response["citations"])),
                    "latency_seconds": response["latency_seconds"],
                    "answer_mode": response.get("answer_mode", "llm"),
                    "routing": response.get("routing"),
                }
                turns.append(row)
                history.extend(
                    [
                        {"role": "user", "content": turn["question"]},
                        {"role": "assistant", "content": response["answer"]},
                    ]
                )
            system_rows.append({"id": conversation["id"], "turns": turns})
        all_turns = [turn for row in system_rows for turn in row["turns"]]
        payload["systems"][name] = {
            "conversations": system_rows,
            "metrics": {
                "turns": len(all_turns),
                "numeric_coverage": sum(row["numeric_coverage"] for row in all_turns) / len(all_turns),
                "citation_overlap": sum(row["citation_overlap"] for row in all_turns) / len(all_turns),
                "citation_coverage": sum(row["citation_coverage"] for row in all_turns) / len(all_turns),
                "latency_mean_seconds": sum(row["latency_seconds"] for row in all_turns) / len(all_turns),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["systems"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
