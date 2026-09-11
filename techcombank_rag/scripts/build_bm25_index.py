#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.b4_factory import load_b4_config
from src.retrieval.bm25 import build_bm25_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic B4 BM25 artifact")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT / "data/index/a31_semantic_multirepr/all",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/b4_retrieval.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_b4_config(args.config)
    output = args.output or args.index_dir / "bm25.json.gz"
    metadata = build_bm25_artifact(
        args.index_dir / "chunks.jsonl",
        output,
        tokenizer=config["tokenizer"],
        k1=float(config["bm25"]["k1"]),
        b=float(config["bm25"]["b"]),
    )
    metadata_path = output.with_suffix("").with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"artifact": str(output), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
