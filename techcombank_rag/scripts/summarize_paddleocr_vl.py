#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.paddleocr_vl import load_raw_records, markdown_to_plain_text


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a PaddleOCR-VL raw checkpoint")
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = load_raw_records(args.raw_dir)
    seconds = [float(row.get("ocr_seconds", 0)) for row in records]
    markdown = [str(row.get("markdown", "")) for row in records]
    plain = [markdown_to_plain_text(text) for text in markdown]
    mtimes = [path.stat().st_mtime for path in args.raw_dir.glob("fragment_*.json")]
    backend_counts = Counter(str(row.get("vl_backend", "unknown")) for row in records)
    native_indices = [
        int(row["logical_index"])
        for row in records
        if row.get("vl_backend") == "paddle-native"
    ]
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fragment_count": len(records),
        "unique_printed_pages": len({int(row["printed_page"]) for row in records}),
        "backend_counts": dict(sorted(backend_counts.items())),
        "native_fallback_logical_indices": native_indices,
        "blank_fragment_count": sum(not text.strip() for text in markdown),
        "markdown_table_fragment_count": sum(
            "|" in text or "<table" in text.lower() for text in markdown
        ),
        "markdown_characters": sum(len(text) for text in markdown),
        "plain_characters": sum(len(text) for text in plain),
        "plain_to_markdown_character_ratio": round(
            sum(len(text) for text in plain) / max(1, sum(len(text) for text in markdown)),
            6,
        ),
        "ocr_seconds_recorded": {
            "sum": round(sum(seconds), 3),
            "mean": round(statistics.mean(seconds), 3) if seconds else None,
            "median": round(statistics.median(seconds), 3) if seconds else None,
            "p95": percentile(seconds, 0.95),
            "max": round(max(seconds), 3) if seconds else None,
        },
        "checkpoint_window_seconds": round(max(mtimes) - min(mtimes), 3)
        if mtimes
        else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
