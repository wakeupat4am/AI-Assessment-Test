#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss

from src.config import get_settings
from src.ingestion.layout_chunker import (
    layout_aware_chunks,
    multi_granularity_chunks,
    selective_cascade_chunks,
)
from src.ingestion.paddleocr_vl import load_raw_records
from src.retrieval.embedder import Embedder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_index(
    *,
    experiment: str,
    chunks: list[dict[str, Any]],
    output_dir: Path,
    embedder: Embedder,
    embedding_model: str,
    raw_manifest: dict[str, Any],
    raw_manifest_path: Path,
    strategy: str,
    extra: dict[str, Any] | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    if not chunks:
        raise ValueError(f"{experiment} produced no chunks")
    ids = [str(chunk["chunk_id"]) for chunk in chunks]
    if len(ids) != len(set(ids)):
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate {experiment} chunk IDs: {duplicates[:5]}")

    started = time.perf_counter()
    vectors = embedder.encode_documents(
        [str(chunk["text"]) for chunk in chunks],
        batch_size=batch_size,
        show_progress=True,
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.jsonl"
    index_path = output_dir / "index.faiss"
    write_jsonl(chunks_path, chunks)
    faiss.write_index(index, str(index_path))
    metadata = {
        "artifact_schema_version": 1,
        "experiment": experiment,
        "pipeline": f"{experiment.lower()}-{strategy}+dense-e5-flatip",
        "document_parser": "PaddleOCR-VL-1.6"
        if experiment != "A4"
        else "selective PyMuPDF + PaddleOCR-VL-1.6",
        "representation": experiment.lower(),
        "strategy": strategy,
        "source_pdf": raw_manifest.get("source_pdf"),
        "source_pdf_sha256": raw_manifest.get("source_pdf_sha256"),
        "raw_fragment_count": int(raw_manifest.get("fragment_count", 0)),
        "raw_manifest_sha256": sha256(raw_manifest_path),
        "embedding_model": embedding_model,
        "embedding_device": embedder.device,
        "pdf_sheet_count": 197,
        "printed_fragment_count": 393,
        "chunk_count": len(chunks),
        "vector_dimension": int(vectors.shape[1]),
        "granularity_counts": dict(
            sorted(Counter(str(row.get("granularity", "unknown")) for row in chunks).items())
        ),
        "block_type_counts": dict(
            sorted(Counter(str(row.get("block_type", "unknown")) for row in chunks).items())
        ),
        "index_build_seconds": round(time.perf_counter() - started, 3),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunks_sha256": sha256(chunks_path),
        "index_sha256": sha256(index_path),
        **(extra or {}),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controlled A2/A3/A4 indexes")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data/processed/paddleocr_vl_1_6/raw",
    )
    parser.add_argument(
        "--baseline-pages", type=Path, default=ROOT / "data/processed/pages.jsonl"
    )
    parser.add_argument("--index-root", type=Path, default=ROOT / "data/index")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--only", choices=("A2", "A3", "A4"), action="append", dest="selected"
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.embedding_model:
        raise ValueError("EMBEDDING_MODEL is required")
    records = load_raw_records(args.raw_dir)
    if len(records) != 393:
        raise ValueError(f"Expected 393 raw fragments, got {len(records)}")
    raw_manifest_path = args.raw_dir / "manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    baseline_pages = load_jsonl(args.baseline_pages)
    if len(baseline_pages) != 393:
        raise ValueError(f"Expected 393 baseline pages, got {len(baseline_pages)}")
    selected = set(args.selected or ("A2", "A3", "A4"))
    embedder = Embedder(settings.embedding_model, settings.embedding_device)

    if "A2" in selected:
        build_index(
            experiment="A2",
            chunks=layout_aware_chunks(records),
            output_dir=args.index_root / "a2_paddleocr_vl_1_6_layout_aware",
            embedder=embedder,
            embedding_model=settings.embedding_model,
            raw_manifest=raw_manifest,
            raw_manifest_path=raw_manifest_path,
            strategy="paddleocr-layout-aware-heading-table",
            batch_size=args.batch_size,
        )
    if "A3" in selected:
        build_index(
            experiment="A3",
            chunks=multi_granularity_chunks(records),
            output_dir=args.index_root / "a3_paddleocr_vl_1_6_multigranularity",
            embedder=embedder,
            embedding_model=settings.embedding_model,
            raw_manifest=raw_manifest,
            raw_manifest_path=raw_manifest_path,
            strategy="paddleocr-row-block-page",
            extra={"page_signature_character_limit": 3200},
            batch_size=args.batch_size,
        )
    if "A4" in selected:
        chunks, cascade_stats = selective_cascade_chunks(baseline_pages, records)
        build_index(
            experiment="A4",
            chunks=chunks,
            output_dir=args.index_root / "a4_selective_cascade",
            embedder=embedder,
            embedding_model=settings.embedding_model,
            raw_manifest=raw_manifest,
            raw_manifest_path=raw_manifest_path,
            strategy="pymupdf-narrative+paddleocr-table-chart",
            extra={"cascade_stats": cascade_stats},
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
