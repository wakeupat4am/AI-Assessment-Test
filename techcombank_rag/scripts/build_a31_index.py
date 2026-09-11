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

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.ingestion.paddleocr_vl import load_raw_records
from src.ingestion.semantic_multirepr import (
    route_subset,
    semantic_multirepresentation_chunks,
)
from src.retrieval.embedder import Embedder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_subset(
    *,
    name: str,
    output_dir: Path,
    chunks: list[dict[str, Any]],
    vectors: np.ndarray,
    selected_indices: list[int],
    common_metadata: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_chunks = [chunks[index] for index in selected_indices]
    selected_vectors = np.ascontiguousarray(vectors[selected_indices])
    index = faiss.IndexFlatIP(selected_vectors.shape[1])
    index.add(selected_vectors)
    chunks_path = output_dir / "chunks.jsonl"
    index_path = output_dir / "index.faiss"
    write_jsonl(chunks_path, selected_chunks)
    faiss.write_index(index, str(index_path))
    metadata = {
        **common_metadata,
        "route_subset": name,
        "chunk_count": len(selected_chunks),
        "vector_dimension": int(selected_vectors.shape[1]),
        "granularity_counts": dict(
            sorted(Counter(str(row.get("granularity")) for row in selected_chunks).items())
        ),
        "block_type_counts": dict(
            sorted(Counter(str(row.get("block_type")) for row in selected_chunks).items())
        ),
        "chunks_sha256": sha256(chunks_path),
        "index_sha256": sha256(index_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build A3.1 semantic multi-representation route indexes"
    )
    parser.add_argument(
        "--a3-index-dir",
        type=Path,
        default=ROOT / "data/index/a3_paddleocr_vl_1_6_multigranularity",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data/processed/paddleocr_vl_1_6/raw",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/index/a31_semantic_multirepr",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--parent-character-limit", type=int, default=2400)
    parser.add_argument("--document-entity", default="Techcombank")
    parser.add_argument("--reporting-year", type=int, default=2025)
    args = parser.parse_args()

    settings = get_settings()
    a3_chunks_path = args.a3_index_dir / "chunks.jsonl"
    a3_metadata_path = args.a3_index_dir / "metadata.json"
    raw_manifest_path = args.raw_dir / "manifest.json"
    a3_chunks = load_jsonl(a3_chunks_path)
    a3_metadata = load_json(a3_metadata_path)
    raw_manifest = load_json(raw_manifest_path)
    records = load_raw_records(args.raw_dir)
    if len(records) != int(raw_manifest.get("fragment_count", len(records))):
        raise ValueError("Raw OCR record count does not match its manifest")
    if not settings.embedding_model:
        raise ValueError("EMBEDDING_MODEL is required")
    indexed_model = str(a3_metadata.get("embedding_model", ""))
    if indexed_model and indexed_model != settings.embedding_model:
        raise ValueError(
            f"A3 uses {indexed_model}, but EMBEDDING_MODEL={settings.embedding_model}"
        )

    chunks = semantic_multirepresentation_chunks(
        a3_chunks,
        records,
        parent_character_limit=args.parent_character_limit,
        document_entity=args.document_entity,
        reporting_year=args.reporting_year,
    )
    started = time.perf_counter()
    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    vectors = embedder.encode_documents(
        [str(chunk["text"]) for chunk in chunks],
        batch_size=args.batch_size,
        show_progress=True,
    )
    metric_rows = [row for row in chunks if row.get("granularity") == "metric"]
    common = {
        "artifact_schema_version": 1,
        "experiment": "A3.1",
        "pipeline": "paddleocr-a3+semantic-metric-heading-parent+dense-e5-flatip",
        "document_parser": "PaddleOCR-VL-1.6 (reused checkpoint; no OCR rerun)",
        "representation": "a3.1",
        "strategy": "row-block-page+metric-value+heading-inheritance+parent-expansion",
        "source_pdf": raw_manifest.get("source_pdf"),
        "source_pdf_sha256": raw_manifest.get("source_pdf_sha256"),
        "pdf_sheet_count": 197,
        "raw_fragment_count": len(records),
        "parent_character_limit": args.parent_character_limit,
        "document_entity": args.document_entity,
        "reporting_year": args.reporting_year,
        "embedding_model": settings.embedding_model,
        "embedding_device": embedder.device,
        "derived_a3_chunks_sha256": sha256(a3_chunks_path),
        "derived_a3_index_sha256": sha256(args.a3_index_dir / "index.faiss"),
        "raw_manifest_sha256": sha256(raw_manifest_path),
        "total_chunk_count": len(chunks),
        "semantic_metric_count": len(metric_rows),
        "inherited_heading_count": sum(bool(row.get("heading_inherited")) for row in chunks),
        "extraction_method_counts": dict(
            sorted(Counter(str(row.get("extraction_method")) for row in metric_rows).items())
        ),
        "build_seconds": round(time.perf_counter() - started, 3),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summaries = {}
    for name in ("all", "narrative", "structured"):
        indices = [
            index for index, chunk in enumerate(chunks) if route_subset(chunk, name)
        ]
        if not indices:
            raise ValueError(f"A3.1 subset {name} is empty")
        summaries[name] = write_subset(
            name=name,
            output_dir=args.output_root / name,
            chunks=chunks,
            vectors=vectors,
            selected_indices=indices,
            common_metadata=common,
        )
    manifest = {
        **common,
        "subsets": {
            name: {
                "path": str((args.output_root / name).resolve()),
                "chunk_count": summary["chunk_count"],
                "chunks_sha256": summary["chunks_sha256"],
                "index_sha256": summary["index_sha256"],
            }
            for name, summary in summaries.items()
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"A3.1 manifest: {args.output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
