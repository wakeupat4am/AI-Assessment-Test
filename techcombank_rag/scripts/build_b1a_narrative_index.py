#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive B1a paragraph/section index from frozen A2 vectors"
    )
    parser.add_argument(
        "--source-index",
        type=Path,
        default=ROOT / "data/index/a2_paddleocr_vl_1_6_layout_aware",
    )
    parser.add_argument(
        "--output-index",
        type=Path,
        default=ROOT / "data/index/b1a_narrative_index",
    )
    args = parser.parse_args()

    source_chunks_path = args.source_index / "chunks.jsonl"
    source_index_path = args.source_index / "index.faiss"
    source_metadata_path = args.source_index / "metadata.json"
    chunks = load_jsonl(source_chunks_path)
    source_index = faiss.read_index(str(source_index_path))
    if source_index.ntotal != len(chunks):
        raise ValueError("Source A2 vector count does not match chunks.jsonl")
    selected = [
        (position, chunk)
        for position, chunk in enumerate(chunks)
        if chunk.get("block_type") in {"narrative", "fallback_page"}
    ]
    if not selected:
        raise ValueError("No A2 narrative/section chunks found")
    vectors = np.vstack(
        [source_index.reconstruct(position) for position, _ in selected]
    ).astype("float32")
    output_index = faiss.IndexFlatIP(source_index.d)
    output_index.add(vectors)
    args.output_index.mkdir(parents=True, exist_ok=True)
    chunks_path = args.output_index / "chunks.jsonl"
    index_path = args.output_index / "index.faiss"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for _, chunk in selected:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    faiss.write_index(output_index, str(index_path))
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        "artifact_schema_version": 1,
        "experiment": "B1a-narrative-index",
        "pipeline": "A2-filtered-paragraph-section-vectors",
        "strategy": "reuse frozen A2 vectors where block_type is narrative/fallback_page",
        "source_index": portable(args.source_index),
        "source_index_sha256": sha256(source_index_path),
        "source_chunks_sha256": sha256(source_chunks_path),
        "source_pdf": source_metadata.get("source_pdf"),
        "source_pdf_sha256": source_metadata.get("source_pdf_sha256"),
        "embedding_model": source_metadata.get("embedding_model"),
        "vector_dimension": source_index.d,
        "chunk_count": len(selected),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunks_sha256": sha256(chunks_path),
        "index_sha256": sha256(index_path),
    }
    (args.output_index / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
