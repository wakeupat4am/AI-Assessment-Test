#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss

from src.config import get_settings
from src.ingestion.chunker import chunk_pages
from src.ingestion.page_mapper import assign_printed_pages, load_overrides, save_mapping
from src.ingestion.pdf_parser import (
    clean_page_text,
    extract_pdf_fragments,
    find_repeated_margin_lines,
)
from src.retrieval.embedder import Embedder


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_source_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a Techcombank Annual Report PDF")
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--mapping-overrides", type=Path)
    parser.add_argument("--chunk-size", type=int, default=3200)
    parser.add_argument("--overlap", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    started = time.perf_counter()
    settings = get_settings()
    if not settings.embedding_model:
        raise ValueError("EMBEDDING_MODEL is required to build an index")
    fragments = extract_pdf_fragments(args.pdf)
    mapped, mapping_summary = assign_printed_pages(
        fragments, load_overrides(args.mapping_overrides)
    )
    repeated_lines = find_repeated_margin_lines(mapped)
    pages: list[dict[str, Any]] = []
    for fragment in mapped:
        page = {key: value for key, value in fragment.items() if key != "raw_text"}
        page["text"] = clean_page_text(
            fragment["raw_text"], repeated_lines, fragment["printed_page"]
        )
        pages.append(page)

    chunks = chunk_pages(pages, chunk_size=args.chunk_size, overlap=args.overlap)
    if not chunks:
        raise RuntimeError("No chunks were extracted from the PDF")
    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    vectors = embedder.encode_documents(
        [chunk["text"] for chunk in chunks],
        batch_size=args.batch_size,
        show_progress=True,
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(settings.processed_dir / "pages.jsonl", pages)
    write_jsonl(settings.processed_dir / "chunks.jsonl", chunks)
    save_mapping(mapping_summary, settings.processed_dir / "page_mapping.json")
    index_chunks_path = settings.index_dir / "chunks.jsonl"
    index_path = settings.index_dir / "index.faiss"
    write_jsonl(index_chunks_path, chunks)
    faiss.write_index(index, str(index_path))
    elapsed = time.perf_counter() - started
    metadata = {
        "artifact_schema_version": 1,
        "pipeline": "baseline-a0-pymupdf-text+dense-e5-flatip",
        "source_pdf": portable_source_path(args.pdf, settings.repository_root),
        "source_pdf_sha256": sha256_file(args.pdf),
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
        "pdf_sheet_count": len({page["pdf_page"] for page in pages}),
        "printed_fragment_count": len(pages),
        "chunk_count": len(chunks),
        "vector_dimension": int(vectors.shape[1]),
        "chunk_size_characters": args.chunk_size,
        "overlap_characters": args.overlap,
        "repeated_margin_lines_removed": sorted(repeated_lines),
        "ingestion_seconds": round(elapsed, 3),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunks_sha256": sha256_file(index_chunks_path),
        "index_sha256": sha256_file(index_path),
    }
    with (settings.index_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
