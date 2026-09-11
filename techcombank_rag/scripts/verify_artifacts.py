#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the shipped vector-index artifact")
    parser.add_argument("--index-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.index_dir / "metadata.json"
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "chunks.jsonl": metadata["chunks_sha256"],
        "index.faiss": metadata["index_sha256"],
    }
    failures: list[str] = []
    for name, checksum in expected.items():
        path = args.index_dir / name
        actual = sha256(path) if path.exists() else "missing"
        if actual != checksum:
            failures.append(f"{name}: expected {checksum}, got {actual}")
    bm25_path = args.index_dir / "bm25.json.gz"
    bm25_metadata_path = args.index_dir / "bm25.metadata.json"
    if bm25_path.exists() or bm25_metadata_path.exists():
        if not bm25_path.exists() or not bm25_metadata_path.exists():
            failures.append("BM25 verification requires bm25.json.gz and bm25.metadata.json")
        else:
            with bm25_metadata_path.open(encoding="utf-8") as handle:
                bm25_metadata = json.load(handle)
            actual_bm25 = sha256(bm25_path)
            if actual_bm25 != bm25_metadata.get("artifact_sha256"):
                failures.append(
                    f"bm25.json.gz: expected {bm25_metadata.get('artifact_sha256')}, got {actual_bm25}"
                )
            if actual_bm25 and bm25_metadata.get("source_chunks_sha256") != expected["chunks.jsonl"]:
                failures.append("BM25 artifact was built from a different chunks.jsonl")
    if failures:
        raise SystemExit("Index verification failed:\n" + "\n".join(failures))
    print(
        f"Index OK: {metadata['chunk_count']} chunks, "
        f"{metadata['vector_dimension']} dimensions, "
        f"model={metadata['embedding_model']}"
    )
    if bm25_path.exists():
        print(f"BM25 OK: {bm25_metadata['document_count']} documents")


if __name__ == "__main__":
    main()
