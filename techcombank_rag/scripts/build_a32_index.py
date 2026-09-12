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

import faiss


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.ingestion.semantic_multirepr import route_subset
from src.retrieval.b4_factory import load_b4_config
from src.retrieval.bm25 import build_bm25_artifact
from src.retrieval.embedder import Embedder
from src.retrieval.metric_identity import enrich_metric_identity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build separate A3.2 metric-aware index")
    parser.add_argument(
        "--source-index",
        type=Path,
        default=ROOT / "data/index/a31_semantic_multirepr/all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/index/a32_metric_aware/all",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/b4e_metric_aware.json"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    settings = get_settings()
    if not settings.embedding_model:
        raise ValueError("EMBEDDING_MODEL is required")

    source_chunks_path = args.source_index / "chunks.jsonl"
    source_metadata_path = args.source_index / "metadata.json"
    source_chunks = [
        json.loads(line)
        for line in source_chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("embedding_model") != settings.embedding_model:
        raise ValueError("A3.1 embedding model and configured model must match")

    chunks = [enrich_metric_identity(row) for row in source_chunks if route_subset(row, "all")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = args.output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for row in chunks:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    started = time.perf_counter()
    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    vectors = embedder.encode_documents(
        [str(row["search_text"]) for row in chunks],
        batch_size=args.batch_size,
        show_progress=True,
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_path = args.output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    metadata = {
        "artifact_schema_version": 1,
        "experiment": "A3.2",
        "pipeline": "a3.1+ocr-search-normalization+metric-identity+dense-e5",
        "representation": "a3.2",
        "source_representation": "a3.1-frozen",
        "source_chunks_sha256": sha256(source_chunks_path),
        "source_index_sha256": sha256(args.source_index / "index.faiss"),
        "chunk_count": len(chunks),
        "vector_dimension": int(vectors.shape[1]),
        "embedding_model": settings.embedding_model,
        "embedding_device": embedder.device,
        "metric_identity_count": sum(bool(row.get("metric_key")) for row in chunks),
        "statement_type_counts": dict(
            sorted(Counter(str(row["statement_type"]) for row in chunks).items())
        ),
        "build_seconds": round(time.perf_counter() - started, 3),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunks_sha256": sha256(chunks_path),
        "index_sha256": sha256(index_path),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    config = load_b4_config(args.config)
    bm25_path = args.output_dir / "bm25.json.gz"
    bm25_metadata = build_bm25_artifact(
        chunks_path,
        bm25_path,
        tokenizer=config["tokenizer"],
        k1=float(config["bm25"]["k1"]),
        b=float(config["bm25"]["b"]),
    )
    (args.output_dir / "bm25.metadata.json").write_text(
        json.dumps(bm25_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**metadata, "bm25": bm25_metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
