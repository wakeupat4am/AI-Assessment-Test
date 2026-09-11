#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pymupdf as fitz

from src.config import get_settings
from src.ingestion.chunker import chunk_pages
from src.ingestion.page_mapper import assign_printed_pages, load_overrides, save_mapping
from src.ingestion.paddleocr_vl import (
    extract_markdown,
    json_safe,
    load_raw_records,
    representation_pages,
)
from src.ingestion.pdf_parser import _page_parts, extract_pdf_fragments


PIPELINE_NAME = "PaddleOCR-VL-1.6"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


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


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _raw_path(raw_dir: Path, fragment: dict[str, Any]) -> Path:
    return raw_dir / (
        f"fragment_{int(fragment['logical_index']):04d}_"
        f"pdf{int(fragment['pdf_page']):03d}_{fragment['page_part']}_"
        f"tr{int(fragment['printed_page']):03d}.json"
    )


def parse_pdf(
    pdf_path: Path,
    raw_dir: Path,
    mapping_overrides: Path | None,
    device: str,
    dpi: int,
    max_fragments: int | None,
    shard_index: int,
    num_shards: int,
    vl_backend: str | None,
    vl_server_url: str | None,
    max_attempts: int,
    force: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fragments = extract_pdf_fragments(pdf_path)
    mapped, mapping_summary = assign_printed_pages(
        fragments, load_overrides(mapping_overrides)
    )
    if max_fragments is not None:
        mapped = mapped[:max_fragments]
    if num_shards <= 0 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    mapped = [
        row
        for row in mapped
        if int(row["logical_index"]) % num_shards == shard_index
    ]
    raw_dir.mkdir(parents=True, exist_ok=True)
    pending = [row for row in mapped if force or not _raw_path(raw_dir, row).exists()]
    pipeline = None
    if pending:
        if vl_backend:
            if not vl_server_url:
                raise ValueError("--vl-server-url is required with --vl-backend")
            try:
                from paddleocr import PaddleOCRVL
            except ImportError as exc:
                raise RuntimeError(
                    "The paddleocr package is required for a VLM service backend."
                ) from exc
            pipeline = PaddleOCRVL(
                pipeline_version="v1.6",
                device=device,
                vl_rec_backend=vl_backend,
                vl_rec_server_url=vl_server_url,
            )
        else:
            try:
                from paddlex import create_pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "PaddleX is required only for ingestion. Follow "
                    "docs/experiments/A1_PADDLEOCR_VL.md; runtime graders use shipped indexes."
                ) from exc
            pipeline = create_pipeline(pipeline=PIPELINE_NAME, device=device)

    parse_started = time.perf_counter()
    completed = 0
    with fitz.open(pdf_path) as document, tempfile.TemporaryDirectory(
        prefix="tcb_paddleocr_vl_"
    ) as temp_dir:
        temp_root = Path(temp_dir)
        for fragment in mapped:
            output_path = _raw_path(raw_dir, fragment)
            if output_path.exists() and not force:
                completed += 1
                continue
            assert pipeline is not None
            page = document[int(fragment["pdf_page_index"])]
            clips = dict(_page_parts(page))
            clip = clips[str(fragment["page_part"])]
            image_path = temp_root / f"fragment_{int(fragment['logical_index']):04d}.png"
            scale = dpi / 72.0
            page.get_pixmap(
                matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False
            ).save(image_path)
            started = time.perf_counter()
            results = None
            for attempt in range(1, max_attempts + 1):
                try:
                    results = list(pipeline.predict(input=str(image_path)))
                    break
                except Exception as exc:
                    if attempt == max_attempts:
                        raise
                    print(
                        f"Retry {attempt}/{max_attempts - 1} for logical fragment "
                        f"{fragment['logical_index']}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    time.sleep(min(2**attempt, 10))
            assert results is not None
            if len(results) != 1:
                raise RuntimeError(
                    f"Expected one PaddleX result for {image_path}, got {len(results)}"
                )
            result = results[0]
            structured = getattr(result, "json", {})
            if callable(structured):
                structured = structured()
            record = {
                "raw_schema_version": 1,
                "raw_id": output_path.stem,
                "pipeline": PIPELINE_NAME,
                "logical_index": int(fragment["logical_index"]),
                "pdf_page_index": int(fragment["pdf_page_index"]),
                "pdf_page": int(fragment["pdf_page"]),
                "page_part": fragment["page_part"],
                "printed_page": int(fragment["printed_page"]),
                "mapping_method": fragment.get("mapping_method"),
                "source": fragment["source"],
                "render_dpi": dpi,
                "vl_backend": vl_backend or "paddle-native",
                "vl_server_url": vl_server_url,
                "ocr_seconds": round(time.perf_counter() - started, 3),
                "markdown": extract_markdown(result),
                "structured_result": json_safe(structured),
            }
            write_json(output_path, record)
            completed += 1
            print(
                f"[{completed}/{len(mapped)}] printed page {record['printed_page']} "
                f"in {record['ocr_seconds']:.3f}s",
                flush=True,
            )
    mapping_summary = dict(mapping_summary)
    mapping_summary["last_worker_fragment_count"] = len(mapped)
    mapping_summary["last_worker_shard"] = {
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    mapping_summary["parse_wall_seconds_this_run"] = round(
        time.perf_counter() - parse_started, 3
    )
    save_mapping(mapping_summary, raw_dir / "page_mapping.json")
    return load_raw_records(raw_dir), mapping_summary


def build_representation_index(
    *,
    records: list[dict[str, Any]],
    representation: str,
    index_dir: Path,
    processed_dir: Path,
    embedder: Any,
    embedding_model: str,
    source_pdf: Path,
    raw_dir: Path,
    mapping_summary: dict[str, Any],
    chunk_size: int,
    overlap: int,
    batch_size: int,
) -> dict[str, Any]:
    import faiss

    started = time.perf_counter()
    pages = representation_pages(records, representation)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise RuntimeError(f"No {representation} chunks were produced")
    vectors = embedder.encode_documents(
        [chunk["text"] for chunk in chunks],
        batch_size=batch_size,
        show_progress=True,
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    processed_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(processed_dir / "pages.jsonl", pages)
    write_jsonl(processed_dir / "chunks.jsonl", chunks)
    write_json(processed_dir / "page_mapping.json", mapping_summary)
    chunks_path = index_dir / "chunks.jsonl"
    index_path = index_dir / "index.faiss"
    write_jsonl(chunks_path, chunks)
    faiss.write_index(index, str(index_path))
    metadata = {
        "artifact_schema_version": 1,
        "experiment": "A1a" if representation == "plain" else "A1b",
        "pipeline": f"{PIPELINE_NAME}-{representation}-fixed-chunk+dense-e5-flatip",
        "document_parser": PIPELINE_NAME,
        "representation": representation,
        "source_pdf": source_pdf.name,
        "source_pdf_sha256": sha256_file(source_pdf),
        "raw_ocr_dir": str(raw_dir.relative_to(ROOT)),
        "raw_fragment_count": len(records),
        "raw_manifest_sha256": sha256_file(raw_dir / "manifest.json"),
        "embedding_model": embedding_model,
        "embedding_device": embedder.device,
        "pdf_sheet_count": len({page["pdf_page"] for page in pages}),
        "printed_fragment_count": len(pages),
        "chunk_count": len(chunks),
        "vector_dimension": int(vectors.shape[1]),
        "chunk_size_characters": chunk_size,
        "overlap_characters": overlap,
        "index_build_seconds": round(time.perf_counter() - started, 3),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunks_sha256": sha256_file(chunks_path),
        "index_sha256": sha256_file(index_path),
    }
    write_json(index_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build controlled A1a/A1b PaddleOCR-VL 1.6 indexes"
    )
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument(
        "--raw-dir", type=Path, default=ROOT / "data/processed/paddleocr_vl_1_6/raw"
    )
    parser.add_argument(
        "--plain-processed-dir",
        type=Path,
        default=ROOT / "data/processed/a1a_paddleocr_vl_1_6_plain_fixed",
    )
    parser.add_argument(
        "--markdown-processed-dir",
        type=Path,
        default=ROOT / "data/processed/a1b_paddleocr_vl_1_6_markdown_fixed",
    )
    parser.add_argument(
        "--plain-index-dir",
        type=Path,
        default=ROOT / "data/index/a1a_paddleocr_vl_1_6_plain_fixed",
    )
    parser.add_argument(
        "--markdown-index-dir",
        type=Path,
        default=ROOT / "data/index/a1b_paddleocr_vl_1_6_markdown_fixed",
    )
    parser.add_argument("--mapping-overrides", type=Path)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--chunk-size", type=int, default=3200)
    parser.add_argument("--overlap", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-fragments", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--vl-backend",
        choices=("vllm-server", "sglang-server", "fastdeploy-server", "llama-cpp-server"),
    )
    parser.add_argument("--vl-server-url")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--skip-parse", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if not settings.embedding_model and not args.parse_only:
        raise ValueError("EMBEDDING_MODEL is required to build the two indexes")
    if args.skip_parse:
        records = load_raw_records(args.raw_dir)
        mapping_path = args.raw_dir / "page_mapping.json"
        if not records or not mapping_path.exists():
            raise FileNotFoundError("Raw OCR checkpoint is incomplete; omit --skip-parse")
        with mapping_path.open(encoding="utf-8") as handle:
            mapping_summary = json.load(handle)
        previous_manifest_path = args.raw_dir / "manifest.json"
        if previous_manifest_path.exists():
            with previous_manifest_path.open(encoding="utf-8") as handle:
                previous_manifest = json.load(handle)
        else:
            previous_manifest = {}
    else:
        previous_manifest = {}
        records, mapping_summary = parse_pdf(
            args.pdf,
            args.raw_dir,
            args.mapping_overrides,
            args.device,
            args.dpi,
            args.max_fragments,
            args.shard_index,
            args.num_shards,
            args.vl_backend,
            args.vl_server_url,
            args.max_attempts,
            args.force,
        )

    logical_indices = [int(record["logical_index"]) for record in records]
    if len(set(logical_indices)) != len(logical_indices):
        raise ValueError("Raw OCR checkpoint contains duplicate logical_index values")
    expected_fragments = int(mapping_summary.get("fragment_count", len(records)))
    if not args.parse_only and len(records) != expected_fragments:
        raise RuntimeError(
            f"Refusing to build a partial index: found {len(records)} of "
            f"{expected_fragments} expected fragments"
        )

    observed_backends = sorted(
        {str(record.get("vl_backend", "unknown")) for record in records}
    )
    observed_server_urls = sorted(
        {
            str(record["vl_server_url"])
            for record in records
            if record.get("vl_server_url")
        }
    )
    current_package_versions = {
        name: _package_version(name)
        for name in ("paddlepaddle-gpu", "paddlepaddle", "paddlex", "paddleocr")
    }
    package_versions = current_package_versions
    if args.skip_parse and not any(current_package_versions.values()):
        package_versions = previous_manifest.get(
            "package_versions", current_package_versions
        )
    manifest = {
        "raw_schema_version": 1,
        "pipeline": PIPELINE_NAME,
        "controlled_experiment": {
            "A1a": "PaddleOCR-VL-1.6 -> deterministic plain text -> fixed chunk",
            "A1b": "PaddleOCR-VL-1.6 -> canonical Markdown -> fixed chunk",
        },
        "source_pdf": args.pdf.name,
        "source_pdf_sha256": sha256_file(args.pdf),
        "fragment_count": len(records),
        "expected_fragment_count": expected_fragments,
        "render_dpi": args.dpi,
        "device": previous_manifest.get("device", args.device)
        if args.skip_parse
        else args.device,
        "vl_backends_observed": observed_backends,
        "vl_server_urls_observed": observed_server_urls,
        "python": platform.python_version(),
        "package_versions": package_versions,
        "total_ocr_seconds": round(
            sum(float(record.get("ocr_seconds", 0)) for record in records), 3
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.raw_dir / "manifest.json", manifest)
    if args.parse_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    from src.retrieval.embedder import Embedder

    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    outputs = []
    for representation, index_dir, processed_dir in (
        ("plain", args.plain_index_dir, args.plain_processed_dir),
        ("markdown", args.markdown_index_dir, args.markdown_processed_dir),
    ):
        outputs.append(
            build_representation_index(
                records=records,
                representation=representation,
                index_dir=index_dir,
                processed_dir=processed_dir,
                embedder=embedder,
                embedding_model=settings.embedding_model,
                source_pdf=args.pdf,
                raw_dir=args.raw_dir,
                mapping_summary=mapping_summary,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
            )
        )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
