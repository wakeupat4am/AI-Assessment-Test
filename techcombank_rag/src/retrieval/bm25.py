from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import math
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*%?|[^\W\d_]+(?:[-_][^\W\d_]+)*", re.UNICODE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lexical_tokens(
    text: str,
    *,
    name: str | None = None,
    unicode_form: str = "NFKC",
    lowercase: bool = True,
    include_bigrams: bool = True,
) -> list[str]:
    del name
    normalized = unicodedata.normalize(unicode_form, text)
    if lowercase:
        normalized = normalized.casefold()
    unigrams = TOKEN_RE.findall(normalized)
    if not include_bigrams:
        return unigrams
    return unigrams + [f"{left}::{right}" for left, right in zip(unigrams, unigrams[1:])]


def build_bm25_artifact(
    chunks_path: Path,
    output_path: Path,
    *,
    tokenizer: dict[str, Any],
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, Any]:
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    postings: dict[str, list[list[int]]] = {}
    document_lengths: list[int] = []
    for document_id, chunk in enumerate(chunks):
        counts = Counter(
            lexical_tokens(
                str(chunk.get("search_text") or chunk.get("text", "")), **tokenizer
            )
        )
        document_lengths.append(sum(counts.values()))
        for token, frequency in counts.items():
            postings.setdefault(token, []).append([document_id, frequency])
    payload = {
        "artifact_schema_version": 1,
        "algorithm": "okapi_bm25",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_chunks": chunks_path.name,
        "source_chunks_sha256": sha256_file(chunks_path),
        "document_count": len(chunks),
        "average_document_length": (
            sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
        ),
        "document_lengths": document_lengths,
        "tokenizer": tokenizer,
        "k1": float(k1),
        "b": float(b),
        "postings": postings,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    return {
        key: value
        for key, value in payload.items()
        if key not in {"document_lengths", "postings"}
    } | {
        "vocabulary_size": len(postings),
        "artifact_sha256": sha256_file(output_path),
    }


class BM25Retriever:
    def __init__(
        self,
        chunks_path: Path,
        artifact_path: Path,
        *,
        candidate_k: int = 20,
    ) -> None:
        self.chunks_path = Path(chunks_path)
        self.artifact_path = Path(artifact_path)
        self.chunks = [
            json.loads(line)
            for line in self.chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        with gzip.open(self.artifact_path, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact["source_chunks_sha256"] != sha256_file(self.chunks_path):
            raise ValueError("BM25 artifact does not match chunks.jsonl")
        if int(artifact["document_count"]) != len(self.chunks):
            raise ValueError("BM25 document count does not match chunks.jsonl")
        self.tokenizer = dict(artifact["tokenizer"])
        self.k1 = float(artifact["k1"])
        self.b = float(artifact["b"])
        self.average_document_length = float(artifact["average_document_length"])
        self.document_lengths = [int(value) for value in artifact["document_lengths"]]
        self.postings = artifact["postings"]
        self.document_count = len(self.chunks)
        self.candidate_k = int(candidate_k)
        self.last_trace: dict[str, Any] | None = None
        self.last_retrieval_calls: list[Any] = []
        self.last_router_call = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        started = time.perf_counter()
        requested_k = int(top_k or self.candidate_k)
        tokens = list(dict.fromkeys(lexical_tokens(query, **self.tokenizer)))
        scores: dict[int, float] = {}
        matched_tokens: list[str] = []
        for token in tokens:
            token_postings = self.postings.get(token)
            if not token_postings:
                continue
            matched_tokens.append(token)
            document_frequency = len(token_postings)
            idf = math.log(
                1.0
                + (self.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for document_id, term_frequency in token_postings:
                length = self.document_lengths[document_id]
                denominator = term_frequency + self.k1 * (
                    1.0
                    - self.b
                    + self.b * length / max(self.average_document_length, 1e-9)
                )
                scores[document_id] = scores.get(document_id, 0.0) + idf * (
                    term_frequency * (self.k1 + 1.0) / denominator
                )
        ranked = heapq.nlargest(
            min(requested_k, len(scores)), scores.items(), key=lambda item: (item[1], -item[0])
        )
        best = ranked[0][1] if ranked else 1.0
        results: list[dict[str, Any]] = []
        for rank, (document_id, raw_score) in enumerate(ranked, 1):
            row = dict(self.chunks[document_id])
            row.update(
                {
                    "score": float(raw_score / best),
                    "bm25_score": float(raw_score),
                    "bm25_rank": rank,
                    "retrieval_source": "bm25",
                }
            )
            results.append(row)
        self.last_trace = {
            "b4": {
                "stage": "bm25",
                "query": query,
                "query_token_count": len(tokens),
                "matched_token_count": len(matched_tokens),
                "matched_tokens": matched_tokens,
                "returned_candidates": len(results),
                "latency_seconds": round(time.perf_counter() - started, 6),
            }
        }
        return results
