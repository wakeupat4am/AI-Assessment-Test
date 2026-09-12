from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.retrieval.bm25 import BM25Retriever, build_bm25_artifact, lexical_tokens
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.finance_reranker import FinanceAwareReranker
from src.retrieval.hybrid import HybridRRFRetriever


def chunk(chunk_id: str, text: str, page: int, **extra):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "printed_page": page,
        "pdf_page": page,
        "granularity": extra.pop("granularity", "row"),
        **extra,
    }


def test_tokenizer_preserves_vietnamese_numbers_acronyms_and_bigrams() -> None:
    tokens = lexical_tokens("Tỷ lệ CASA 2025 là 40,4%")
    assert {"tỷ", "lệ", "casa", "2025", "40,4%", "tỷ::lệ"}.issubset(tokens)


def test_bm25_artifact_is_deterministic_and_rejects_wrong_chunks(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in [
                chunk("a", "CASA năm 2025 đạt 40,4%", 5),
                chunk("b", "Chiến lược ngân hàng số", 8, granularity="block"),
            ]
        ),
        encoding="utf-8",
    )
    artifact = tmp_path / "bm25.json.gz"
    tokenizer = {"unicode_form": "NFKC", "lowercase": True, "include_bigrams": True}
    metadata = build_bm25_artifact(chunks, artifact, tokenizer=tokenizer)
    assert metadata["document_count"] == 2
    retriever = BM25Retriever(chunks, artifact)
    assert retriever.retrieve("CASA 2025", 1)[0]["chunk_id"] == "a"
    chunks.write_text(chunks.read_text(encoding="utf-8") + json.dumps(chunk("c", "x", 9)) + "\n", encoding="utf-8")
    try:
        BM25Retriever(chunks, artifact)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched chunks must fail")


def test_bm25_indexes_search_text_without_overwriting_evidence(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            chunk(
                "cash",
                "OCR evidence: thu nhập tử dịch vụ",
                301,
                search_text="thu nhập từ hoạt động dịch vụ nhận được",
            ),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "bm25.json.gz"
    build_bm25_artifact(chunks, artifact, tokenizer={"include_bigrams": True})
    result = BM25Retriever(chunks, artifact).retrieve("dịch vụ nhận được", 1)[0]
    assert result["chunk_id"] == "cash"
    assert "tử" in result["text"]


class FakeRetriever:
    def __init__(self, rows):
        self.rows = rows
        self.last_trace = None

    def retrieve(self, query, top_k=None):
        del query
        return [dict(row) for row in self.rows[:top_k]]


def test_hybrid_rrf_rewards_overlap_and_preserves_metadata() -> None:
    dense = FakeRetriever([
        {**chunk("a", "A", 1), "score": 0.9},
        {**chunk("b", "B", 2), "score": 0.8},
    ])
    sparse = FakeRetriever([
        {**chunk("c", "C", 3), "score": 1.0},
        {**chunk("a", "A", 1), "score": 0.7},
    ])
    retriever = HybridRRFRetriever(dense, sparse, candidate_k=2)
    rows = retriever.retrieve("q", 3)
    assert rows[0]["chunk_id"] == "a"
    assert rows[0]["printed_page"] == 1
    assert {item["source"] for item in rows[0]["retrieval_sources"]} == {"dense", "bm25"}


FINANCE_CONFIG = {
    "base_weight": 0.2,
    "constraint_weight": 0.8,
    "year_weight": 0.3,
    "acronym_weight": 0.2,
    "metric_weight": 0.4,
    "number_weight": 0.0,
    "granularity_weight": 0.1,
    "financial_terms": ["casa"],
    "explanatory_markers": ["tại sao"],
}


def test_finance_reranker_promotes_metric_and_year_constraint() -> None:
    base = FakeRetriever([
        {**chunk("noise", "CASA tổng quan", 1), "score": 1.0, "fusion_score_normalized": 1.0},
        {**chunk("gold", "CASA năm 2024 đạt 39%", 2), "score": 0.8, "fusion_score_normalized": 0.8},
    ])
    rows = FinanceAwareReranker(base, FINANCE_CONFIG, candidate_k=2).retrieve("CASA năm 2024", 1)
    assert rows[0]["chunk_id"] == "gold"
    assert rows[0]["finance_constraint_components"]["year"] == 1.0


class FakeCrossEncoder:
    def predict(self, pairs, batch_size, show_progress_bar):
        del batch_size, show_progress_bar
        return np.asarray([0.1 if "noise" in text else 2.0 for _, text in pairs])


def test_cross_encoder_reranks_candidate_pool_and_returns_requested_k() -> None:
    base = FakeRetriever([
        {**chunk("noise", "noise", 1), "score": 1.0},
        {**chunk("gold", "gold evidence", 2), "score": 0.8},
    ])
    retriever = CrossEncoderReranker(
        base,
        model_name="fake",
        model=FakeCrossEncoder(),
        candidate_k=2,
        threads=1,
    )
    rows = retriever.retrieve("question", 1)
    assert [row["chunk_id"] for row in rows] == ["gold"]
    assert rows[0]["score"] == 1.0
    assert retriever.last_trace["b4"]["cross_encoder_candidates"] == 2
