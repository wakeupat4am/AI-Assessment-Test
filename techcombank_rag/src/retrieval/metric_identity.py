from __future__ import annotations

import re
import unicodedata
from typing import Any


OCR_TOKEN_CORRECTIONS = {
    "tử": "từ",
    "đông": "đồng",
    "hoặt": "hoạt",
    "tiên": "tiền",
    "chúng": "chứng",
    "dựng": "dụng",
}
QUALIFIER_PHRASES = (
    "nhận được",
    "đã trả",
    "thuần",
    "hợp nhất",
    "riêng lẻ",
    "cuối kỳ",
    "đầu kỳ",
    "bình quân",
    "trước thuế",
    "sau thuế",
)
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
ROW_LABEL_RE = re.compile(r"(?:Cột\s*1|Chỉ tiêu)\s*:\s*([^|\n]+)", re.IGNORECASE)


def normalize_search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    for wrong, correct in OCR_TOKEN_CORRECTIONS.items():
        normalized = re.sub(rf"\b{re.escape(wrong)}\b", correct, normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


def normalized_terms(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+|\d+(?:[.,]\d+)*", normalize_search_text(text).casefold())


def infer_statement_type(text: str) -> str:
    value = normalize_search_text(text).casefold()
    if "lưu chuyển tiền tệ" in value:
        return "cash_flow"
    if "kết quả hoạt động kinh doanh" in value or "báo cáo kết quả" in value:
        return "income_statement"
    if "cân đối kế toán" in value or "tình hình tài chính" in value:
        return "balance_sheet"
    if "thuyết minh báo cáo tài chính" in value or "thuyết minh" in value:
        return "financial_note"
    if "kết quả hoạt động" in value:
        return "performance_review"
    return "unspecified"


def extract_metric_label(chunk: dict[str, Any]) -> str:
    semantic = chunk.get("semantic_fields") or {}
    if semantic.get("metric"):
        return normalize_search_text(str(semantic["metric"]))
    text = str(chunk.get("retrieval_text") or chunk.get("text", ""))
    matches = ROW_LABEL_RE.findall(text)
    for label in matches:
        clean = normalize_search_text(label)
        if clean and not clean.isupper():
            return clean
    return normalize_search_text(matches[-1]) if matches else ""


def extract_qualifiers(text: str) -> list[str]:
    value = normalize_search_text(text).casefold()
    return [phrase for phrase in QUALIFIER_PHRASES if phrase in value]


def enrich_metric_identity(chunk: dict[str, Any]) -> dict[str, Any]:
    item = dict(chunk)
    original = str(item.get("text", ""))
    heading = " ".join(
        str(item.get(key, ""))
        for key in ("section_heading", "local_heading", "heading")
    )
    label = extract_metric_label(item)
    statement_type = infer_statement_type(f"{heading} {original}")
    qualifiers = extract_qualifiers(label)
    normalized_label = " ".join(normalized_terms(label))
    metric_key = f"{statement_type}::{normalized_label}" if normalized_label else ""
    years = list(dict.fromkeys(YEAR_RE.findall(original)))
    prefix = "\n".join(
        part
        for part in (
            f"Loại báo cáo: {statement_type}",
            f"Nhãn chỉ tiêu chuẩn hóa: {label}" if label else "",
            f"Từ khóa phân biệt: {', '.join(qualifiers)}" if qualifiers else "",
        )
        if part
    )
    item.update(
        {
            "representation": "a3.2",
            "original_representation": item.get("representation"),
            "metric_label": label,
            "metric_key": metric_key,
            "statement_type": statement_type,
            "metric_qualifiers": qualifiers,
            "metric_years": years,
            "search_text": f"{prefix}\n{normalize_search_text(original)}".strip(),
        }
    )
    item["chunk_id"] = re.sub(r"^a31_", "a32_", str(item["chunk_id"]))
    item["parent_id"] = re.sub(r"^a31_", "a32_", str(item.get("parent_id", "")))
    return item


def query_constraints(query: str) -> dict[str, Any]:
    normalized = normalize_search_text(query)
    return {
        "normalized": normalized,
        "terms": normalized_terms(normalized),
        "qualifiers": extract_qualifiers(normalized),
        "years": list(dict.fromkeys(YEAR_RE.findall(normalized))),
        "statement_type": infer_statement_type(normalized),
    }


def candidate_matches_constraints(query: str, chunk: dict[str, Any]) -> bool:
    constraints = query_constraints(query)
    searchable = str(chunk.get("search_text") or chunk.get("text", ""))
    normalized = normalize_search_text(searchable).casefold()
    if constraints["qualifiers"] and not all(
        qualifier in normalized for qualifier in constraints["qualifiers"]
    ):
        return False
    requested_type = constraints["statement_type"]
    candidate_type = str(chunk.get("statement_type") or infer_statement_type(searchable))
    if requested_type != "unspecified" and candidate_type not in {
        requested_type,
        "unspecified",
    }:
        return False
    return True
