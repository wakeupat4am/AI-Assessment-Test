from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from src.retrieval.metric_identity import normalize_search_text, normalized_terms


YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
ROW_RE = re.compile(
    r"(?:Cột\s*1|Chỉ tiêu)\s*:\s*(?P<label>[^|\n]+)\s*\|\s*(?P<body>.*?)"
    r"(?=\n(?:Cột\s*1|Chỉ tiêu)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
VALUE_RE = re.compile(
    r"(?P<year>(?:19|20)\d{2})(?:\\n|\s)*"
    r"(?P<unit>[^:|\n]{1,40})\s*:\s*"
    r"(?P<value>\(?-?[\d.]+(?:,\d+)?\)?|-)"
)
COMPARISON_RE = re.compile(
    r"\b(?:so\s+với|tăng|giảm|thay\s+đổi|chênh\s+lệch|bao\s+nhiêu\s+phần\s+trăm)\b",
    re.IGNORECASE,
)
IGNORED_LABEL_TERMS = {
    "của",
    "techcombank",
    "năm",
    "là",
    "bao",
    "nhiêu",
    "tại",
    "ngày",
}


@dataclass(frozen=True)
class MetricAnswer:
    answer: str
    page: int
    label: str
    values: dict[str, str]
    calculation: dict[str, str] | None = None
    source_kind: str = "metric_row"


def _decimal(value: str) -> Decimal | None:
    if value == "-":
        return None
    negative = value.startswith("(") and value.endswith(")")
    raw = value.strip("()")
    # Annual-report tables use dots as thousands separators and commas as decimals.
    normalized = raw.replace(".", "").replace(",", ".")
    try:
        number = Decimal(normalized)
    except InvalidOperation:
        return None
    return -number if negative else number


def _format_vi(number: Decimal, decimals: int = 0) -> str:
    quantizer = Decimal(1).scaleb(-decimals)
    rounded = number.quantize(quantizer, rounding=ROUND_HALF_UP)
    rendered = f"{abs(rounded):,.{decimals}f}"
    rendered = rendered.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"-{rendered}" if rounded < 0 else rendered


def _label_coverage(query: str, label: str) -> float:
    query_terms = set(normalized_terms(query)) - IGNORED_LABEL_TERMS
    label_terms = set(normalized_terms(label)) - IGNORED_LABEL_TERMS
    return len(query_terms & label_terms) / len(label_terms) if label_terms else 0.0


def _rows(chunk: dict[str, Any]) -> list[tuple[str, dict[str, tuple[str, str]]]]:
    text = str(chunk.get("text", ""))
    parsed = []
    for match in ROW_RE.finditer(text):
        label = normalize_search_text(match.group("label"))
        values: dict[str, tuple[str, str]] = {}
        for value_match in VALUE_RE.finditer(match.group("body")):
            values[value_match.group("year")] = (
                value_match.group("value"),
                normalize_search_text(value_match.group("unit")).casefold(),
            )
        if values:
            parsed.append((label, values))
    return parsed


def _answer_semantic_metric(
    query: str,
    chunks: list[dict[str, Any]],
    requested_years: list[str],
    minimum_label_coverage: float,
) -> MetricAnswer | None:
    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for chunk in chunks:
        semantic = chunk.get("semantic_fields") or {}
        label = normalize_search_text(str(semantic.get("metric", "")))
        value = normalize_search_text(str(semantic.get("value", "")))
        if not label or not value:
            continue
        coverage = _label_coverage(query, label)
        if coverage < minimum_label_coverage:
            continue
        reporting_year = str(chunk.get("reporting_year", ""))
        searchable = normalize_search_text(str(chunk.get("text", "")))
        if requested_years and not all(
            year == reporting_year or year in searchable for year in requested_years
        ):
            continue
        candidate = (coverage, chunk, semantic)
        if best is None or coverage > best[0]:
            best = candidate
    if best is None:
        return None

    _, chunk, semantic = best
    label = normalize_search_text(str(semantic["metric"]))
    value = normalize_search_text(str(semantic["value"]))
    unit = normalize_search_text(str(semantic.get("unit") or ""))
    if unit and unit.casefold() not in value.casefold():
        value = f"{value} {unit.casefold()}"
    details = []
    for raw_detail in semantic.get("details") or []:
        detail = normalize_search_text(str(raw_detail)).lstrip("▲△ ")
        if not detail or detail.casefold() == unit.casefold():
            continue
        if detail not in details:
            details.append(detail)
    year = str(chunk.get("reporting_year") or (requested_years[-1] if requested_years else ""))
    year_phrase = f" năm {year}" if year else ""
    detail_phrase = ""
    if details:
        normalized_details = [item[:1].lower() + item[1:] for item in details]
        detail_phrase = ", " + "; ".join(normalized_details)
    page = int(chunk["printed_page"])
    answer = f"{label}{year_phrase} là {value}{detail_phrase} [tr. {page}]."
    values = {year: value} if year else {"value": value}
    return MetricAnswer(answer, page, label, values, source_kind="semantic_metric")


def answer_metric_query(
    query: str, chunks: list[dict[str, Any]], *, minimum_label_coverage: float = 0.72
) -> MetricAnswer | None:
    """Answer only an unambiguous metric row; otherwise defer to the LLM.

    This is deliberately conservative: all requested years must occur on the same
    source row, the label must substantially match the query, and units must agree.
    """
    requested_years = list(dict.fromkeys(YEAR_RE.findall(query)))
    if not requested_years:
        return None

    best: tuple[float, dict[str, Any], str, dict[str, tuple[str, str]]] | None = None
    for chunk in chunks:
        if str(chunk.get("representation", "")) != "a3.2":
            continue
        if chunk.get("metric_constraint_pass") is False:
            continue
        for label, values in _rows(chunk):
            if not all(year in values for year in requested_years):
                continue
            coverage = _label_coverage(query, label)
            if coverage < minimum_label_coverage:
                continue
            candidate = (coverage, chunk, label, values)
            if best is None or coverage > best[0]:
                best = candidate
    if best is None:
        return _answer_semantic_metric(
            query, chunks, requested_years, minimum_label_coverage
        )

    _, chunk, label, values = best
    selected = {year: values[year][0] for year in requested_years}
    units = {values[year][1] for year in requested_years}
    if len(units) != 1:
        return None
    unit = next(iter(units))
    page = int(chunk["printed_page"])

    if len(requested_years) == 1 and not COMPARISON_RE.search(query):
        year = requested_years[0]
        answer = f"{label} năm {year} là {selected[year]} {unit} [tr. {page}]."
        return MetricAnswer(answer, page, label, selected)

    if len(requested_years) != 2 or not COMPARISON_RE.search(query):
        return None
    newer_year, older_year = requested_years
    newer = _decimal(selected[newer_year])
    older = _decimal(selected[older_year])
    if newer is None or older is None or older == 0:
        return None
    delta = newer - older
    percentage = delta / abs(older) * Decimal(100)
    direction = "tăng" if delta >= 0 else "giảm"
    answer = (
        f"{label} năm {newer_year} {direction} {_format_vi(abs(delta))} {unit}, "
        f"tương đương khoảng {_format_vi(abs(percentage), 2)}% so với năm "
        f"{older_year} [tr. {page}]."
    )
    return MetricAnswer(
        answer,
        page,
        label,
        selected,
        {
            "newer_year": newer_year,
            "older_year": older_year,
            "delta": str(delta),
            "percentage": str(percentage),
            "unit": unit,
        },
    )
