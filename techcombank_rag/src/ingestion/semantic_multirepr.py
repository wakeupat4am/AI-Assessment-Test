from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from src.ingestion.paddleocr_vl import markdown_to_plain_text


HEADING_RE = re.compile(r"^\s{0,3}#{1,4}\s+(.+?)\s*$", re.MULTILINE)
GENERIC_HEADING_RE = re.compile(
    r"^(?:điểm nhấn(?:\s+\d{4})?|tổng quan|kết quả(?:\s+\d{4})?|"
    r"tiếp theo|thành tựu|[“”\"']+)$",
    re.IGNORECASE,
)
VALUE_START_RE = re.compile(
    r"^(?:[▲▼+\-]\s*)?(?:top\s*)?(?:~\s*)?\(?\d[\d.,\s]*(?:%|x)?\)?",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"(?<!\w)(?:~\s*)?\(?\d[\d.,]*(?:%|x)?\)?", re.IGNORECASE)
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
UNIT_RE = re.compile(
    r"\b(?:nghìn tỷ đồng|tỷ đồng|triệu đồng|triệu khách hàng|triệu người|"
    r"nghìn người|phần trăm|điểm cơ bản|khách hàng|giao dịch)\b|%",
    re.IGNORECASE,
)
DETAIL_RE = re.compile(
    r"^(?:tăng|giảm|tăng trưởng|trung bình|tại\b|so với|n/n|yoy|cagr|"
    r"dẫn đầu|thuộc nhóm|được thực hiện|chất lượng|triển vọng|~)",
    re.IGNORECASE,
)
LABEL_EXCLUSION_RE = re.compile(
    r"^(?:nghìn tỷ đồng|tỷ đồng|triệu đồng|triệu khách hàng|triệu người|"
    r"tăng|giảm|tăng trưởng|trung bình|tại\b|dẫn đầu|thuộc nhóm|"
    r"được thực hiện|chất lượng|n/n|yoy|cagr)",
    re.IGNORECASE,
)
IMAGE_HTML_RE = re.compile(r"<[^>]+>")
ENTITY_HEADING_RE = re.compile(r"^(?:công ty|khối)\b", re.IGNORECASE)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" |\n\t")


def _normalized(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def _blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = (
        record.get("structured_result", {})
        .get("res", {})
        .get("parsing_res_list", [])
    )
    return value if isinstance(value, list) else []


def _bbox(block: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = block.get("block_bbox") or block.get("bbox") or block.get("block_position")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _markdown_heading(record: dict[str, Any]) -> str:
    match = HEADING_RE.search(str(record.get("markdown", "")))
    return _clean(markdown_to_plain_text(match.group(1))) if match else ""


def _is_generic_heading(heading: str) -> bool:
    return not heading or bool(GENERIC_HEADING_RE.fullmatch(_clean(heading)))


def infer_section_paths(records: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Infer a conservative document heading path across consecutive pages."""
    current_section = ""
    paths: dict[str, list[str]] = {}
    for record in sorted(records, key=lambda item: int(item["logical_index"])):
        local = _markdown_heading(record)
        if local and not _is_generic_heading(local):
            current_section = local
        path: list[str] = []
        if current_section:
            path.append(current_section)
        if local and local != current_section:
            path.append(local)
        paths[str(record["raw_id"])] = path
    return paths


def _is_value(text: str) -> bool:
    value = _clean(text)
    return bool(value and len(value) <= 100 and VALUE_START_RE.match(value))


def _is_label(text: str) -> bool:
    value = _clean(text)
    if not value or len(value) > 90 or len(value.split()) > 14:
        return False
    if (
        _is_value(value)
        or NUMBER_RE.search(value)
        or LABEL_EXCLUSION_RE.match(value)
        or value.endswith((".", ";", ":"))
    ):
        return False
    return bool(re.search(r"[A-Za-zÀ-ỹ]", value))


def _same_column(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    min_width = max(1.0, min(left[2] - left[0], right[2] - right[0]))
    left_center = (left[0] + left[2]) / 2
    right_center = (right[0] + right[2]) / 2
    return overlap / min_width >= 0.25 or abs(left_center - right_center) <= 100


def _coordinate_metric_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[tuple[int, str, tuple[float, float, float, float]]] = []
    for index, block in enumerate(_blocks(record)):
        if str(block.get("block_label", "")) in {"header", "footer", "number", "image"}:
            continue
        text = _clean(markdown_to_plain_text(str(block.get("block_content", ""))))
        box = _bbox(block)
        if text and box:
            blocks.append((index, text, box))
    labels = [item for item in blocks if _is_label(item[1])]
    values = [item for item in blocks if _is_value(item[1])]
    if not labels or not values:
        return []

    output: list[dict[str, Any]] = []
    for _, value, value_box in values:
        eligible = [
            item
            for item in labels
            if 0 <= value_box[1] - item[2][3] <= 220 and _same_column(item[2], value_box)
        ]
        if not eligible:
            continue
        label_item = min(
            eligible,
            key=lambda item: (
                value_box[1] - item[2][3],
                abs((item[2][0] + item[2][2]) - (value_box[0] + value_box[2])),
            ),
        )
        details: list[str] = []
        for _, text, box in blocks:
            gap = box[1] - value_box[3]
            if 0 <= gap <= 230 and _same_column(box, value_box):
                if UNIT_RE.search(text) or DETAIL_RE.match(text):
                    details.append(text)
        output.append(
            {
                "metric": label_item[1],
                "value": value,
                "details": details[:3],
                "extraction_method": "layout_nearest_label",
                "extraction_confidence": 0.92,
            }
        )
    return output


def _inline_metric_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for block in _blocks(record):
        plain = markdown_to_plain_text(str(block.get("block_content", "")))
        lines = [_clean(line) for line in plain.splitlines() if _clean(line)]
        if len(lines) >= 2 and _is_label(lines[0]) and any(_is_value(line) for line in lines[1:]):
            first_value = next(index for index, line in enumerate(lines[1:], 1) if _is_value(line))
            output.append(
                {
                    "metric": lines[0],
                    "value": lines[first_value],
                    "details": lines[first_value + 1 : first_value + 3],
                    "extraction_method": "structured_multiline",
                    "extraction_confidence": 0.84,
                }
            )
    return output


def _markdown_metric_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    markdown = IMAGE_HTML_RE.sub("", str(record.get("markdown", "")))
    lines = []
    for raw in markdown.splitlines():
        raw = re.sub(r"^\s{0,3}#{1,6}\s+", "", raw)
        value = _clean(markdown_to_plain_text(raw))
        if value and value not in {"___", "---"}:
            lines.append(value)
    output: list[dict[str, Any]] = []
    for index, line in enumerate(lines[:-1]):
        if not _is_label(line) or not _is_value(lines[index + 1]):
            continue
        details: list[str] = []
        for candidate in lines[index + 2 : index + 5]:
            if _is_label(candidate) and not (UNIT_RE.search(candidate) or DETAIL_RE.match(candidate)):
                break
            if UNIT_RE.search(candidate) or DETAIL_RE.match(candidate) or _is_value(candidate):
                details.append(candidate)
        output.append(
            {
                "metric": line,
                "value": lines[index + 1],
                "details": details[:3],
                "extraction_method": "markdown_sequence_fallback",
                "extraction_confidence": 0.68,
            }
        )
    return output


def extract_metric_values(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract generic KPI label/value groups without a metric-name allowlist."""
    priority = {
        "layout_nearest_label": 3,
        "structured_multiline": 2,
        "markdown_sequence_fallback": 1,
    }
    chosen: dict[str, dict[str, Any]] = {}
    candidates = [
        *_coordinate_metric_candidates(record),
        *_inline_metric_candidates(record),
        *_markdown_metric_candidates(record),
    ]
    for candidate in candidates:
        metric_key = _normalized(str(candidate["metric"]))
        if not metric_key:
            continue
        existing = chosen.get(metric_key)
        if existing is None or priority[str(candidate["extraction_method"])] > priority[
            str(existing["extraction_method"])
        ]:
            details = [_clean(str(item)) for item in candidate.get("details", []) if _clean(str(item))]
            joined = " ".join([str(candidate["value"]), *details])
            candidate = {
                **candidate,
                "details": details,
                "numbers": NUMBER_RE.findall(joined),
                "years": list(dict.fromkeys(YEAR_RE.findall(joined))),
                "unit": (UNIT_RE.search(joined).group(0) if UNIT_RE.search(joined) else None),
            }
            chosen[metric_key] = candidate
    return list(chosen.values())


def _context_text(path: list[str]) -> str:
    return " > ".join(item for item in path if item)


def _resolved_entity(path: list[str], document_entity: str) -> str:
    """Resolve document scope without treating every section title as an entity."""
    section = path[0] if path else ""
    if section and ENTITY_HEADING_RE.match(section):
        if section.casefold().startswith("khối") and document_entity:
            return f"{document_entity} — {section}"
        return section
    return document_entity


def semantic_multirepresentation_chunks(
    a3_chunks: Iterable[dict[str, Any]],
    records: Iterable[dict[str, Any]],
    *,
    parent_character_limit: int = 2400,
    document_entity: str = "",
    reporting_year: int | None = None,
) -> list[dict[str, Any]]:
    """Create A3.1 chunks from A3 plus raw OCR metadata, without rerunning OCR."""
    source_chunks = [dict(row) for row in a3_chunks]
    record_list = list(records)
    records_by_id = {str(row["raw_id"]): row for row in record_list}
    paths = infer_section_paths(record_list)
    by_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in source_chunks:
        by_raw[str(chunk["ocr_raw_id"])].append(chunk)

    parent_texts: dict[str, str] = {}
    for raw_id, rows in by_raw.items():
        blocks = [row for row in rows if row.get("granularity") == "block"]
        pages = [row for row in rows if row.get("granularity") == "page"]
        parent_parts = [str(row["text"]) for row in (blocks or pages)]
        parent_texts[raw_id] = "\n\n".join(parent_parts)[:parent_character_limit].rstrip()

    output: list[dict[str, Any]] = []
    for chunk in source_chunks:
        item = dict(chunk)
        raw_id = str(item["ocr_raw_id"])
        path = paths.get(raw_id, [])
        context = _context_text(path)
        entity = _resolved_entity(path, document_entity)
        original_text = str(item["text"])
        scope_lines = [
            f"Thực thể: {entity}" if entity else "",
            f"Kỳ báo cáo: {reporting_year}" if reporting_year else "",
            f"Ngữ cảnh tài liệu: {context}" if context else "",
        ]
        scope = "\n".join(line for line in scope_lines if line)
        if scope:
            item["text"] = f"{scope}\n{original_text}"
        item["chunk_id"] = re.sub(r"^a3_", "a31_", str(item["chunk_id"]))
        item["representation"] = "a3.1"
        item["original_representation"] = "a3"
        item["section_path"] = path
        item["section_heading"] = path[0] if path else ""
        item["document_entity"] = document_entity
        item["resolved_entity"] = entity
        item["reporting_year"] = reporting_year
        item["local_heading"] = _markdown_heading(records_by_id.get(raw_id, {}))
        item["heading_inherited"] = bool(
            len(path) > 1 and path[0] != item["local_heading"]
        )
        item["source_id"] = raw_id
        item["parent_id"] = f"a31_parent_{raw_id}"
        if item.get("granularity") == "row":
            item["parent_text"] = parent_texts.get(raw_id, "")
        output.append(item)

    for record in record_list:
        raw_id = str(record["raw_id"])
        path = paths.get(raw_id, [])
        context = _context_text(path)
        entity = _resolved_entity(path, document_entity)
        for sequence, metric in enumerate(extract_metric_values(record), 1):
            detail_text = " | ".join(metric.get("details", []))
            lines = [
                f"Thực thể: {entity}" if entity else "",
                f"Kỳ báo cáo: {reporting_year}" if reporting_year else "",
                f"Ngữ cảnh tài liệu: {context}" if context else "",
                f"Chỉ tiêu: {metric['metric']}",
                f"Giá trị: {metric['value']}",
                f"Chi tiết: {detail_text}" if detail_text else "",
            ]
            text = "\n".join(line for line in lines if line)
            output.append(
                {
                    "chunk_id": f"a31_{raw_id}_metric_{sequence:04d}",
                    "text": text,
                    "logical_index": int(record["logical_index"]),
                    "pdf_page_index": int(record["pdf_page_index"]),
                    "pdf_page": int(record["pdf_page"]),
                    "page_part": record["page_part"],
                    "printed_page": int(record["printed_page"]),
                    "mapping_method": record.get("mapping_method"),
                    "source": record["source"],
                    "ocr_raw_id": raw_id,
                    "parser": "PaddleOCR-VL-1.6+A3.1-semantic",
                    "representation": "a3.1",
                    "original_representation": None,
                    "granularity": "metric",
                    "block_type": "metric_value",
                    "heading": context,
                    "section_path": path,
                    "section_heading": path[0] if path else "",
                    "document_entity": document_entity,
                    "resolved_entity": entity,
                    "reporting_year": reporting_year,
                    "local_heading": _markdown_heading(record),
                    "heading_inherited": bool(len(path) > 1),
                    "source_id": raw_id,
                    "parent_id": f"a31_parent_{raw_id}",
                    "parent_text": parent_texts.get(raw_id, ""),
                    "semantic_fields": {
                        "entity": entity or None,
                        "metric": metric["metric"],
                        "value": metric["value"],
                        "details": metric.get("details", []),
                        "numbers": metric.get("numbers", []),
                        "years": metric.get("years", []),
                        "unit": metric.get("unit"),
                    },
                    "extraction_method": metric["extraction_method"],
                    "extraction_confidence": metric["extraction_confidence"],
                }
            )

    ids = [str(item["chunk_id"]) for item in output]
    if len(ids) != len(set(ids)):
        raise ValueError("A3.1 generated duplicate chunk IDs")
    return output


def route_subset(chunk: dict[str, Any], route: str) -> bool:
    granularity = str(chunk.get("granularity", ""))
    block_type = str(chunk.get("block_type", ""))
    if route == "all":
        return True
    if route == "structured":
        return granularity in {"metric", "row"} or block_type == "table"
    if route == "narrative":
        return granularity in {"block", "page"} and block_type in {
            "narrative",
            "fallback_page",
            "page_signature",
        }
    raise ValueError(f"Unknown A3.1 route subset: {route}")
