from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_overrides(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    return {str(key): int(value) for key, value in values.items()}


def assign_printed_pages(
    fragments: list[dict[str, Any]], overrides: dict[str, int] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map PDF regions to printed pages using repeated detected anchors.

    Candidate page numbers are read only from top/bottom outer margins. The dominant
    offset between logical order and those anchors is then used for pages where a
    number is absent, such as chapter dividers. Overrides use keys like `5:left`.
    """
    overrides = overrides or {}
    offsets: Counter[int] = Counter()
    for fragment in fragments:
        logical_index = int(fragment["logical_index"])
        for candidate in fragment.get("printed_candidates", []):
            offsets[int(candidate) - logical_index] += 1
    if not offsets or offsets.most_common(1)[0][1] < 3:
        raise ValueError(
            "Could not infer a reliable printed-page offset. Supply --mapping-overrides."
        )

    dominant_offset, anchor_count = offsets.most_common(1)[0]
    mapped: list[dict[str, Any]] = []
    for fragment in fragments:
        record = dict(fragment)
        key = f"{record['pdf_page']}:{record['page_part']}"
        expected = int(record["logical_index"]) + dominant_offset
        candidates = [int(value) for value in record.get("printed_candidates", [])]
        if key in overrides:
            printed_page = overrides[key]
            method = "override"
        elif expected in candidates:
            printed_page = expected
            method = "detected"
        else:
            printed_page = expected
            method = "inferred_from_anchors"
        record["printed_page"] = printed_page
        record["mapping_method"] = method
        mapped.append(record)

    summary = {
        "strategy": "landscape spread split + outer-margin anchors + dominant offset",
        "dominant_offset": dominant_offset,
        "anchor_count": anchor_count,
        "fragment_count": len(mapped),
        "detected_count": sum(r["mapping_method"] == "detected" for r in mapped),
        "inferred_count": sum(
            r["mapping_method"] == "inferred_from_anchors" for r in mapped
        ),
        "override_count": sum(r["mapping_method"] == "override" for r in mapped),
        "records": [
            {
                "pdf_page_index": r["pdf_page_index"],
                "pdf_page": r["pdf_page"],
                "page_part": r["page_part"],
                "printed_page": r["printed_page"],
                "mapping_method": r["mapping_method"],
                "printed_candidates": r["printed_candidates"],
            }
            for r in mapped
        ],
    }
    return mapped, summary


def save_mapping(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
