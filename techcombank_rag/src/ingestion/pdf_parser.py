from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf as fitz


NUMBER_RE = re.compile(r"^\s*(\d{1,3})\s*$")


def _page_parts(page: fitz.Page) -> list[tuple[str, fitz.Rect]]:
    """Split landscape report spreads into left/right printed pages."""
    rect = page.rect
    if rect.width >= rect.height * 1.2:
        middle = rect.x0 + rect.width / 2
        return [
            ("left", fitz.Rect(rect.x0, rect.y0, middle, rect.y1)),
            ("right", fitz.Rect(middle, rect.y0, rect.x1, rect.y1)),
        ]
    return [("full", rect)]


def _printed_candidates(page: fitz.Page, clip: fitz.Rect) -> list[int]:
    candidates: list[int] = []
    for word in page.get_text("words", clip=clip, sort=True):
        x0, y0, _x1, _y1, token = word[:5]
        match = NUMBER_RE.match(str(token))
        in_vertical_margin = y0 <= page.rect.height * 0.14 or y0 >= page.rect.height * 0.86
        if not match or not in_vertical_margin:
            continue
        local_x = x0 - clip.x0
        if local_x <= clip.width * 0.20 or local_x >= clip.width * 0.80:
            candidates.append(int(match.group(1)))
    return sorted(set(candidates))


def extract_pdf_fragments(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract one record per physical printed-page region.

    The Techcombank PDF uses one landscape PDF sheet for two printed pages. Both
    the 1-based PDF sheet and zero-based PDF index are retained for provenance.
    """
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    fragments: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as document:
        for pdf_index, page in enumerate(document):
            for page_part, clip in _page_parts(page):
                text = page.get_text("text", clip=clip, sort=True)
                fragments.append(
                    {
                        "logical_index": len(fragments),
                        "pdf_page_index": pdf_index,
                        "pdf_page": pdf_index + 1,
                        "page_part": page_part,
                        "raw_text": text,
                        "printed_candidates": _printed_candidates(page, clip),
                        "source": pdf_path.name,
                    }
                )
    return fragments


def find_repeated_margin_lines(fragments: list[dict[str, Any]]) -> set[str]:
    """Find only exact, frequently repeated lines near page margins."""
    counts: Counter[str] = Counter()
    for fragment in fragments:
        lines = [line.strip() for line in fragment["raw_text"].splitlines() if line.strip()]
        for line in set(lines[:4] + lines[-4:]):
            if len(line) >= 4 and not NUMBER_RE.match(line):
                counts[line] += 1
    threshold = max(8, int(len(fragments) * 0.20))
    return {line for line, count in counts.items() if count >= threshold}


def clean_page_text(
    raw_text: str, repeated_lines: set[str], printed_page: int | None
) -> str:
    """Conservatively remove repeated margins and normalize whitespace."""
    raw_lines = raw_text.replace("\u00a0", " ").splitlines()
    cleaned: list[str] = []
    for index, raw_line in enumerate(raw_lines):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if line in repeated_lines:
            continue
        near_edge = index < 5 or index >= max(0, len(raw_lines) - 5)
        if near_edge and printed_page is not None and line == str(printed_page):
            continue
        if line or (cleaned and cleaned[-1] != ""):
            cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()
