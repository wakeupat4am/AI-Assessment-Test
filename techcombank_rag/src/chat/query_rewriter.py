from __future__ import annotations

import json
import re
from typing import Protocol

from src.llm.base import GenerationResult, coerce_generation_result
from src.chat.text_safety import normalize_history, normalize_utf8_text


class Generator(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0,
        purpose: str = "answer",
        model: str | None = None,
    ) -> GenerationResult | str: ...


def rewrite_query(
    current_question: str,
    conversation_history: list[dict[str, str]],
    client: Generator | None = None,
    max_history_turns: int = 4,
    trace_sink: list[GenerationResult] | None = None,
    preserve_metric_identity: bool = False,
) -> str:
    """Use the configured LLM to resolve references without answering."""
    question = normalize_utf8_text(current_question).strip()
    if not question:
        raise ValueError("Question cannot be empty")
    if not conversation_history:
        return question
    if preserve_metric_identity:
        if is_self_contained_question(question):
            return question
        deterministic = _resolve_metric_followup(question, conversation_history)
        if deterministic:
            return deterministic
    if client is None:
        raise ValueError("An LLM client is required when conversation history is present")

    recent = normalize_history(conversation_history[-(max_history_turns * 2) :])
    prompt = (
        "Lịch sử hội thoại (JSON):\n"
        f"{json.dumps(recent, ensure_ascii=False)}\n\n"
        f"Câu hỏi hiện tại: {question}\n\n"
        "Hãy viết lại câu hỏi hiện tại thành một truy vấn tìm kiếm độc lập. "
        "Giữ nguyên ý định, giải quyết đại từ/tham chiếu bằng lịch sử, không trả lời, "
        "không thêm dữ kiện, và chỉ xuất đúng truy vấn đã viết lại. Nếu câu hỏi đã "
        "độc lập, giữ thay đổi ở mức tối thiểu."
    )
    generated = client.generate(
        [
            {
                "role": "system",
                "content": "Bạn chỉ viết lại truy vấn; tuyệt đối không trả lời câu hỏi.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        purpose="rewrite",
    )
    result = coerce_generation_result(generated, purpose="rewrite")
    if trace_sink is not None:
        trace_sink.append(result)
    rewritten = result.text.strip()
    if not rewritten:
        raise RuntimeError("The LLM returned an empty standalone query")
    return rewritten.strip('"').strip()


FOLLOWUP_RE = re.compile(
    r"^(?:còn\b|thế\b|vậy\b|như vậy\b|năm trước\b|năm sau\b|"
    r"chênh lệch\b|tăng bao nhiêu\b|giảm bao nhiêu\b)",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
ANAPHORA_RE = re.compile(
    r"^(?:còn\b|thế\b|vậy\b|như vậy\b)|"
    r"\b(?:nó|đó|con số (?:đó|này)|số liệu (?:đó|này|trên)|"
    r"chỉ tiêu (?:đó|này|trên)|các số liệu trên|những số liệu trên|"
    r"hai số liệu trên|ba số liệu trên|cả hai|cả ba|chúng)\b",
    re.IGNORECASE,
)


def is_self_contained_question(question: str) -> bool:
    """Return true when history is unnecessary to understand the query."""
    normalized = normalize_utf8_text(question).strip()
    return bool(normalized) and not ANAPHORA_RE.search(normalized)


def _resolve_metric_followup(
    question: str, conversation_history: list[dict[str, str]]
) -> str | None:
    """Preserve the last explicit user metric without trusting assistant facts."""
    if not FOLLOWUP_RE.search(question.strip()):
        return None
    anchor = ""
    for turn in reversed(normalize_history(conversation_history)):
        if turn.get("role") != "user":
            continue
        candidate = turn.get("content", "").strip()
        if candidate and not FOLLOWUP_RE.search(candidate):
            anchor = candidate
            break
    if not anchor:
        return None
    current_years = YEAR_RE.findall(question)
    if question.casefold().startswith("còn") and current_years:
        if YEAR_RE.search(anchor):
            return YEAR_RE.sub(current_years[-1], anchor)
        return f"{anchor.rstrip('?')} năm {current_years[-1]}"
    metric_phrase = re.split(
        r"\bnăm\s+(?:19|20)\d{2}\b|\blà bao nhiêu\b|\bthay đổi\b",
        anchor,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ?.,;:-")
    return f"{metric_phrase}: {question}" if metric_phrase else None
