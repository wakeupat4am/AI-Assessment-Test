from __future__ import annotations

import json
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
) -> str:
    """Use the configured LLM to resolve references without answering."""
    question = normalize_utf8_text(current_question).strip()
    if not question:
        raise ValueError("Question cannot be empty")
    if not conversation_history:
        return question
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
