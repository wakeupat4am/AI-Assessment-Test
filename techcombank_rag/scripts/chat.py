#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chat.chatbot import Chatbot


def main() -> None:
    bot = Chatbot()
    history: list[dict[str, str]] = []
    print("Techcombank Annual Report 2025 RAG. Gõ 'exit' để thoát.")
    while True:
        try:
            question = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            response = bot.ask(question, history)
        except Exception as exc:
            print(
                "Assistant: Không thể xử lý lượt chat này. "
                f"Vui lòng thử lại. ({type(exc).__name__})"
            )
            continue
        print(f"Assistant: {response['answer']}")
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response["answer"]},
            ]
        )


if __name__ == "__main__":
    main()
