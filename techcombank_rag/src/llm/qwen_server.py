from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def extract_generated_text(result: Any) -> str:
    if isinstance(result, list) and result:
        result = result[0]
    if isinstance(result, dict):
        result = result.get("generated_text", result.get("text", result))
    if isinstance(result, list) and result:
        last = result[-1]
        if isinstance(last, dict):
            result = last.get("content", last.get("text", last))
    if isinstance(result, list):
        result = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in result
        )
    return str(result).strip()


def create_handler(generator: Any, model_name: str, max_new_tokens: int) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
                return
            self._json(404, {"error": {"message": "Not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._json(404, {"error": {"message": "Not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                messages = []
                for message in request.get("messages", []):
                    content = message.get("content", "")
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content}]
                    messages.append({"role": message.get("role", "user"), "content": content})
                limit = int(request.get("max_tokens", max_new_tokens))
                with lock:
                    result = generator(
                        text=messages,
                        max_new_tokens=min(limit, max_new_tokens),
                        generate_kwargs={"do_sample": False},
                        enable_thinking=False,
                    )
                answer = extract_generated_text(result)
                tokenizer = getattr(generator, "tokenizer", None)
                prompt_tokens = 0
                completion_tokens = 0
                if tokenizer is not None:
                    try:
                        prompt_tokens = len(
                            tokenizer.apply_chat_template(
                                messages,
                                tokenize=True,
                                add_generation_prompt=True,
                            )
                        )
                        completion_tokens = len(
                            tokenizer.encode(answer, add_special_tokens=False)
                        )
                    except Exception:
                        # Usage is optional in the compatibility protocol. The
                        # client records it as unavailable if this block fails.
                        prompt_tokens = 0
                        completion_tokens = 0
                self._json(
                    200,
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": answer},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    },
                )
            except Exception as error:
                self._json(500, {"error": {"message": str(error)}})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible Qwen server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-memory-gib", type=int, default=22)
    args = parser.parse_args()

    import torch
    from transformers import pipeline

    model_path = Path(args.model_path).resolve()
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Incomplete model directory: {model_path}")
    max_memory = {
        index: f"{args.max_memory_gib}GiB" for index in range(torch.cuda.device_count())
    }
    generator = pipeline(
        "image-text-to-text",
        model=str(model_path),
        device_map="auto",
        trust_remote_code=True,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "max_memory": max_memory,
        },
    )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        create_handler(generator, args.model_name, args.max_new_tokens),
    )
    print(f"Serving {args.model_name} at http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
