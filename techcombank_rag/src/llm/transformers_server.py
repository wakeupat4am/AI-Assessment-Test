"""Minimal OpenAI-compatible server for a local Transformers text model.

This module is intentionally provider infrastructure only.  The RAG pipeline
talks to it through the same ``openai_compatible`` client used for vLLM,
llama.cpp, and the author-managed Qwen endpoint.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def create_handler(
    model: Any,
    tokenizer: Any,
    model_name: str,
    max_new_tokens: int,
) -> type[BaseHTTPRequestHandler]:
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
                self._json(
                    200,
                    {"object": "list", "data": [{"id": model_name, "object": "model"}]},
                )
                return
            self._json(404, {"error": {"message": "Not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._json(404, {"error": {"message": "Not found"}})
                return
            try:
                import torch

                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                messages = request.get("messages", [])
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(prompt, return_tensors="pt")
                limit = min(int(request.get("max_tokens", max_new_tokens)), max_new_tokens)
                with lock, torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=limit,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = output[0, inputs["input_ids"].shape[1] :]
                answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
                prompt_tokens = int(inputs["input_ids"].shape[1])
                completion_tokens = int(generated.shape[0])
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model_path).resolve()
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Incomplete model directory: {model_path}")
    dtype = torch.bfloat16 if args.device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        create_handler(model, tokenizer, args.model_name, args.max_new_tokens),
    )
    print(f"Serving {args.model_name} at http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
