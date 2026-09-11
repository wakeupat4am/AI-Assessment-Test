#!/usr/bin/env python3
"""Expose a dedicated local router port without loading Qwen weights twice."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class RouterProxyHandler(BaseHTTPRequestHandler):
    backend: str
    timeout_seconds: float
    max_body_bytes: int

    def _proxy(self) -> None:
        if self.path.rstrip("/") not in {"/v1/models", "/v1/chat/completions"}:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > self.max_body_bytes:
            self.send_error(413)
            return
        body = self.rfile.read(length) if length else None
        request = Request(
            f"{self.backend}{self.path}",
            data=body,
            method=self.command,
            headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
                self.send_response(response.status)
                self.send_header(
                    "Content-Type", response.headers.get("Content-Type", "application/json")
                )
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Qwen-Role", "router")
                self.end_headers()
                self.wfile.write(payload)
        except HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (URLError, TimeoutError) as exc:
            payload = json.dumps({"error": f"router backend unavailable: {exc}"}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    do_GET = _proxy
    do_POST = _proxy

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.client_address[0]} - {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedicated Qwen router-port proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=float, default=90)
    parser.add_argument("--max-body-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    RouterProxyHandler.backend = args.backend.rstrip("/")
    RouterProxyHandler.timeout_seconds = args.timeout_seconds
    RouterProxyHandler.max_body_bytes = args.max_body_bytes
    server = ThreadingHTTPServer((args.host, args.port), RouterProxyHandler)
    print(
        f"Qwen router endpoint: http://{args.host}:{args.port}/v1 "
        f"-> {RouterProxyHandler.backend}/v1",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
