#!/usr/bin/env python3
"""Validate provider configuration without sending a model request."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Settings, get_settings


def validate_provider(settings: Settings) -> dict[str, str]:
    provider = settings.llm_provider.casefold()
    if provider not in {"openai_compatible", "openai", "anthropic"}:
        raise ValueError(
            "LLM_PROVIDER must be openai_compatible, openai, or anthropic"
        )
    if not settings.default_model:
        raise ValueError("Set LLM_MODEL, LLM_MODEL_FAST, or LLM_MODEL_STRONG")
    if provider == "openai_compatible" and not settings.llm_base_url:
        raise ValueError("LLM_BASE_URL is required for openai_compatible")
    fallback_name = ""
    if provider == "openai":
        fallback_name = "OPENAI_API_KEY"
    elif provider == "anthropic":
        fallback_name = "ANTHROPIC_API_KEY"
    local_compatible = provider == "openai_compatible" and any(
        host in settings.llm_base_url.casefold()
        for host in ("127.0.0.1", "localhost", "[::1]")
    )
    if not settings.llm_api_key and not os.getenv(fallback_name, "") and not local_compatible:
        raise ValueError(f"Set LLM_API_KEY{f' or {fallback_name}' if fallback_name else ''}")

    metadata = json.loads((settings.index_dir / "metadata.json").read_text(encoding="utf-8"))
    indexed_model = str(metadata["embedding_model"])
    if settings.embedding_model != indexed_model:
        raise ValueError(
            f"EMBEDDING_MODEL must be {indexed_model!r} for the shipped index, "
            f"not {settings.embedding_model!r}"
        )
    return {
        "provider": provider,
        "model": settings.default_model,
        "base_url": (
            settings.llm_base_url if provider == "openai_compatible" else "provider default"
        ),
        "credential": "configured" if settings.llm_api_key or os.getenv(fallback_name, "") else "local endpoint",
        "embedding_model": indexed_model,
        "index_dir": str(settings.index_dir),
    }


def main() -> None:
    try:
        details = validate_provider(get_settings())
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"Preflight failed: {error}") from error
    print("Preflight OK (no API request sent)")
    for name, value in details.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
