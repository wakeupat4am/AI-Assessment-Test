from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.preflight import validate_provider
from src.config import Settings


APP_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = APP_ROOT.parent


@pytest.mark.parametrize(
    ("provider", "base_url", "key_name"),
    [
        ("openai_compatible", "https://llm.example.test/v1", "LLM_API_KEY"),
        ("openai", "", "OPENAI_API_KEY"),
        ("anthropic", "", "ANTHROPIC_API_KEY"),
    ],
)
def test_provider_contract_accepts_replaceable_credentials(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    base_url: str,
    key_name: str,
) -> None:
    monkeypatch.setenv(key_name, "different-test-key")
    settings = Settings(
        index_dir=APP_ROOT / "data/index/a32_metric_aware/all",
        embedding_model="intfloat/multilingual-e5-small",
        llm_provider=provider,
        llm_base_url=base_url,
        llm_model="replaceable-model",
        llm_api_key="different-test-key" if key_name == "LLM_API_KEY" else "",
    )
    result = validate_provider(settings)
    assert result["provider"] == provider
    assert result["model"] == "replaceable-model"
    assert result["credential"] == "configured"


def test_portable_verify_runs_outside_repository(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "LLM_PROVIDER": "openai_compatible",
            "LLM_BASE_URL": "http://127.0.0.1:8000/v1",
            "LLM_MODEL": "replaceable-model",
            "LLM_API_KEY": "different-test-key",
            "EMBEDDING_MODEL": "intfloat/multilingual-e5-small",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(APP_ROOT / "scripts/portable_run.py"),
            "verify",
            "--skip-setup",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Index OK: 3658 chunks" in completed.stdout
    assert "Preflight OK" in completed.stdout
    assert str(APP_ROOT / "data/index/a32_metric_aware/all") in completed.stdout


def test_root_launcher_exists_and_has_no_machine_specific_path() -> None:
    launcher = (REPOSITORY_ROOT / "run.sh").read_text(encoding="utf-8")
    assert "/home/ubuntu" not in launcher
    assert "portable_run.py" in launcher
