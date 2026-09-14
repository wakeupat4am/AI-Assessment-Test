#!/usr/bin/env python3
"""Portable, no-touch launcher for the selected submission pipeline."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = APP_ROOT.parent
VENV_DIR = APP_ROOT / ".venv"
REQUIREMENTS = APP_ROOT / "requirements.txt"
STAMP = VENV_DIR / ".requirements.sha256"


def venv_python() -> Path:
    name = "python.exe" if os.name == "nt" else "python"
    return VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / name


def requirements_digest() -> str:
    digest = hashlib.sha256()
    digest.update(REQUIREMENTS.read_bytes())
    digest.update(f"{sys.version_info.major}.{sys.version_info.minor}".encode())
    return digest.hexdigest()


def ensure_environment(skip_setup: bool) -> Path:
    python = venv_python()
    expected = requirements_digest()
    installed = STAMP.read_text(encoding="utf-8").strip() if STAMP.exists() else ""
    if skip_setup:
        if not python.exists():
            raise SystemExit("--skip-setup requires an existing techcombank_rag/.venv")
        return python
    if not python.exists():
        print(f"Creating virtual environment with {sys.executable}", flush=True)
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    if installed != expected:
        print("Installing pinned runtime dependencies", flush=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            cwd=REPOSITORY_ROOT,
            check=True,
        )
        STAMP.write_text(expected + "\n", encoding="utf-8")
    return python


def final_environment() -> dict[str, str]:
    environment = os.environ.copy()
    fixed = {
        "INDEX_DIR": APP_ROOT / "data/index/a32_metric_aware/all",
        "B4_CONFIG": APP_ROOT / "config/b4e_metric_aware.json",
        "B4_BM25_ARTIFACT": APP_ROOT / "data/index/a32_metric_aware/all/bm25.json.gz",
        "B4_FROZEN_INDEX_DIR": APP_ROOT / "data/index/a31_semantic_multirepr/all",
        "B4_FROZEN_CONFIG": APP_ROOT / "config/b4_retrieval.json",
        "FINANCIAL_REASONING_CONFIG": APP_ROOT / "config/financial_reasoning.json",
    }
    environment.update({name: str(path) for name, path in fixed.items()})
    environment.update(
        {
            "B4_RETRIEVAL_MODE": "metric_aware",
            "XROUTER_ENABLED": "false",
            "B2_ENABLED": "false",
            "ENABLE_HYDE": "false",
            "B5_AGENT_ENABLED": "false",
            "FINANCIAL_REASONING_ENABLED": "true",
        }
    )
    environment.setdefault("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    environment.setdefault("EMBEDDING_DEVICE", "cpu")
    return environment


def run(python: Path, arguments: list[str], environment: dict[str, str]) -> None:
    subprocess.run(
        [str(python), *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )


def repository_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    caller_path = (Path.cwd() / path).resolve()
    return caller_path if caller_path.exists() else (REPOSITORY_ROOT / path).resolve()


def verify(python: Path, environment: dict[str, str]) -> None:
    verifier = APP_ROOT / "scripts/verify_artifacts.py"
    for index in (
        APP_ROOT / "data/index/a31_semantic_multirepr/all",
        APP_ROOT / "data/index/a32_metric_aware/all",
    ):
        run(python, [str(verifier), "--index-dir", str(index)], environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("chat", "demo", "evaluate", "verify", "test", "setup"),
        default="chat",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=APP_ROOT / "data/evaluation/public.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APP_ROOT / "data/evaluation/results",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Use an existing .venv; intended for CI and repeated local runs.",
    )
    args = parser.parse_args()

    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        raise SystemExit("Python 3.11 or 3.12 is required")
    python = ensure_environment(args.skip_setup)
    if args.mode == "setup":
        print(f"Environment ready: {python}")
        return

    environment = final_environment()
    if args.mode == "test":
        run(python, ["-m", "pytest", "-q"], environment | {"PYTHONPATH": str(APP_ROOT)})
        return

    verify(python, environment)
    if args.mode == "verify":
        run(python, [str(APP_ROOT / "scripts/preflight.py")], environment)
        return

    run(python, [str(APP_ROOT / "scripts/preflight.py")], environment)
    if args.mode == "chat":
        run(python, [str(APP_ROOT / "scripts/chat.py")], environment)
    elif args.mode == "demo":
        run(
            python,
            [
                str(APP_ROOT / "scripts/demo.py"),
                str(repository_path(args.questions)),
                "--output-dir",
                str(repository_path(args.output_dir)),
                "--run-name",
                "A32-B4e-B6-portable-demo",
            ],
            environment,
        )
    else:
        run(
            python,
            [
                str(APP_ROOT / "scripts/evaluate.py"),
                str(repository_path(args.questions)),
                "--output-dir",
                str(repository_path(args.output_dir)),
                "--run-name",
                "A32-B4e-B6-portable-evaluation",
            ],
            environment,
        )


if __name__ == "__main__":
    main()
