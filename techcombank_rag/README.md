# Baseline implementation

Use the repository-root `Makefile` and `.env.example`; this directory contains
the application implementation and shipped data artifacts.

Pipeline: PyMuPDF spread splitting and printed-page mapping → conservative text
cleaning → page-bounded 3,200-character chunks → multilingual E5 embeddings →
FAISS `IndexFlatIP` → provider-configurable grounded generation → deterministic
citation and numeric-support validation.

The baseline intentionally excludes OCR, layout reconstruction, hybrid search,
reranking, query decomposition, calculator tools, and agentic search. Those are
experimental tracks and must be compared against this frozen A0/B0 result.

For an author-managed Qwen server, set all machine-specific paths explicitly:

```bash
QWEN_MODEL_PATH=/path/to/model \
QWEN_PYTHON=/path/to/python \
QWEN_DEPS=/optional/dependency/path \
QWEN_GPU_IDS=2,3 \
./scripts/serve_qwen.sh
```

The grading runtime only needs the server URL in root `.env`; it does not need
the serving script or local model weights. `scripts/serve_transformers_text.py`
is an optional thin entry point for testing another local Transformers causal
LM through the same OpenAI-compatible provider interface.

The frozen deployable control remains A3.1+B4b. The selected candidate is
A3.2+B4e plus the selective B6 bounded financial-reasoning layer; see
[`docs/experiments/A32_B4E_METRIC_AWARE.md`](docs/experiments/A32_B4E_METRIC_AWARE.md)
and [`docs/experiments/B6_FINANCIAL_REASONING.md`](docs/experiments/B6_FINANCIAL_REASONING.md)
for configuration, regression gates, public/dev/holdout results and trade-offs.
Use the repository-root `make run-financial`; `make run` still reproduces the
control.
