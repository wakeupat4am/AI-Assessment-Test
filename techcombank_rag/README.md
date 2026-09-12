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
the serving script or local model weights.

The frozen deployable control remains A3.1+B4b. A separate metric-aware repair
candidate is shipped as A3.2+B4e; see
[`docs/experiments/A32_B4E_METRIC_AWARE.md`](docs/experiments/A32_B4E_METRIC_AWARE.md)
for its configuration, regression gates, multi-turn results, and trade-offs.
