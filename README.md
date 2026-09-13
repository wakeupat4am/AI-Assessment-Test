# Techcombank Annual Report 2025 RAG

Reproducible Vietnamese question answering over the 197-sheet Techcombank
Annual Report. The selected runtime is A3.2+B4e+B6: metric-aware A3.2 evidence,
dense E5 + BM25 retrieval, and a selective bounded financial-reasoning layer.
Unrecognized queries retain the frozen A3.2+B4e path. It uses shipped
FAISS/BM25 artifacts and never re-runs ingestion.

## Clean-machine quick start

Prerequisites: Python 3.11 or 3.12, `make`, and an accessible LLM endpoint.

```bash
cp .env.example .env
# Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY in .env
make run-financial
```

`make run-financial` creates an isolated virtual environment, installs pinned
packages, verifies both shipped index families, then starts the CLI. No
host-specific path is required. `make run` remains the A3.1+B4b control.

## Provider configuration

Use one of these environment-only configurations; keys and model names are not
hard-coded in Python.

```dotenv
# Qwen, vLLM, llama.cpp or another OpenAI-compatible server
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://server:8000/v1
LLM_MODEL=Qwen/Qwen3.5-9B
LLM_API_KEY=local
```

```dotenv
# Native OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.6-terra
LLM_API_KEY=...
```

```dotenv
# Native Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-5
LLM_API_KEY=...
```

`LLM_MODEL_FAST` and `LLM_MODEL_STRONG` are optional. When set, lightweight
purposes use FAST and answer/repair use STRONG. When only `LLM_MODEL` is set,
all purposes gracefully fall back to that single model.

## Reproducible evaluation

```bash
make test
make verify-financial-indexes
make evaluate-financial QUESTIONS=techcombank_rag/data/evaluation/public.json
make benchmark-financial-all
make baseline
```

`make baseline` always writes the frozen public result to
`techcombank_rag/data/evaluation/baseline/baseline-a0-b0.jsonl` and its adjacent
summary. Each question records retrieval ranks, answer, citations, refusal,
latency breakdown, provider/model, purpose, input/cache/output/reasoning tokens,
and cost. The summary includes Recall/Hit@1/5/10, answer and citation accuracy,
refusal accuracy, p50/p95, failure taxonomy, and same-token cost projections.

Evaluation splits are deliberately separated:

- `public.json`: the 10 organizer questions used in the demo.
- `dev.json`: 20 questions used for diagnosis and threshold calibration.
- `holdout.json`: 10 questions that must not be used for tuning.

Automatic answer accuracy is a deterministic diagnostic heuristic. Copy
`manual_review.example.json`, review each result, and pass `--manual-review` for
the report's primary human score.

## Shipped artifacts

The selected runtime artifacts are under
`techcombank_rag/data/index/a32_metric_aware/all/` with the frozen fallback at
`techcombank_rag/data/index/a31_semantic_multirepr/all/`. Both contain
`index.faiss`, `chunks.jsonl`, `metadata.json`, `bm25.json.gz`, and
`bm25.metadata.json`.
Metadata records the PDF hash, embedding model, dimensions, chunking parameters,
and checksums for both runtime files. `make verify-financial-indexes` detects a corrupt or
mismatched artifact. `make ingest` exists only to document the author's build;
graders do not need it.

The small frozen A0/B0 FAISS artifact is also included at
`techcombank_rag/data/index/` so `make baseline` genuinely reruns the original
one-shot baseline. Other experimental indexes are omitted; their raw outputs,
summaries and manual reviews are retained.

Implementation detail and known limitations are in
[`techcombank_rag/BASELINE_DECISIONS.md`](techcombank_rag/BASELINE_DECISIONS.md),
and the measurement contract is in
[`techcombank_rag/docs/measurement.md`](techcombank_rag/docs/measurement.md).
The frozen 09/09 server run is summarized in
[`techcombank_rag/docs/baseline_a0_b0.md`](techcombank_rag/docs/baseline_a0_b0.md).

The controlled Track A comparison—A0 versus [PaddleOCR-VL 1.6](https://arxiv.org/abs/2606.03264) plain text
(A1a) and Markdown (A1b)—is documented with manual-reviewed metrics, failure
analysis, costs, and checksums in
[`techcombank_rag/docs/experiments/A1_PADDLEOCR_VL.md`](techcombank_rag/docs/experiments/A1_PADDLEOCR_VL.md).
The measured per-question outputs and summaries are shipped for audit. The
alternative indexes are intentionally excluded from the deployable package;
their build and benchmark entry points remain as `make benchmark-a1a` and
`make benchmark-a1b` for an author-side reproduction.

The continuation—A2 layout-aware heading/table chunks, A3 row/block/page
multi-granularity, and A4 selective PyMuPDF + Paddle structured regions—is
documented in
[`techcombank_rag/docs/experiments/A2_A4_DOCUMENT_INTELLIGENCE.md`](techcombank_rag/docs/experiments/A2_A4_DOCUMENT_INTELLIGENCE.md).
Human-reviewed results and machine-readable traces are shipped; A3.1 and the
aligned A3.2 metric-aware index are the runtime indexes. The author-side commands are
`make benchmark-a2`, `make benchmark-a3`, and `make benchmark-a4`; graders do
not rerun OCR or ingestion.

Track B1a adds a deterministic, config-driven router inspired by
[X-Router](https://aclanthology.org/2026.findings-acl.994/) over A2 narrative,
structured, and A3 multi-granularity paths without changing answer generation.
Its architecture, tests, public/dev ablation and limitations are documented in
[`techcombank_rag/docs/experiments/B1A_XROUTER_RULE_BASED.md`](techcombank_rag/docs/experiments/B1A_XROUTER_RULE_BASED.md).
Run `make ablate-b1a` or `make ablate-b1a-dev`; set `XROUTER_ENABLED=true` to
enable the router in the normal chatbot runtime.

Track B1b replaces only that route decision with a constrained Qwen LLM-agent
call. The negative controlled result, token/latency accounting, public/dev
ablation, two-port server layout, and retained B1a decision are documented in
[`techcombank_rag/docs/experiments/B1B_XROUTER_LLM_AGENT.md`](techcombank_rag/docs/experiments/B1B_XROUTER_LLM_AGENT.md).
Use `make router-proxy`, `make evaluate-router-b1b`, `make ablate-b1b`, and
`make ablate-b1b-dev` to reproduce it. The stricter A2-only comparison is
available through `make ablate-b1b-controlled-a2` and
`make ablate-b1b-controlled-a2-dev`.

Track B2 keeps B1a as its parent and adds an
[AutoSearch](https://aclanthology.org/2026.findings-acl.1399/)-inspired controller,
printed-page deduplication, diversity-aware evidence selection, token budgets,
and full retrieval traces. Each feature is independently switchable and the
controller stops after one search when evidence is sufficient, with a hard
maximum of two rounds. Architecture, algorithms, tests, public/dev five-arm
ablation, trade-offs, and failure cases are documented in
[`techcombank_rag/docs/experiments/B2_ADAPTIVE_RETRIEVAL.md`](techcombank_rag/docs/experiments/B2_ADAPTIVE_RETRIEVAL.md).
Run `make ablate-b2` or `make ablate-b2-dev`; set `B2_ENABLED=true` plus the
three `ENABLE_*` flags to activate it in the normal chatbot runtime.
Track B4 adds a shipped BM25 artifact over the frozen A3.1 chunks and compares
BM25-only, dense+BM25 RRF, deterministic finance-aware reranking, and a
multilingual cross-encoder. The recommended quality/latency trade-off is B4b
dense+BM25 RRF. Controlled public/dev/untuned-holdout evidence is documented in
[`techcombank_rag/docs/experiments/B4_HYBRID_RETRIEVAL.md`](techcombank_rag/docs/experiments/B4_HYBRID_RETRIEVAL.md).
Run `make build-bm25`, `make verify-b4-index`, and the three `ablate-b4*`
targets to reproduce it; set `B4_RETRIEVAL_MODE=hybrid` to enable it.

B6 keeps A3.2+B4e frozen and selectively adds entity/fact planning, at most one
focused missing-fact retrieval round, typed arithmetic/fact synthesis, and
operand-level printed-page provenance. Generic and unseen intents bypass B6.
Run `make run-financial`, `make evaluate-financial`, or
`make benchmark-financial-all`; design, citations, negative ablation and frozen
public/dev/holdout evidence are in
[`techcombank_rag/docs/experiments/B6_FINANCIAL_REASONING.md`](techcombank_rag/docs/experiments/B6_FINANCIAL_REASONING.md).
