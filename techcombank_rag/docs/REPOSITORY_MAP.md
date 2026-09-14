# Repository map

This map separates the small grading surface from author-only research code.
The root `Makefile` is the supported entry point; graders do not run ingestion.

## Grading path

| Location | Responsibility |
|---|---|
| `Makefile` | Clean-machine setup, chat, live demo, batch evaluation and artifact verification. |
| `.env.example` | Canonical provider/model/runtime configuration without secrets. |
| `SUBMISSION.md` | Decisions, measured results, costs, limitations and demo link. |
| `techcombank_rag/scripts/chat.py` | Thin multi-turn terminal entry point. |
| `techcombank_rag/scripts/demo.py` | Streams all batch answers, citations and latency for the unedited demo. |
| `techcombank_rag/scripts/evaluate.py` | Non-interactive batch entry point that writes JSONL and summary artifacts. |
| `techcombank_rag/src/chat/` | Conversation grounding, answer generation, citation checks and bounded financial reasoning. |
| `techcombank_rag/src/retrieval/` | Dense/BM25 retrieval, fusion, metric identity and optional research controllers. |
| `techcombank_rag/src/llm/` | Provider abstraction, usage/cost accounting and optional local serving helpers. |
| `techcombank_rag/src/evaluation/` | Shared scoring and summary contract used by batch and live-demo entry points. |
| `techcombank_rag/data/index/a31_semantic_multirepr/all/` | Frozen A3.1 fallback FAISS/BM25/chunk artifacts. |
| `techcombank_rag/data/index/a32_metric_aware/all/` | Selected A3.2 metric-aware FAISS/BM25/chunk artifacts. |

## Research path

| Location | Responsibility |
|---|---|
| `techcombank_rag/src/ingestion/` | Offline PDF parsing and document representations; not required for grading. |
| `techcombank_rag/src/routing/` | B1 rule and LLM router experiments; disabled in the selected runtime. |
| `techcombank_rag/scripts/ablate_*.py` | Controlled Track B ablations. |
| `techcombank_rag/scripts/build_*.py` | Author-side index construction. |
| `techcombank_rag/data/evaluation/experiments/` | Raw outputs, summaries and manual reviews retained for auditability. |
| `techcombank_rag/docs/experiments/` | Experiment design, negative results and interpretation. |

## Supported commands

```bash
make run-financial       # selected multi-turn CLI
make demo-financial      # stream the 10 public questions for the demo video
make evaluate-financial  # non-interactive JSONL + summary evaluation
make verify-financial-indexes
make test
```

`make run` and `make baseline` reproduce frozen controls. Other Make targets are
author-side experiments and are not prerequisites for grading.
