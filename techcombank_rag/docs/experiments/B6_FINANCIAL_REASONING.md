# B6 — Bounded financial reasoning over A3.2+B4e

## Purpose

B6 keeps the shipped A3.2 metric-aware index and B4e hybrid retriever unchanged.
It adds a small, deterministic control layer for failure classes observed on the
20-question development set: entity confusion, incomplete multi-fact evidence,
unsafe arithmetic, and missing derivation provenance.

This is not a free-form agent and does not use LangChain. The controller exposes
only bounded retrieval and calculator operations, with at most two retrieval
rounds. `FINANCIAL_REASONING_ENABLED=false` reproduces the frozen A3.2+B4e
control.

## Pipeline

```text
question
  -> deterministic intent + entity scope + required-fact plan
  -> A3.2+B4e retrieval (20 candidates)
  -> entity/fact-aware ranking
  -> fact coverage check
       -> sufficient: stop
       -> missing: focused retrieval for missing facts (one additional round)
  -> compact evidence ledger
  -> deterministic calculator/synthesis when supported
       -> otherwise existing Qwen answer generation
  -> citations + operand-level provenance
```

## Safety and generalization boundaries

- No answer, page number, or gold label is embedded in source code.
- Rules and financial vocabulary live in `config/financial_reasoning.json`.
- A subsidiary mentioned incidentally inside a bank page does not change the
  page entity. Entity aliases are inspected only in authoritative metadata and
  the prominent source/header region.
- Arithmetic is performed only when every operand is present in retrieved
  evidence. Each operand records its printed page.
- Missing facts may trigger one focused retrieval round; repeated unconstrained
  agent loops are impossible.
- The 20-question set is development data, not a claim of hidden-test accuracy.

## Feature flags

```dotenv
FINANCIAL_REASONING_ENABLED=false
FINANCIAL_REASONING_CONFIG=techcombank_rag/config/financial_reasoning.json
ENABLE_FINANCIAL_ENTITY_SCOPE=true
ENABLE_FINANCIAL_FACT_COVERAGE=true
ENABLE_FINANCIAL_CALCULATOR=true
ENABLE_DERIVATION_PROVENANCE=true
```

The master switch defaults to false. The four sub-flags exist for ablation and
default to true only after the master switch is enabled.

## Reproduce

Interactive chat:

```bash
make run-financial
```

Fixed development benchmark:

```bash
make benchmark-financial-dev
```

Compare that output with the frozen A3.2+B4e result generated with
`FINANCIAL_REASONING_ENABLED=false`. Use identical provider, model, index,
questions, `TOP_K`, and retrieval thresholds in both runs.

## Frozen development result

Both rows below use the same reviewed 20-question set, A3.2 artifacts,
multilingual-e5-small embeddings and local Qwen3.5-9B endpoint. `Automatic` is
the deliberately strict token/numeric heuristic; `Manual` follows the stored
per-question audit file and accepts directly supported alternative pages.

| System | Automatic | Manual | Hit@5 | Recall@10 | Citation validator | Mean / p50 / p95 latency | LLM calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen A3.2+B4e | 35% | 55% | 65% | 71% | 85% | 6.67 / 6.24 / 15.91 s | 17 | 57,675 |
| A3.2+B4e+B6 | 75% | 100% | 90% | 85.5% | 100% | 3.99 / 0.54 / 11.07 s | 8 | 32,977 |

B6 stopped after one retrieval round for 16/20 questions and used the bounded
second round for 4/20, for 1.2 rounds on average. Eight answers used the
financial calculator/fact-synthesis path, four reused A3.2 deterministic metric
grounding, and eight required Qwen. Local measured API cost was USD 0. The
same-token projections recorded by the evaluator were USD 0.1450 for GPT-5.6,
USD 0.0742 for GPT-5.6 Terra, and USD 0.1088 for Claude Sonnet 5.

These are development-set results, not an estimate of hidden-test accuracy.
The remaining qualitative weakness is completeness: the answer to the broad
three-capability synthesis identifies all three groups but describes the
ecosystem less fully than the reference answer. Rules should not be expanded
from this set without holdout evidence.

The selective guardrail was then checked on the frozen public and holdout
splits. Public retained 100% Hit@5/10 and reached 100% manual accuracy. Holdout
retrieval remained exactly equal to the frozen B4b path at 77.78% Hit@1/5 and
88.89% Hit@10; manual accuracy remained 70%. The holdout automatic heuristic
varied from 40% to 30% because Qwen wording changed despite temperature zero,
which is why semantic review—not this heuristic—is the primary answer metric.
No B6 intent fired on holdout; unseen intents followed the unchanged control
path.

## Research provenance

The implementation borrows the task shape—not training code or claimed paper
results—from [FinQA](https://arxiv.org/abs/2109.00122) and
[TAT-QA](https://arxiv.org/abs/2105.07624) for numerical reasoning over financial
text and tables; [IRCoT](https://arxiv.org/abs/2212.10509) and
[Adaptive-RAG](https://arxiv.org/abs/2403.14403) for retrieval conditioned on
missing reasoning evidence; and [Program-Aided Language Models
(PAL)](https://arxiv.org/abs/2211.10435) for delegating arithmetic to executable
code. B6 is intentionally lighter: deterministic planning, at most one focused
follow-up round, and a typed calculator rather than learned routing or free-form
code generation.

## Observability

Every response includes `routing.financial_reasoning` with query type, intent,
entity scope, required/covered/missing facts, focused queries, search rounds,
candidate counts, and retrieval time. Deterministic outputs additionally include
`financial_grounding` with operation, expression, operands, operand pages, and
result.
