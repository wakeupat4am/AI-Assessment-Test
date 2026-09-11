# Measurement contract for baseline A0/B0

## Freeze rule

Do not compare an improvement until `make baseline` completes with zero runtime
errors. The frozen files are `baseline-a0-b0.jsonl` and
`baseline-a0-b0.summary.json`. A rerun is comparable only when the question and
index SHA-256 values in the summary match.

## Metrics

- Retrieval Recall@1/5/10: fraction of all gold printed pages retrieved within k.
- Retrieval Hit@1/5/10: whether any gold printed page appears within k.
- Answer accuracy: manual review when supplied; otherwise token-F1 plus exact
  numeric coverage, clearly labelled as a diagnostic heuristic.
- Citation precision/recall: overlap between cited and gold printed pages.
- Citation validator pass rate: syntax/page/numeric-support guard result.
- Refusal accuracy: answerable questions should be answered and unanswerable
  questions should be refused.
- Latency: end-to-end warm-call mean, p50 and p95; per-row logs separate retrieval,
  LLM and other time.
- Cost: provider-reported tokens priced at the dated rate table. Local Qwen has
  zero API charge, not zero infrastructure cost.

## Usage trace

Every generation call logs provider, model, purpose, latency, uncached input,
cached input, cache-write input, output and reasoning tokens, plus cost and price
source. Supported purpose labels include `rewrite`, `route`, `search_planning`,
`answer`, and `repair`; A0/B0 normally exercises only answer and repair because
public questions are single-turn.

## Cost scenarios

The summary applies the exact measured token mix to a dated price snapshot for
GPT-5.6 Sol/Terra/Luna and Claude Sonnet/Opus/Fable. This is a counterfactual
token-cost estimate, not a claim that those models emit the same number of
tokens or achieve the same quality. Recheck pricing before submission or supply
the four `LLM_*_PRICE_PER_MILLION` overrides.

Price snapshot: 2026-09-09. Sources: OpenAI API model/pricing pages and Anthropic
Claude Platform pricing/model pages.

## Failure taxonomy

Each row receives exactly one primary category: `retrieval_miss`,
`false_refusal`, `unsafe_answer`, `citation_error`, `answer_error`,
`runtime_error`, or `none`. Keep the raw row when manually relabelling so the
reported taxonomy remains auditable.

## Split policy

Use `dev.json` to tune thresholds. Never inspect `holdout.json` output until a
configuration is locked. The 10 public questions are reserved for the organizer
demo and final comparable run.
