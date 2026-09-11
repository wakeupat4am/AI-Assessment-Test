# B2 — AutoSearch-inspired adaptive retrieval

## Outcome

B2 is implemented as a modular wrapper around B1a. It does not change ingestion, indexes, routing policy, or answer generation. It adds one optional second retrieval round, page-aware deduplication, lexical MMR evidence selection, a final evidence token budget, and per-query traces.

This is intentionally **AutoSearch-inspired**, not a reproduction of AutoSearch RL training.

## Architecture

```text
query + conversation context
  -> B1a deterministic X-Router
  -> A2 narrative / A2 structured / A3 multi-repr / multi-hop RRF
  -> initial retrieval (top 20)
  -> page dedup (max 2 genuinely different chunks per printed page)
  -> diversity preselection (MMR-like, top 10)
  -> whole-question evidence sufficiency check
       sufficient   -> stop after round 1
       insufficient -> one focused follow-up query -> retrieve top 12
                      -> merge by chunk_id -> dedup -> diversity
  -> optional reranker hook (none is configured in this experiment)
  -> evidence budget (max 5 chunks in the controlled ablation, max 5,000 estimated tokens)
  -> unchanged Qwen answer generator with printed-page citations
```

The final answer layer refuses when AutoSearch reaches its search limit and the required evidence remains incomplete. Retrieval similarity alone is never treated as proof of answerability.

## Exact algorithms

### Adaptive controller

1. Search once with the original standalone query.
2. Process candidates using only the enabled feature flags.
3. Check required years, tracked metric/entity terms, numeric intent, cross-year values, explanation evidence, and definition evidence.
4. If every requirement is covered, stop after one round.
5. Otherwise produce one short query for the missing year, metric, comparison value, definition, or explanation.
6. Search once more, merge candidates by `chunk_id`, retain the best score, reapply enabled processing, and stop unconditionally after round 2.

No LLM call is used for the controller. The Qwen calls counted in the experiment are the pre-existing rewrite/answer calls.

### Page deduplication

- Rank by retrieval score and preserve `printed_page`, `pdf_page`, `chunk_id`, units, headers, and representation metadata.
- Remove exact chunk duplicates.
- Within one printed page, suppress row/block/page candidates whose lexical Jaccard-or-containment similarity is at least `0.88`.
- Retain at most two genuinely different chunks per page.

### Diversity selection

The preselector uses:

```text
0.7 * normalized retrieval relevance
- 0.3 * maximum lexical similarity to selected evidence
- same-page penalty
- repeated-representation penalty
+ query-aware row/block/page preference
```

A relevance safety rail reserves a slot inside the final evidence window for a strong direct metric/entity match (at least 50% informative query-token coverage and at least 85% of the top retrieval score). It does not move that chunk when it is already inside the evidence window. This prevents novelty from crowding out the answer-bearing row/block.

### Token budget and caches

- Final evidence stops at either the chunk limit or token limit.
- Token count uses a conservative tokenizer-free UTF-8 estimate so selection does not load another tokenizer.
- Dense query embeddings use a 128-entry in-process LRU cache.
- Candidate text token sets and repeated similarity inputs are cached.
- No indexed chunk embedding is recomputed.

## Configuration

```dotenv
B2_ENABLED=false
ENABLE_AUTO_SEARCH=false
ENABLE_PAGE_DEDUP=false
ENABLE_DIVERSITY_SELECTION=false
B2_INITIAL_TOP_K=20
B2_SECOND_ROUND_TOP_K=12
B2_MAX_SEARCH_ROUNDS=2
B2_MAX_PER_PAGE=2
B2_PAGE_SIMILARITY_THRESHOLD=0.88
B2_FINAL_K=6
B2_MAX_EVIDENCE_TOKENS=5000
B2_DIVERSITY_LAMBDA=0.70
B2_LOG_PATH=data/evaluation/runtime/b2_search.jsonl
QUERY_EMBEDDING_CACHE_SIZE=128
```

All features can be enabled independently. `B2_ENABLED=true` enables the wrapper; the three required flags determine its behavior. The ablation sets them explicitly and uses `final_k=5` to keep the answer evidence count controlled against B1a.

## Reproduction

From repository root:

```bash
make test
make ablate-b2
make ablate-b2-dev
```

Equivalent direct command on the CECS server:

```bash
cd /home/ubuntu/TCB_Test
PYTHONPATH=/home/ubuntu/TCB_Test/techcombank_rag \
  /home/ubuntu/techcombank_rag/.venv/bin/python \
  techcombank_rag/scripts/ablate_b2.py \
  techcombank_rag/data/evaluation/public.json --run-label public
```

## Public ablation results (10 questions)

The nine answerable questions are the denominator for retrieval/citation metrics. Qwen is served locally, so measured monetary cost is USD 0 in this run; token counts and latency remain measured. Automatic answer accuracy is a diagnostic semantic/numeric heuristic, not exact string matching and not a new manual review.

| Method | Auto accuracy | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Avg rounds | 2nd search | Evidence tokens/query | Duplicate page slots | Retrieval ms | E2E p50/p95 s | LLM tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A. B1a one-shot | 0.3000 | 0.3333 | 0.4444 | 0.6667 | 0.3333/0.3333 | 1.000 | 0.000 | 2350.9 | 0.50 | 93.2 | 4.829/10.113 | 34,035 |
| B. + page dedup | 0.3000 | 0.3333 | 0.4444 | 0.6667 | 0.3333/0.3333 | 1.000 | 0.000 | 2350.9 | 0.30 | 66.7 | 4.856/8.713 | 34,035 |
| C. + dedup + diversity | 0.3000 | 0.3333 | 0.3333 | 0.6667 | 0.3333/0.3333 | 1.000 | 0.000 | 2315.8 | 0.10 | 102.9 | 4.546/8.684 | 32,924 |
| D. + AutoSearch | 0.3000 | 0.3333 | 0.4444 | 0.6667 | 0.3333/0.3333 | 1.200 | 0.200 | 2180.9 | 0.50 | 72.2 | 4.532/7.333 | 28,122 |
| E. full B2 | 0.3000 | 0.3333 | 0.3333 | 0.7778 | 0.3333/0.3333 | 1.200 | 0.200 | 2391.0 | 0.10 | 112.0 | 3.956/7.302 | 27,218 |

Public conclusion: full B2 increases broader coverage (`Hit@10` +0.1111), removes 80% of duplicate page slots, and reduces total LLM tokens by 20.0%, while preserving automatic answer accuracy and citation P/R. It does not improve answer accuracy, and its `Hit@5` is lower by 0.1111; the diversity configuration is therefore a measured trade-off, not an unconditional replacement for B1a.

## Dev ablation results

| Method | Auto accuracy | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Avg rounds | 2nd search | Evidence tokens/query | Duplicate page slots | Retrieval ms | E2E p50/p95 s | LLM tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A. B1a one-shot | 0.1500 | 0.1667 | 0.7222 | 0.7222 | 0.2778/0.2778 | 1.000 | 0.000 | 2290.0 | 1.25 | 53.1 | 4.568/7.393 | 60,407 |
| B. + page dedup | 0.2000 | 0.1667 | 0.7222 | 0.7222 | 0.3056/0.3333 | 1.000 | 0.000 | 2313.5 | 0.60 | 48.5 | 4.306/7.239 | 60,938 |
| C. + dedup + diversity | 0.2000 | 0.1111 | 0.5000 | 0.7222 | 0.2778/0.2778 | 1.000 | 0.000 | 2110.4 | 0.20 | 71.5 | 3.297/6.954 | 62,730 |
| D. + AutoSearch | 0.1500 | 0.1667 | 0.7222 | 0.7222 | 0.2778/0.2778 | 1.100 | 0.100 | 2317.8 | 1.20 | 52.0 | 4.119/7.188 | 54,139 |
| E. full B2 | 0.2000 | 0.1111 | 0.5000 | 0.7222 | 0.2778/0.2778 | 1.100 | 0.100 | 2072.8 | 0.20 | 86.9 | 3.411/6.951 | 56,040 |

Dev conclusion: full B2 improves the diagnostic answer score by 0.05, preserves Hit@10, removes 84% of duplicate page slots, reduces evidence tokens per query by 9.5%, reduces total LLM tokens by 7.2%, and lowers p50 by 25.3%. It again reduces Hit@5. Page dedup alone is the safest default on dev: it raises answer accuracy and citation recall without reducing Hit@5 or Hit@10.

## Observability

Every query writes a JSONL event containing:

- original and follow-up queries;
- one or two search rounds and stop reason;
- initial, post-dedup, post-diversity, and final evidence counts;
- final printed pages and estimated evidence tokens;
- sufficiency confidence and missing-information labels;
- retrieval, rerank, and total retrieval latency;
- underlying B1 route traces for each round.

## Files changed

- `src/retrieval/auto_search.py`: controller, sufficiency checker, follow-up generation, merge, traces, wrapper.
- `src/retrieval/page_dedup.py`: printed-page and overlap deduplication.
- `src/retrieval/diversity.py`: MMR-like selection, representation penalties, direct-match safety rail, token budget.
- `src/retrieval/retriever.py`: query-embedding cache.
- `src/chat/chatbot.py`: optional B2 wrapper, final selected evidence, grounded refusal after exhausted search.
- `src/config.py`, root/project `.env.example`: feature flags and tunables.
- `scripts/ablate_b2.py`: five-arm public/dev evaluation and Markdown/JSON output.
- `scripts/compare_b1_b2_answers.py`: separate 10-question B1a/B1b/B2 answer artifact.
- `tests/test_b2_adaptive_retrieval.py`: controller, dedup, diversity, insufficiency, no-gain second search, and budget regression tests.
- root `Makefile`: `ablate-b2` and `ablate-b2-dev` commands.

No ingestion or index-building code was changed.

## Tests and acceptance checks

- 93 repository tests pass locally and on the CECS server.
- 11 B2-specific tests cover one-hop numeric, cross-year comparison, two-page evidence, row/block/page duplicates, narrative explanation, insufficient evidence, productive and unproductive second search, independent flags, token budget, and direct metric evidence retention.
- Easy public questions use one search; adaptive search never exceeds two rounds.
- All 10 public queries completed with zero runtime errors.
- Page metadata remains present in retrieved chunks and all produced citations pass the citation syntax/availability validator.

## Observed failure cases

1. A high-similarity chunk can contain the requested year and a plausible number without containing the exact requested metric. The deterministic checker still needs more Vietnamese metric aliases.
2. Diversity improves broad coverage and duplicate usage but can demote the organizer's single gold page even when an equivalent supporting page exists. This explains the `Hit@5` trade-off.
3. A second retrieval can find an additional gold page without giving the answer generator enough coherent evidence to improve automatic accuracy.
4. Printed gold pages are sometimes stricter than factual support: correct values cited from another report page count as citation misses.
5. No reranker is configured, so the final ordering depends on dense retrieval plus inexpensive lexical evidence selection.

## Recommendation

Keep B2 available behind flags, but do not claim it dominates B1a yet. The next iteration should calibrate `lambda_relevance` and the direct-match threshold on dev only, then freeze them before holdout. If latency permits, rerank only the approximately 10 deduplicated/diverse candidates—not the raw top 20–100—and improve the sufficiency checker with metric/year/value tuples plus table-header and unit binding.
