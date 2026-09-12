<!-- Filled from SUBMISSION_TEMPLATE.md. Keep these headings unchanged. -->

# Submission

## How to run

- **Clean machine:** prerequisites are Python 3.11 or 3.12 with `venv`, `make`, network access for the first dependency/model download, and an accessible OpenAI-compatible, OpenAI, or Anthropic generation endpoint. From the repository root, this single command creates `.venv`, installs pinned dependencies, verifies the shipped A3.1 FAISS and BM25 checksums, and opens the final A3.1+B4b CLI:

  ```bash
  INDEX_DIR=techcombank_rag/data/index/a31_semantic_multirepr/all \
    B4_RETRIEVAL_MODE=hybrid \
    B4_BM25_ARTIFACT=techcombank_rag/data/index/a31_semantic_multirepr/all/bm25.json.gz \
    EMBEDDING_MODEL=intfloat/multilingual-e5-small \
    LLM_PROVIDER=openai_compatible LLM_BASE_URL=http://YOUR_HOST:PORT/v1 \
    LLM_MODEL=YOUR_MODEL LLM_API_KEY=YOUR_KEY make run
  ```

  Alternatively, copy `.env.example` to `.env`, change only the endpoint/model/key, then run `make run`. No `/home/ubuntu/...` path, OCR service, ingestion job, or author server is required.

- **Different key or local model:** configuration is environment-only; no key or model name is hard-coded in Python. Examples:

  ```bash
  # OpenAI
  LLM_PROVIDER=openai LLM_MODEL=YOUR_OPENAI_MODEL LLM_API_KEY=... make run

  # Anthropic
  LLM_PROVIDER=anthropic LLM_MODEL=YOUR_ANTHROPIC_MODEL LLM_API_KEY=... make run

  # Qwen/vLLM/llama.cpp/any OpenAI-compatible local server
  LLM_PROVIDER=openai_compatible LLM_BASE_URL=http://127.0.0.1:8000/v1 \
    LLM_MODEL=YOUR_LOCAL_MODEL LLM_API_KEY=local make run
  ```

  `LLM_MODEL_FAST` and `LLM_MODEL_STRONG` are optional. If they are blank, every purpose gracefully falls back to `LLM_MODEL`.

- **Batch run:** no human input is required:

  ```bash
  INDEX_DIR=techcombank_rag/data/index/a31_semantic_multirepr/all \
    B4_RETRIEVAL_MODE=hybrid \
    B4_BM25_ARTIFACT=techcombank_rag/data/index/a31_semantic_multirepr/all/bm25.json.gz \
    EMBEDDING_MODEL=intfloat/multilingual-e5-small \
    LLM_PROVIDER=openai_compatible LLM_BASE_URL=http://YOUR_HOST:PORT/v1 \
    LLM_MODEL=YOUR_MODEL LLM_API_KEY=YOUR_KEY \
    make evaluate QUESTIONS=techcombank_rag/data/evaluation/public.json
  ```

  Results are written as per-question JSONL plus an adjacent summary containing retrieval, answer, citation, refusal, latency, token, cost, and failure metrics.

- **Environment variables:** `.env.example` is the canonical, commented list. The runtime reads the following groups:

  | Group | Variables | Purpose |
  |---|---|---|
  | Generation | `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_FAST`, `LLM_MODEL_STRONG`, `LLM_TEMPERATURE`, `LLM_TIMEOUT_SECONDS`, `MAX_NEW_TOKENS` | Provider abstraction, credentials, model-tier fallback and generation limits. |
  | Conversation/search | `MAX_HISTORY_TURNS`, `MAX_SEARCH_STEPS`, `TOP_K`, `EVALUATION_RETRIEVAL_K`, `MIN_RETRIEVAL_SCORE` | Multi-turn context, evidence count, evaluation depth and refusal threshold. |
  | Embedding/artifacts | `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`, `EMBEDDING_THREADS`, `INDEX_DIR`, `PROCESSED_DIR`, `EVALUATION_DIR`, `A31_INDEX_ROOT`, `A31_PARENT_CHARACTERS`, `QUERY_EMBEDDING_CACHE_SIZE` | Query encoder, shipped index locations, output locations, parent expansion and cache. The embedding model must match index metadata. |
  | Selected B4 retriever | `B4_RETRIEVAL_MODE`, `B4_CONFIG`, `B4_BM25_ARTIFACT`, `B4_CROSS_ENCODER_MODEL`, `B4_CROSS_ENCODER_DEVICE`, `B4_CROSS_ENCODER_BATCH_SIZE`, `B4_CROSS_ENCODER_MAX_LENGTH`, `B4_CROSS_ENCODER_THREADS` | Selects `off`, BM25, hybrid B4b, deterministic reranking, or optional cross-encoder settings. The shipped default is `hybrid`. |
  | Optional X-Router | `XROUTER_ENABLED`, `XROUTER_TYPE`, `XROUTER_CONFIG`, `XROUTER_CONFIDENCE_THRESHOLD`, `XROUTER_LOG_PATH`, `NARRATIVE_INDEX_DIR`, `STRUCTURED_INDEX_DIR`, `MULTI_REPR_INDEX_DIR` | B1 retrieval-route ablations; disabled in the final path. |
  | Optional router LLM | `ROUTER_LLM_PROVIDER`, `ROUTER_LLM_BASE_URL`, `ROUTER_LLM_MODEL`, `ROUTER_LLM_API_KEY`, `ROUTER_LLM_TEMPERATURE`, `ROUTER_LLM_TIMEOUT_SECONDS`, `ROUTER_LLM_MAX_NEW_TOKENS` | Independently configurable B1b planner endpoint; falls back to the main provider values. |
  | Optional B2 | `B2_ENABLED`, `ENABLE_AUTO_SEARCH`, `ENABLE_PAGE_DEDUP`, `ENABLE_DIVERSITY_SELECTION`, `B2_INITIAL_TOP_K`, `B2_SECOND_ROUND_TOP_K`, `B2_MAX_SEARCH_ROUNDS`, `B2_MAX_PER_PAGE`, `B2_PAGE_SIMILARITY_THRESHOLD`, `B2_FINAL_K`, `B2_MAX_EVIDENCE_TOKENS`, `B2_DIVERSITY_LAMBDA`, `B2_LOG_PATH` | Adaptive retrieval, page deduplication, diversity selection and token budget; disabled by default. |
  | Optional HyDE | `ENABLE_HYDE`, `HYDE_MODE`, `HYDE_PROMPT_VARIANT`, `HYDE_CONFIG`, `HYDE_PROVIDER`, `HYDE_MODEL`, `HYDE_BASE_URL`, `HYDE_API_KEY`, `HYDE_TEMPERATURE`, `HYDE_TIMEOUT_SECONDS`, `HYDE_MAX_TOKENS`, `HYDE_CANDIDATE_K`, `HYDE_RRF_K`, `HYDE_ORIGINAL_WEIGHT`, `HYDE_HYPOTHETICAL_WEIGHT`, `HYDE_CACHE_SIZE`, `HYDE_LOG_PATH` | B3 hypothetical-document ablation; disabled by default and never used as answer evidence. |
  | Optional B5 | `B5_AGENT_ENABLED`, `B5_AGENT_CONFIG`, `B5_AGENT_LOG_PATH` | Audited retrieval/calculator tool calling; disabled by default. |
  | Cost/debug | `LLM_INPUT_PRICE_PER_MILLION`, `LLM_CACHED_INPUT_PRICE_PER_MILLION`, `LLM_CACHE_WRITE_PRICE_PER_MILLION`, `LLM_OUTPUT_PRICE_PER_MILLION`, `DEBUG` | Optional pricing override and diagnostic logging. |
  | Author-only local Qwen | `QWEN_MODEL_PATH`, `QWEN_PYTHON`, `QWEN_DEPS`, `QWEN_GPU_IDS`, `QWEN_MAX_MEMORY_GIB` | Optional serving helper; not required by graders. |

- **Shipped index:** the final runtime uses `techcombank_rag/data/index/a31_semantic_multirepr/all/`:

  - `chunks.jsonl`: 3,658 A3.1 row/block/page/metric children;
  - `index.faiss`: 384-dimensional normalized E5 vectors;
  - `bm25.json.gz`: Vietnamese lexical index over the identical chunks;
  - `metadata.json` and `bm25.metadata.json`: source hashes, model, dimensions, counts, parameters and artifact checksums.

  `make verify-index` validates both retrieval artifacts. Graders do not run ingestion.

  The compact frozen A0/B0 dense index is additionally shipped at
  `techcombank_rag/data/index/`, allowing `make baseline` to reproduce the
  original comparator without changing the selected final runtime.

## Depth track(s) chosen

I chose two linked tracks because the frozen baseline showed that generation was not the first bottleneck: evidence often never reached the answer model.

1. **Document Intelligence.** The question was whether better parsing alone could improve a cheap dense RAG system. I progressed from PyMuPDF fixed chunks through PaddleOCR-VL 1.6 plain/Markdown serialization, layout-aware chunks, multi-granularity row/block/page representations, and finally A3.1 semantic metric children with inherited headings and parent expansion. This track was informed by the [PaddleOCR-VL 1.6 technical report](https://arxiv.org/abs/2606.03264), but every representation was evaluated with the same E5 embedder, Qwen answer model, question set and page-aware scoring.

2. **Retrieval and controlled reasoning.** Once A3.1 exposed stronger evidence but remaining misses were retrieval/ranking failures, I evaluated deterministic and LLM routers inspired by [X-Router](https://aclanthology.org/2026.findings-acl.994/), bounded second retrieval inspired by [AutoSearch](https://aclanthology.org/2026.findings-acl.1399/), [HyDE](https://aclanthology.org/2023.acl-long.99/) query expansion, BM25/dense fusion informed by [BGE-M3](https://arxiv.org/abs/2402.03216), deterministic and cross-encoder reranking, and bounded calculator/retrieval tools. I deliberately do not claim full paper reproduction where I did not train the original policy.

### Research path that produced the final system

The starting point was a deliberately simple **A0/B0 baseline**: PyMuPDF text extraction, page-bounded fixed chunks, multilingual E5, FAISS top-k, Qwen3.5-9B and deterministic citation/numeric validation. On the 10 published questions it achieved manual answer accuracy 60%, Hit@5 44.44%, Hit@10 55.56%, and canonical citation recall 33.33%. Its failure taxonomy—three retrieval misses, one false refusal and one citation error—made retrieval the first optimization target. The frozen configuration and metrics are preserved in the [A0/B0 baseline report](techcombank_rag/docs/baseline_a0_b0.md).

The first controlled result was negative: replacing only extraction with PaddleOCR-VL plain text (A1a) tied A0 on manual accuracy and Hit@5/10, while Markdown fixed chunks (A1b) regressed. Better OCR did not automatically create better retrieval units ([A1 report](techcombank_rag/docs/experiments/A1_PADDLEOCR_VL.md)). Layout-aware A2 then increased manual public accuracy to 70%; A3 row/block/page increased dev Hit@5/10 to 77.78% ([A2–A4 report](techcombank_rag/docs/experiments/A2_A4_DOCUMENT_INTELLIGENCE.md)). A3.1 generalized the useful part: it added 196 metric-value children, inherited headings, entity/year/unit fields, child-first retrieval and bounded parent expansion without any public-question allowlist. Against original A0, A3.1 one-shot raised public Hit@5 from 44.44% to 100% and Hit@10 from 55.56% to 100% ([A3.1 report](techcombank_rag/docs/experiments/A31_SEMANTIC_MULTIREPR.md)).

Track B then asked whether more control would improve that representation. Rule routing helped some A2 dev retrieval but did not beat A3.1 one-shot ([B1a](techcombank_rag/docs/experiments/B1A_XROUTER_RULE_BASED.md)). An LLM router added roughly three seconds and made less reliable choices ([B1b](techcombank_rag/docs/experiments/B1B_XROUTER_LLM_AGENT.md)). Dedup/diversity reduced representation crowding but could demote exact evidence ([B2](techcombank_rag/docs/experiments/B2_ADAPTIVE_RETRIEVAL.md)). HyDE fusion improved some dev retrieval ranks but added an LLM call and raised p50 from about 4.8 s to 13.1 s ([B3](techcombank_rag/docs/experiments/B3_HYDE.md)). The decisive experiment was B4: BM25-only, dense+BM25 fusion, finance rules, and a multilingual cross-encoder were compared on the same A3.1 chunks across public/dev/untuned holdout. **B4b dense E5 + BM25 weighted RRF** gave the best deployable quality/latency/cost balance ([B4 report](techcombank_rag/docs/experiments/B4_HYBRID_RETRIEVAL.md)). B5 finally demonstrated safe arithmetic tools, but its aggregate automatic accuracy did not improve and tail latency increased, so it remains optional ([B5 report](techcombank_rag/docs/experiments/B5_TOOL_CALLING.md)).

After freezing that selection, an adversarial multi-turn test exposed a narrower metric-identity failure: cash-flow service receipts, financial-note service income, and NFI share most lexical terms. I therefore kept A3.1+B4b byte-for-byte as the control and built a separate **A3.2+B4e metric-aware candidate**. It adds search-only OCR normalization, metric/statement qualifiers, conservative selective routing, identity-preserving follow-up rewriting, and deterministic same-row calculations. It tied the frozen system on public/dev/holdout retrieval, while a 10-question metric-confusion diagnostic improved Hit@5 from 80% to 100%, manually reviewed answer accuracy from 60% to 100%, and citation precision/recall from 60% to 100%. This is reported as targeted error repair—not a hidden-test estimate—and its two-index trade-off and raw evidence are documented in the [A3.2+B4e report](techcombank_rag/docs/experiments/A32_B4E_METRIC_AWARE.md).

## Architecture

The final default is intentionally smaller than the experiment tree:

```text
OFFLINE, SHIPPED
197 PDF sheets
  -> PaddleOCR-VL 1.6 checkpoint (393 printed-page fragments)
  -> A3 row/block/page representations
  -> A3.1 metric-value children + inherited section/entity/year/unit
  -> one E5 embedding pass
  -> FAISS index + BM25 inverted index + checksummed metadata

ONLINE, PER QUESTION
query + bounded conversation context
  -> E5 dense top-20 -----------+
  -> BM25 lexical top-20 -------+-> weighted Reciprocal Rank Fusion
                                      -> top child evidence
                                      -> bounded parent-context expansion
                                      -> provider-configurable grounded LLM
                                      -> printed-page + verbatim-number validator
                                      -> cited answer, repair, or refusal
```

Dense retrieval handles paraphrase and semantic similarity; BM25 protects exact financial terms, abbreviations, years and numbers. RRF combines ranks without pretending their raw scores are calibrated. A compact child is retrieved first so a specific row or metric can rank highly; only then is its parent block appended so the answer model sees the heading, table context and units. Every chunk preserves its printed page. The answer layer may cite only retrieved pages and rejects unsupported numeric claims. Multi-turn history helps resolve follow-ups, but it is never treated as a source of financial truth.

The final path uses no router, HyDE generation, reranker model or autonomous loop. Those components remain independently switchable for ablation. B5's calculator accepts operands only when both literal values and source pages exist in retrieved evidence.

The optional A3.2+B4e mode is selected explicitly with `B4_RETRIEVAL_MODE=metric_aware`; it does not replace or mutate the default above. Ordinary queries still traverse the exact frozen A3.1+B4b path, while ambiguity-risk queries can use the separate metric-aware index and audited row answerer.

## What I tried that did not work

- **PaddleOCR-VL followed by the same fixed chunker:** A1a tied A0 at 60% manual accuracy and 44.44%/55.56% Hit@5/10; A1b Markdown fell to 40% accuracy and 22.22%/33.33% Hit@5/10. The finding was that OCR fidelity and retrieval representation are different problems.
- **Promoting every multi-granularity or selective-cascade variant:** A3 had the best early dev coverage and latency, while A4 recovered some glossary/table pages, but neither dominated A2 on public answer accuracy. This led to A3.1 rather than selecting a whole-page-heavy index.
- **LLM X-Router:** on a 25-query route diagnostic, Qwen routing accuracy was 72% versus 100% for the reference rules; p50 router latency was 3.61 s. End-to-end public accuracy did not improve and dev Hit@5 fell 11.11 points versus the rule router. Adding reasoning was not automatically better routing.
- **AutoSearch-inspired B2 as a default:** page dedup reduced duplicate-page slots and full B2 reduced tokens, but diversity selection lowered early Hit@5. The sufficiency checker did not reliably bind entity, metric, year, value and unit, so B2 remains behind flags.
- **HyDE-only and unconditional HyDE:** HyDE-only lost exact financial matches. The best fusion variant improved dev Hit@5 from 55.56% to 61.11% and Hit@10 from 77.78% to 83.33%, but p50 end-to-end latency rose from 4.77 s to 13.07 s. Hypothetical text is therefore never evidence and HyDE is not default.
- **Generic cross-encoder reranking:** B4d reached 80% manual dev accuracy, but CPU retrieval cost approximately 9.8–11.6 s and holdout manual accuracy remained 70%, equal to B4b. It was outside the deployment Pareto frontier.
- **Always-on agent/tool orchestration:** B5 correctly calculated the holdout difference `8.645 - 6.075 = 2.570 tỷ đồng` from page-257 evidence, but aggregate automatic accuracy stayed unchanged while holdout retrieval mean rose from about 72 ms to 1.77 s and p95 reached 13.61 s. It is retained only as an audited optional capability.

These negative results are kept with raw rows and traces under `techcombank_rag/data/evaluation/experiments/`; they are not retroactively hidden or merged into the winning run.

## Evaluation

- **Method:** I froze the initial A0/B0 result before improvements. `public.json` contains the 10 published questions, `dev.json` contains 20 diagnostic questions used for calibration, and `holdout.json` contains 10 questions not used for tuning. Retrieval uses organizer gold printed pages and reports Hit@1/5/10. Citation precision/recall compares every cited page with all recorded gold pages; alternative supporting pages are manually accepted only after inspecting the retrieved source. Refusal is scored separately for answerable/unanswerable questions. The automatic answer diagnostic requires complete numeric recall plus token-F1 >= 0.45; it is not exact string matching, but still undercounts concise correct paraphrases. The primary public answer judgement is an auditable per-question manual semantic review.

- **Results on the 10 published questions:**

  | Metric | Original A0/B0 | Final A3.1+B4b |
  |---|---:|---:|
  | Manual semantic answer accuracy | 60.0% | 90.0% |
  | Automatic diagnostic accuracy | not used as primary | 40.0% |
  | Retrieval Hit@1 | 11.11% | 66.67% |
  | Retrieval Hit@5 | 44.44% | 100.0% |
  | Retrieval Hit@10 | 55.56% | 100.0% |
  | Canonical citation precision / recall | 33.33% / 33.33% | 77.78% / 77.78% |
  | Refusal accuracy | 70.0% | 90.0% |
  | End-to-end p50 / p95 | 4.640 / 18.946 s | 5.197 / 9.603 s |

  The final choice is not based on public alone. Across all 40 public/dev/holdout questions, B4b achieved 72.5% manual accuracy, Hit@1/5/10 of 63.9%/83.3%/91.7%, and canonical citation P/R of 63.9%. On the untouched holdout, it improved manual accuracy from A3.1 dense-only's 50% to 70%, Hit@1 from 44.4% to 77.8%, and citation overlap from 33.3% to 55.6% at roughly 61 ms retrieval latency.

- **Audit evidence:** the exact B4 public/dev/holdout tables are in the [B4 experiment report](techcombank_rag/docs/experiments/B4_HYBRID_RETRIEVAL.md); the ten public outputs are listed in [B4 public answers](techcombank_rag/docs/experiments/B4_PUBLIC_ANSWERS.md); retrieval/citation disagreements are documented in the [evidence analysis](techcombank_rag/docs/experiments/B4_EVIDENCE_ANALYSIS.md). Machine-readable per-question rows, summaries, manual reviews and ablation tables are under [`techcombank_rag/data/evaluation/experiments/`](techcombank_rag/data/evaluation/experiments/). This separates measured outputs from interpretation and keeps negative experiments reproducible.

- **What the numbers made me change:** the A0 failure taxonomy moved work from prompting to evidence construction. A1's flat/regressive result moved the focus from OCR serialization to layout-aware semantic chunks. A3's coverage led to A3.1 child/parent representations. Router, B2 and HyDE results prevented adding unconditional LLM decisions. B4's public/dev/holdout evidence selected B4b over the higher-dev-accuracy but 11-second B4d reranker. B5's single arithmetic win justified keeping validated tools, but its aggregate/latency result prevented making the system agentic by default.

## Cost and latency

- **Ingestion:** the one-time PaddleOCR-VL checkpoint covered all 393 printed-page fragments from 197 PDF sheets. On the recorded CPU/contention run, checkpoint wall time was 35,573.843 s (9 h 52 m 53.843 s). A3.1 then reused that frozen checkpoint—no OCR or LLM call—and built 3,658 chunks plus three FAISS views in 483.753 s (8 m 3.753 s). The selected `all` runtime directory, including FAISS, chunks and BM25, is approximately 19 MiB; the full A3.1 family is approximately 35 MiB. Self-hosted OCR/embedding incurred USD 0 API charges; hardware/electricity was not converted to money. Graders use the shipped index and pay none of this ingestion time.

- **Per query:** on the published final B4b run, mean retrieval was 67.5 ms and end-to-end p50/p95 was 5.197/9.603 s. The 10 questions used 38,763 tokens, averaging about 3,876 tokens/query. Self-hosted Qwen API charge was recorded as USD 0; this excludes hardware cost. Same-token list-price projections from the repository's dated pricing snapshot were approximately `$0.01618/query` for GPT-5.6 Sol, `$0.00817/query` for GPT-5.6 Terra, `$0.00082/query` for GPT-5.6 Luna, `$0.01214/query` for Claude Sonnet 5, and `$0.02023/query` for Claude Opus 5. These are exposure estimates, not invoices or quality claims.

- **Models used:** PaddleOCR-VL 1.6 for offline document parsing; `intfloat/multilingual-e5-small` (384 dimensions, normalized) for offline chunk and online query embeddings; Qwen/Qwen3.5-9B at temperature 0 through an OpenAI-compatible endpoint for measured generation. B4d's `BAAI/bge-reranker-v2-m3` and all alternative providers are optional and not required by the final path.

## What I deliberately did not build

- I did not require cloud deployment; the assessment asks for clean-machine reproducibility, provider portability and shipped indexes, which the repository supplies.
- I did not train the full [AutoSearch](https://aclanthology.org/2026.findings-acl.1399/) reinforcement-learning policy. B2 is explicitly a bounded, deterministic/LLM-light inspired controller with at most two rounds.
- I did not claim [RAG-on-a-Diet](https://aclanthology.org/2026.acl-long.1562/): it requires learned hop-wise routing over multiple model tiers and trajectory/reward training. B5 is only bounded host tool execution.
- I did not put HyDE, an LLM router, cross-encoder or tool agent on every request because measured latency/quality did not justify them.
- I did not fine-tune on the 10 public questions or add query-specific metric allowlists. A3.1 extraction rules operate on layout and document structure.
- I did not train a learned sparse retriever or benchmark many Vietnamese embedding models before the deadline; BM25 gave the required exact-term complement with a small, auditable artifact.
- I did not allow generated hypothetical text, conversation memory, or calculator operands without source evidence to become citations.

## With 10x time and budget

1. Build a larger Vietnamese financial QA evaluation set with complete multi-page gold evidence, independent dual manual review and an untouched test split.
2. Benchmark Vietnamese/multilingual dense encoders and learned sparse retrievers on exactly the same A3.1 chunks; compare E5+BM25 against dense+[SPLADE](https://arxiv.org/abs/2107.05720)-style fusion.
3. Add an auditable financial glossary/alias field (`LNTT/PBT`, `CASA`, `RBG`, `NIM`, `ROE`) while preserving verbatim source text for citations.
4. Improve multi-hop sufficiency using explicit `(entity, metric, year, value, unit)` slots, then test selective second retrieval and selective HyDE only for a verified missing slot.
5. Train/calibrate a RAG-on-a-Diet-style hop controller over fast/medium/strong models only after enough trajectories exist to measure the quality-cost frontier.
6. Add targeted chart/figure understanding and a multilingual financial reranker optimized for the actual deployment hardware.

## Demo video

**TODO before submission:** add the 3–5 minute unedited video link showing a clean start and one uninterrupted batch run over all 10 published questions, including real latency, citations, refusal behavior and any visible failure.
