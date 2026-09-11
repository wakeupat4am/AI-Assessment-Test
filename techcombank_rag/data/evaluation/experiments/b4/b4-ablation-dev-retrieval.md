# B4 retrieval-only ablation — dev

| Arm | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Mean ms | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| B0_one_shot | 0.3889 | 0.5556 | 0.7778 | 0.4677 | 56.7 | 52.3 |
| B4a_bm25 | 0.5000 | 0.7778 | 0.8333 | 0.6083 | 22.4 | 22.7 |
| B4b_hybrid_rrf | 0.5556 | 0.7778 | 0.8889 | 0.6515 | 67.0 | 66.6 |
| B4c_finance_rerank | 0.5556 | 0.7778 | 0.8889 | 0.6509 | 82.7 | 80.7 |
| B4d_cross_encoder | 0.5000 | 0.8333 | 0.8889 | 0.6315 | 64660.8 | 61514.6 |

> A3.1 chunks, E5 dense index, answer model/prompt, questions, and evaluation k are fixed.
> Only the first-stage retrieval/reranking method changes.
