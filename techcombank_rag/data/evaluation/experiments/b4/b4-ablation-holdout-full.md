# B4 end-to-end ablation — holdout

| Arm | Auto acc. | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Retrieval ms | E2E p50/p95 s | Calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0_one_shot | 0.3000 | 0.4444 | 0.6667 | 0.7778 | 0.3333/0.3333 | 0.6000 | 61.2 | 4.619/8.990 | 11 | 37103 |
| B4a_bm25 | 0.4000 | 0.4444 | 0.7778 | 0.8889 | 0.4444/0.4444 | 0.8000 | 14.4 | 5.445/9.453 | 11 | 38448 |
| B4b_hybrid_rrf | 0.4000 | 0.7778 | 0.7778 | 0.8889 | 0.5556/0.5556 | 0.8000 | 60.8 | 5.236/8.625 | 11 | 34983 |
| B4c_finance_rerank | 0.4000 | 0.7778 | 0.7778 | 0.8889 | 0.5556/0.5556 | 0.8000 | 68.1 | 5.874/9.376 | 11 | 41310 |
| B4d_cross_encoder | 0.4000 | 0.5556 | 0.8889 | 1.0000 | 0.4444/0.4444 | 0.8000 | 11585.8 | 16.895/22.538 | 11 | 39369 |

> A3.1 chunks, E5 dense index, answer model/prompt, questions, and evaluation k are fixed.
> Only the first-stage retrieval/reranking method changes.
