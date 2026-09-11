# B4 end-to-end ablation — public

| Arm | Auto acc. | Hit@1 | Hit@5 | Hit@10 | Citation P/R | Refusal | Retrieval ms | E2E p50/p95 s | Calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0_one_shot | 0.4000 | 0.5556 | 1.0000 | 1.0000 | 0.6667/0.6667 | 1.0000 | 59.7 | 6.187/7.449 | 10 | 36077 |
| B4a_bm25 | 0.4000 | 0.4444 | 0.8889 | 0.8889 | 0.5556/0.5556 | 0.9000 | 15.2 | 5.052/6.478 | 10 | 34879 |
| B4b_hybrid_rrf | 0.4000 | 0.6667 | 1.0000 | 1.0000 | 0.7778/0.7778 | 0.9000 | 67.5 | 5.197/9.603 | 11 | 38763 |
| B4c_finance_rerank | 0.4000 | 0.6667 | 1.0000 | 1.0000 | 0.6667/0.6667 | 0.9000 | 70.3 | 5.569/9.702 | 11 | 38053 |
| B4d_cross_encoder | 0.4000 | 0.3333 | 0.8889 | 0.8889 | 0.4444/0.4444 | 0.9000 | 9808.1 | 15.303/23.785 | 11 | 39317 |

> A3.1 chunks, E5 dense index, answer model/prompt, questions, and evaluation k are fixed.
> Only the first-stage retrieval/reranking method changes.
