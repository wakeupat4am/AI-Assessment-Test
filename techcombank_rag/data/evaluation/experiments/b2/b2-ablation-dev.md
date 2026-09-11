# B2 ablation — dev

| Method | Auto | Dedup | Diversity | Auto accuracy | Hit@5 | Hit@10 | Citation P/R | Avg rounds | Trigger | Evidence tokens | Duplicate slots | Retrieval ms | E2E p50 s | LLM tokens |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_baseline_b1a | N | N | N | 0.1500 | 0.7222 | 0.7222 | 0.2778/0.2778 | 1.000 | 0.000 | 2290.0 | 1.25 | 53.1 | 4.567 | 60407 |
| B_page_dedup | N | Y | N | 0.2000 | 0.7222 | 0.7222 | 0.3056/0.3333 | 1.000 | 0.000 | 2313.4 | 0.60 | 48.5 | 4.306 | 60938 |
| C_dedup_diversity | N | Y | Y | 0.2000 | 0.5000 | 0.7222 | 0.2778/0.2778 | 1.000 | 0.000 | 2110.4 | 0.20 | 71.5 | 3.297 | 62730 |
| D_auto_search | Y | N | N | 0.1500 | 0.7222 | 0.7222 | 0.2778/0.2778 | 1.100 | 0.100 | 2317.8 | 1.20 | 52.0 | 4.119 | 54139 |
| E_full_b2 | Y | Y | Y | 0.2000 | 0.5000 | 0.7222 | 0.2778/0.2778 | 1.100 | 0.100 | 2072.8 | 0.20 | 86.9 | 3.410 | 56040 |

> Automatic answer accuracy is diagnostic; no new manual review is applied.
