# Track A continuation — A2, A3, A4

## Câu hỏi nghiên cứu

A1 cho thấy PaddleOCR-VL 1.6 không tự động cải thiện RAG khi output bị đưa qua
fixed character chunking. Ba thí nghiệm tiếp theo tách các giả thuyết còn lại:

- **A2:** cấu trúc heading/table có hữu ích nếu chunk boundary hiểu layout?
- **A3:** row, block và page có bù trừ cho nhau theo loại câu hỏi không?
- **A4:** có thể giữ narrative text rẻ/chính xác của PyMuPDF và chỉ bổ sung
  PaddleOCR-VL cho table/chart regions hay không?

Không OCR lại. Cả ba dùng chung 393 raw PaddleOCR-VL checkpoints của A1.

## Biến được khóa

- Embedding: `intfloat/multilingual-e5-small`, normalized vectors.
- Index: FAISS `IndexFlatIP`.
- Generator: Qwen3.5-9B qua OpenAI-compatible API, temperature 0.
- Retrieval: top 5 evidence cho answer; top 10 cho evaluation.
- Prompt, refusal threshold, citation validator và 10 public questions không đổi.
- Public dùng manual review; dev chỉ dùng để chẩn đoán, không dùng làm bằng
  chứng answer accuracy cuối cùng. Holdout không được đụng trong vòng này.

## A2 — layout-aware heading/table chunks

Nguồn là `structured_result.res.parsing_res_list`, không parse lại layout bằng
regex. Mỗi block có label, content, bbox và reading order.

- `doc_title` và `paragraph_title` tạo heading context.
- Narrative blocks liên tiếp dưới cùng heading được gom tới budget 3.200 ký tự.
- Table là boundary cứng; không trộn table với narrative kế bên.
- HTML/Markdown table được parse thành rows, rồi ghép deterministic
  `column header: cell value`. Table lớn được chia theo row; heading được lặp lại.
- Running header/footer/page number và image block rỗng bị loại.
- Không summary và không gọi LLM trong ingestion.

Mục tiêu là kiểm tra chunk boundary/layout, không kiểm tra model retrieval mới.

## A3 — multi-granularity row/block/page

Cùng nội dung PaddleOCR-VL được index ở ba view:

- **row:** từng table row với column header/value và heading context;
- **block:** layout-aware section/table chunks từ A2;
- **page:** một page signature deterministic gồm label và preview của các block,
  giới hạn 3.200 ký tự để không vượt context hữu dụng của E5-small.

Không áp dụng reranker hay page-diversity rule trong A3. Điều này cố ý giữ
retrieval algorithm giống A0; nếu row/block cùng một trang crowd top-k, đó là
failure mode cần ghi nhận trước khi sang Track B.

## A4 — selective cascade

- Narrative chunks được lấy nguyên từ A0 processed pages và dùng đúng fixed
  chunk 3.200/overlap 480. Parser provenance là `PyMuPDF`.
- PaddleOCR-VL chỉ bổ sung non-empty `table`, `chart`, `figure_title`,
  `vision_footnote`, `display_formula` regions trên các trang có table/chart.
- Narrative OCR của Paddle không được trộn lại, tránh duplication và giữ ưu thế
  text layer của PDF.

Raw checkpoint hiện được tạo bằng full-page PaddleOCR-VL, vì vậy vòng này đo
**retrieval quality của selective representation**, chưa phải measured compute
saving của region-level selective inference. Muốn claim tiết kiệm ingestion,
cần một run riêng: layout detector → crop table/chart → VLM recognition, và đo
số region/GPU-seconds thực tế.

## Artifact contract

```text
data/index/a2_paddleocr_vl_1_6_layout_aware/
data/index/a3_paddleocr_vl_1_6_multigranularity/
data/index/a4_selective_cascade/
  index.faiss
  chunks.jsonl
  metadata.json

data/evaluation/experiments/
  a2-public.jsonl / a2-public.summary.json
  a3-public.jsonl / a3-public.summary.json
  a4-public.jsonl / a4-public.summary.json
  a2-dev.jsonl / a3-dev.jsonl / a4-dev.jsonl
  track-a-comparison.json
```

## Index build thực tế

| Run | Vectors | Thành phần | Dung lượng | Build CPU |
|---|---:|---|---:|---:|
| A0 | 446 | 446 fixed narrative chunks | khoảng 1,8 MB | artifact đã freeze |
| A2 | 1.111 | 833 narrative + 274 table + 4 fallback | 3,3 MB | 215,962 s |
| A3 | 3.462 | 1.111 block + 1.974 row + 377 page | 8,8 MB | 458,199 s |
| A4 | 2.487 | 446 PyMuPDF + 2.041 structured regions | 6,3 MB | 269,386 s |

A4 chọn 154/393 printed pages (39,2%) có table/chart, nhưng mỗi table row lại là
một vector nên index vẫn lớn gấp 5,58 lần A0. PaddleOCR-VL trả về 23 chart blocks
nhưng `block_content` của chúng rỗng; pipeline không bịa text cho chart. Vì vậy
A4 hiện cải thiện table/formula/footnote, chưa kiểm tra được chart semantic QA.

## Kết quả public — 10 câu của đề

Answer accuracy bên dưới là human-reviewed. Citation recall vẫn được tính cứng
theo canonical gold pages của đề; một citation ở trang lặp lại có thể thực sự
hợp lệ nhưng vẫn nhận điểm 0 ở metric này. Đây là lý do giữ cả manual notes và
automatic canonical score trong artifact.

| Metric | A0 | A1a | A1b | A2 | A3 | A4 |
|---|---:|---:|---:|---:|---:|---:|
| Recall@1 | 0,111111 | 0,000000 | 0,000000 | **0,333333** | **0,333333** | **0,333333** |
| Recall@5 | 0,444444 | 0,444444 | 0,222222 | 0,444444 | **0,555556** | **0,555556** |
| Recall@10 | 0,555556 | 0,555556 | 0,333333 | **0,666667** | **0,666667** | 0,555556 |
| Answer accuracy (manual) | 0,600000 | 0,600000 | 0,400000 | **0,700000** | 0,600000 | 0,600000 |
| Citation recall (canonical) | 0,333333 | 0,222222 | 0,111111 | 0,333333 | 0,333333 | **0,444444** |
| Refusal accuracy | 0,700000 | 0,700000 | 0,600000 | **0,800000** | 0,700000 | 0,700000 |
| p50 latency (s) | 4,639500 | 4,651000 | 4,864500 | 5,165500 | **4,034500** | 5,821000 |
| p95 latency (s) | 18,945900 | 8,815650 | 9,441750 | 9,512600 | **6,292950** | 15,052850 |
| Input tokens | 51.078 | 36.526 | 38.096 | 33.653 | **28.079** | 46.573 |
| GPT-5.6 projected cost (USD) | 0,215152 | 0,152484 | 0,158824 | 0,142252 | **0,118636** | 0,197232 |
| Claude Sonnet 5 projected cost (USD) | 0,161364 | 0,114363 | 0,119118 | 0,106689 | **0,088977** | 0,147924 |

### Public question-level findings

- **A2 cứu SQ-01:** canonical page 4 đi từ rank 3 ở A0 lên rank 1; câu trả lời
  đúng 302 điểm giao dịch và 29/34 tỉnh thành. Đây là cải thiện thực sự tạo ra
  mức answer accuracy 70%, cao hơn A0 10 điểm phần trăm.
- **A3 crowding:** SQ-01 vẫn có page 4 ở rank 2 nhưng top-5 chứa các view trùng
  trang; generator lại từ chối. SQ-04 có page 53 và 49 lặp hai lần trước
  canonical page 5 ở rank 8. Multi-granularity tăng coverage nhưng chưa đủ nếu
  thiếu parent/child grouping và page diversity.
- **A4 cứu glossary/table:** SQ-08 và SQ-09 đưa đúng row glossary lên rank 1,
  trích đúng trang 387 và 386. Tuy nhiên SQ-07 mất canonical page 59 khỏi top 10
  do structured rows cạnh tranh với narrative chunks.
- **Lỗi chung không phải OCR:** SQ-03 không phương án nào đưa canonical page 5
  vào top 10; SQ-06 thường ưu tiên số 28,3/53.391 ở trang khác thay vì đủ cặp
  53,4 + CAGR 16,5%; SQ-07 của A2/A3 có page 59 trong top-5 nhưng answer stage
  vẫn từ chối. Đây là lỗi retrieval discrimination và evidence utilization,
  phù hợp để xử lý ở Track B thay vì tiếp tục đổi parser.

## Kết quả dev — 20 câu chẩn đoán

Dev chưa human-review nên answer/citation score tự động chỉ là tín hiệu phụ.
Retrieval recall là số đáng dùng để so representation ở vòng này.

| Metric | A2 | A3 | A4 |
|---|---:|---:|---:|
| Recall@1 | 0,166667 | 0,166667 | **0,333333** |
| Recall@5 | 0,611111 | **0,777778** | 0,611111 |
| Recall@10 | 0,611111 | **0,777778** | 0,722222 |
| Answer accuracy (automatic diagnostic) | 0,150000 | 0,200000 | **0,300000** |
| Citation recall (automatic) | 0,222222 | 0,444444 | **0,500000** |
| Refusal accuracy | 0,600000 | **0,700000** | **0,700000** |
| p50 latency (s) | **4,561000** | 4,766500 | 4,614500 |
| p95 latency (s) | **6,994250** | 8,740850 | 13,231600 |
| Input tokens | 61.484 | **60.060** | 77.904 |

A3 tăng Recall@5/10 lên 77,78%, cao nhất trên dev. Điều này ủng hộ giả thuyết
multi-granularity có giá trị, nhưng public failure cho thấy phải thêm
dedup/diversification hoặc parent-child retrieval trước khi đưa thẳng vào
production.

## Quyết định sau gate

1. **Promote A2 thành Track A winner hiện tại.** A2 đạt answer accuracy cao nhất
   (70%), Recall@10 cao hơn A0 11,11 điểm phần trăm, giữ citation recall, giảm
   34,1% input tokens và giảm projected GPT-5.6 cost từ 0,215152 xuống 0,142252
   USD cho cùng 10 câu. Đây là phương án cân bằng nhất ở retrieval baseline hiện
   tại.
2. **Giữ A3 làm candidate cho Track B, không deploy nguyên trạng.** Nó có recall
   dev tốt nhất, latency/cost public tốt nhất, nhưng 3.462 vectors và duplicate
   views gây top-k crowding. Bước hợp lý là retrieve theo từng view, merge theo
   printed page, rồi diversity/rerank trước generation.
3. **Không promote A4 ở cấu hình hiện tại.** A4 cải thiện canonical citation và
   glossary table, nhưng answer accuracy không hơn A0, p50 chậm hơn, cost cao hơn
   A2/A3 và index 2.487 vectors. Muốn gọi đây là low-cost cascade cần build lại
   theo region-level OCR thật và đo GPU-seconds, không chỉ reuse full-page OCR.
4. **Không mở holdout trong vòng này.** Sau khi Track B fix duplicate/evidence
   selection và threshold được khóa trên dev, holdout mới là gate cuối để chống
   overfit vào 10 public questions.

## Reproducibility và audit

- Server run hoàn tất không có runtime error: 30 public question-runs và 60 dev
  question-runs (LLM call count có thể cao hơn khi citation repair chạy).
- `pytest`: 23 tests passed, gồm parser/table serialization, A2 boundaries, A3
  unique row/block/page IDs và A4 exact baseline narrative/selective regions.
- Mỗi index có `metadata.json`, SHA-256 của source PDF/raw manifest/chunks/index,
  và đã qua `verify_artifacts.py` trên server.
- Human labels nằm ở `manual_review.a2.public.json`,
  `manual_review.a3.public.json`, `manual_review.a4.public.json`; comparison tổng
  nằm ở `track-a-comparison.json`.
