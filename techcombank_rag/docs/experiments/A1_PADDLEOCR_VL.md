# A1 Document Intelligence — PaddleOCR-VL 1.6

## Mục tiêu

Thí nghiệm đầu tiên của Track A kiểm tra liệu một document parser đa phương thức có cải thiện RAG so với A0 hay không, đồng thời cô lập ảnh hưởng của **cách biểu diễn output**:

- **A0:** PyMuPDF embedded text → fixed chunk.
- **A1a:** PaddleOCR-VL 1.6 → deterministic plain text → fixed chunk.
- **A1b:** PaddleOCR-VL 1.6 → canonical Markdown → fixed chunk.

Không thay đổi retriever, embedding model, FAISS metric, chunk-size, overlap, top-k, threshold, answer model hoặc prompt. Vì A1a và A1b đọc từ cùng một raw OCR checkpoint, chúng không chịu nhiễu do model parse sinh output khác nhau giữa hai lần chạy.

## Vì sao dùng full pipeline

`PaddleOCR-VL-1.6-0.9B` chỉ là VLM recognition submodel. Thí nghiệm này dùng top-level pipeline `PaddleOCR-VL-1.6`, gồm PP-DocLayoutV3, layout analysis, region crop, reading order, VLM recognition và result assembly. Đây mới là phép thử Document Intelligence đầy đủ.

Pipeline 1.6 được phát hành ngày 28/05/2026. Model có 0,9B tham số; tài liệu chính thức báo cáo 96,33% trên OmniDocBench v1.6 và cải thiện text, table, formula recognition. Các con số này là kết quả upstream, **không được xem là metric của tập Techcombank**; quyết định đưa vào hệ thống chỉ dựa trên benchmark nội bộ bên dưới.

## Thiết kế kiểm soát

| Biến | A0 | A1a | A1b |
|---|---|---|---|
| Parser | PyMuPDF text layer | PaddleOCR-VL 1.6 full pipeline | PaddleOCR-VL 1.6 full pipeline |
| Intermediate | plain text | plain text | Markdown |
| Chunking | fixed 3.200 chars, overlap 480 | giống A0 | giống A0 |
| Page boundary | không cross printed page | không cross printed page | không cross printed page |
| Embedding | `intfloat/multilingual-e5-small` | giống A0 | giống A0 |
| Index | normalized vectors, FAISS FlatIP | giống A0 | giống A0 |
| Generator | Qwen3.5-9B, temperature 0 | giống A0 | giống A0 |
| Public eval | cùng 10 câu và gold printed pages | giống A0 | giống A0 |

### Mapping trang và citation

PDF có cả sheet dọc một trang và spread ngang hai trang. Trước OCR, code split spread thành `left/right`, render từng printed-page fragment ở 160 DPI, rồi gắn metadata `pdf_page`, `page_part`, `printed_page`. PaddleOCR chỉ tạo nội dung dùng để retrieval; text layer của PDF chỉ được dùng để suy ra mapping trang in, không bị trộn vào nội dung A1.

Mỗi chunk chỉ thuộc một printed page. Nếu đáp án cần bằng chứng trên nhiều trang, evaluator đo recall trên **toàn bộ** `gold_printed_pages`, và chatbot phải liệt kê đủ citation liên quan.

### A1a: plain text serialization

Đầu vào là trường `markdown_texts` của raw PaddleX result. Converter thuần deterministic:

1. bỏ heading/list/emphasis/fence syntax;
2. link chỉ giữ anchor text, image chỉ giữ alt text;
3. HTML table chuyển cell thành chuỗi phân cách bằng `;`;
4. Markdown table bỏ separator row nhưng giữ toàn bộ header/value;
5. chuẩn hóa whitespace, không summarize và không gọi LLM.

### A1b: Markdown serialization

Giữ nguyên canonical `markdown_texts` từ PaddleX, bao gồm heading, bảng Markdown/HTML và công thức. Chỉ fixed-chunk theo character window như A0; không semantic chunking ở bước này để tránh trộn Track A với Track B.

## Artifact contract

```text
data/processed/paddleocr_vl_1_6/raw/
  fragment_*.json                 # checkpoint dùng chung A1a/A1b
  manifest.json
  page_mapping.json
data/index/a1a_paddleocr_vl_1_6_plain_fixed/
  index.faiss
  chunks.jsonl
  metadata.json
data/index/a1b_paddleocr_vl_1_6_markdown_fixed/
  index.faiss
  chunks.jsonl
  metadata.json
data/evaluation/experiments/
  a1a-public.jsonl
  a1a-public.summary.json
  a1b-public.jsonl
  a1b-public.summary.json
```

Hai index là artifact runtime cần ship. Raw OCR/model weights là artifact nghiên cứu, không phải prerequisite trên clean machine của ban tổ chức.

## Reproduce ingestion (author only)

Official setup cho GPU:

```bash
python -m venv .venv-paddleocr-vl
.venv-paddleocr-vl/bin/pip install paddlepaddle-gpu==3.3.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
.venv-paddleocr-vl/bin/pip install -r techcombank_rag/requirements-paddleocr-vl.txt
PADDLE_PYTHON=.venv-paddleocr-vl/bin/python PADDLE_DEVICE=gpu:0 make ingest-a1
```

CPU fallback dùng `paddlepaddle==3.3.0` và `PADDLE_DEVICE=cpu`. Khi native dynamic inference quá chậm, full pipeline vẫn giữ PP-DocLayoutV3 cục bộ và có thể đưa riêng VLM recognition qua llama.cpp bằng checkpoint GGUF chính thức:

```bash
llama-server -hf PaddlePaddle/PaddleOCR-VL-1.6-GGUF \
  --host 127.0.0.1 --port 8111 --temp 0
.venv-paddleocr-vl/bin/python techcombank_rag/scripts/ingest_paddleocr_vl.py \
  --pdf "$PDF" --device cpu \
  --vl-backend llama-cpp-server --vl-server-url http://127.0.0.1:8111/v1
```

Pipeline checkpoint theo từng fragment và có `--skip-parse`, `--num-shards`, `--shard-index` để resume/chạy song song. Không chạy lại ingestion khi chấm bài.

## Server run được đo

- Host: `cecs_server_1`, 64 logical CPU, 251 GiB RAM.
- PaddlePaddle 3.3.0 CPU; PaddleX 3.6.1; PaddleOCR 3.6.0 wrapper.
- Full pipeline: PP-DocLayoutV3 + PaddleOCR-VL-1.6-0.9B.
- Render: 160 DPI, one printed-page fragment per prediction.
- Native Paddle smoke test: printed page 1, 107,877 giây OCR; output 89 Markdown characters và đúng title/reading order của cover.
- llama.cpp GGUF smoke test trên cùng trang: 17,043 giây (nhanh hơn 6,33×); full pipeline và serializer không đổi.
- Lý do CPU: process mới không khởi tạo được NVML/CUDA tại thời điểm chạy; Qwen API đang hoạt động trên GPU 2–3 nên không restart driver/server để tránh phá baseline.
- Full parse dùng checkpoint atomic theo fragment. Pass chính ghi 390 fragment qua
  `llama-cpp-server`; ba fragment lỗi/timeout (logical index 18, 34, 59) được
  điền bằng native Paddle cùng checkpoint 1.6. Không dùng text layer làm fallback
  cho nội dung OCR.
- Để vượt qua contention của host, pass được resume bằng 8 residue worker không
  giao nhau và hai llama.cpp process, mỗi process 4 slot. Đây là parallelism của
  ingestion offline, không phải thay đổi model hay biến thí nghiệm.
- Đủ 393/393 printed fragments, trong đó 4 fragment hợp lệ không có text và 151
  fragment chứa Markdown/HTML table.
- Checkpoint window: 35.573,843 giây (9 giờ 52 phút 53,843 giây). Tổng
  `ocr_seconds` cộng theo record là 119.054,687 giây vì nhiều worker chạy song
  song; median 216,966 giây, p95 797,408 giây. Các số này phản ánh CPU host đang
  bị contention, không phải latency online của chatbot.
- Raw Markdown có 1.367.516 ký tự; A1a plain còn 797.109 ký tự (58,29%).
- Build index A1a: 176,772 giây, 440 chunks. Build index A1b: 260,628 giây,
  653 chunks. Cả hai index có 384 chiều và đã pass artifact verifier.

## Benchmark public — kết quả

Tất cả run dùng cùng 10 câu public, gold pages, E5-small, FAISS FlatIP, top-5
evidence/top-10 retrieval evaluation, Qwen3.5-9B và temperature 0. `Answer
accuracy` bên dưới là manual review của cả 10 câu; automatic token-F1 chỉ là
diagnostic. Không run nào có runtime error.

| Metric | A0 | A1a plain | Δ A1a−A0 | A1b Markdown | Δ A1b−A0 |
|---|---:|---:|---:|---:|---:|
| Retrieval Recall@1 | 0,111111 | 0,000000 | -0,111111 | 0,000000 | -0,111111 |
| Retrieval Recall@5 | 0,444444 | 0,444444 | 0,000000 | 0,222222 | -0,222222 |
| Retrieval Recall@10 | 0,555556 | 0,555556 | 0,000000 | 0,333333 | -0,222223 |
| Answer accuracy (manual) | 0,600000 | 0,600000 | 0,000000 | 0,400000 | -0,200000 |
| Citation precision | 0,333333 | 0,222222 | -0,111111 | 0,111111 | -0,222222 |
| Citation recall | 0,333333 | 0,222222 | -0,111111 | 0,111111 | -0,222222 |
| Citation validator pass | 0,800000 | 1,000000 | +0,200000 | 1,000000 | +0,200000 |
| Refusal accuracy | 0,700000 | 0,700000 | 0,000000 | 0,600000 | -0,100000 |
| Latency p50 (s) | 4,639500 | 4,651000 | +0,011500 | 4,864500 | +0,225000 |
| Latency p95 (s) | 18,945900 | 8,815650 | -10,130250 | 9,441750 | -9,504150 |
| LLM calls | 13 | 10 | -3 | 10 | -3 |
| Input tokens | 51.078 | 36.526 | -14.552 | 38.096 | -12.982 |
| Output tokens | 542 | 319 | -223 | 322 | -220 |
| Projected GPT-5.6 cost (USD) | 0,215152 | 0,152484 | -0,062668 | 0,158824 | -0,056328 |
| Projected Claude Sonnet 5 cost (USD) | 0,161364 | 0,114363 | -0,047001 | 0,119118 | -0,042246 |

Qwen local có measured monetary cost bằng 0 trong harness; bảng cost model lớn là
projection trên đúng số token của từng run. A1 có p95 thấp hơn A0 chủ yếu vì cả
hai A1 run không kích hoạt ba repair call như A0. Vì vậy không được diễn giải
chênh lệch p95 này thành lợi ích trực tiếp của OCR.

`Citation validator pass=1,0` chỉ chứng minh citation nằm trong context đã
retrieve. Canonical citation recall lại giảm, cho thấy validator cú pháp không
thay thế phép đo citation so với gold page.

## Phân tích theo câu hỏi

- **A1a cứu retrieval của sq-07:** gold page 59 từ miss ở A0 lên rank 4. Tuy
  nhiên generator vẫn false-refuse dù evidence nằm trong top 5. Đây là lỗi
  generation/answerability gate, không còn là lỗi parser.
- **A1a không cải thiện coverage tổng:** Recall@5/@10 bằng A0 nhưng gold page 4
  của sq-01 tụt rank 3 → 8; sq-08 tụt khỏi top 10. Sq-02 vẫn trả đúng nhờ trang
  lặp 43 nhưng citation canonical page 4 bị mất.
- **A1b làm retrieval xấu đi:** gold của sq-01, sq-02 và sq-08 rời top 10;
  page 5 của sq-04 tụt xuống rank 6, ngoài top-5 evidence. Recall@10 giảm 22,22
  điểm phần trăm.
- **Markup tạo fragmentation:** cùng raw OCR, A1b tạo 653 chunks so với 440 của
  A1a (+48,41%) và dùng nhiều hơn 1.570 input tokens (+4,30%). Với fixed
  character windows, table tags/pipes tiêu tốn window nhưng không tạo signal
  đủ mạnh cho E5-small.
- **Lỗi answer vẫn còn khi retrieve đúng:** sq-07 có page 59 ở rank 4 (A1a) và
  rank 2 (A1b) nhưng cả hai từ chối. A1b sq-09 retrieve canonical page 386 ở
  rank 3 nhưng diễn giải CASA thành nhãn tỷ lệ thay vì “Tiền gửi không kỳ hạn”.
- **Citation trên trang lặp:** sq-05 A0/A1a trả đúng 1,13% từ trang 55, trong khi
  gold canonical là trang 5. Manual answer accuracy chấp nhận nội dung đúng,
  nhưng citation metric vẫn giữ tiêu chuẩn canonical để so run nhất quán.

## Quyết định tại gate

Không chọn A1 chỉ vì parser mới hơn. Candidate phải:

1. tăng Recall@5/10 hoặc cứu được lỗi table/chart cụ thể;
2. không giảm citation recall do chunk chứa markup nhiễu;
3. không làm answer accuracy/refusal xấu đi;
4. serving latency và token cost vẫn trong budget (OCR là offline one-time cost);
5. index có thể ship và chạy lại không cần PaddleOCR/model weights.

Kết luận thực nghiệm: **không promote A1a hoặc A1b để thay A0**.

- A1a hòa A0 ở Recall@5/@10 và manual answer accuracy, nhưng Recall@1 và
  citation recall thấp hơn. Nó chỉ nên được giữ như candidate phụ cho fusion vì
  cứu được page 59 ở sq-07.
- A1b thua rõ ràng ở retrieval, answer và citation; không phù hợp với fixed
  character chunking hiện tại.
- Nếu buộc chọn một output PaddleOCR-VL cho bước Track A tiếp theo, chọn
  **A1a plain**. Thử nghiệm kế tiếp phải tách riêng ảnh hưởng của
  structure-aware/table-aware chunking; không được ghi nhận nó là chiến thắng
  của parser ở benchmark này.
- A0 vẫn là baseline bị khóa. Kết quả âm là một finding quan trọng: với PDF có
  text layer tốt và bộ eval lookup nhỏ, VLM OCR đắt hơn không tự động tạo RAG tốt
  hơn.

## Failure analysis cần đọc sau benchmark

- **OCR recognition error:** sai chữ/số/ký hiệu `%`, dấu thập phân, đơn vị.
- **Layout/reading-order error:** ghép sai cột hoặc heading vào bảng.
- **Representation loss (A1a):** flatten làm mất quan hệ header–cell.
- **Markup noise (A1b):** tag/pipe làm embedding hoặc prompt dài hơn mà không tăng signal.
- **Fixed-window boundary:** đúng OCR nhưng evidence bị cắt; lỗi này thuộc chunking/retrieval, không quy oan cho parser.
- **Retrieval miss:** gold page không vào top 10.
- **Generation/citation error:** gold page đã retrieve nhưng answer/citation sai.

## Commands benchmark

```bash
make benchmark-a1a
make benchmark-a1b
```

Mỗi run ghi JSONL từng câu và summary gồm Recall@1/5/10, answer/citation/refusal, p50/p95 latency, token usage, cost projection và failure taxonomy.

## Artifacts và checksum

- A1a index SHA-256: `afe37dcc35264e6e1aed3666968bf6c64883f7abcdef289fdcb27d77aac9b6ae`
- A1a chunks SHA-256: `f382018207cc9b0650dc914fd8080d719ff546beb847368418fdf7fc2a85fba8`
- A1b index SHA-256: `df2b98db6924689e42e5d9de914c346f70ad6080a850291c6b62b26afd2db103`
- A1b chunks SHA-256: `49f38bca51fbdaa57d3b93a1f2f0d5e93589fc80146c005f193af980a515e68a`
- Shared raw manifest SHA-256: `873faa472f4fc8e23d4cb4fa69a08d3730fe7c78d8e190d595cb62eb0ec56e27`

Machine-readable outputs nằm trong `data/evaluation/experiments/`, gồm two
JSONL runs, summaries, manual reviews, per-question comparison và ingestion
stats.

## Tài liệu upstream

- [PaddleOCR-VL 1.6 algorithm documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.en.md)
- [PaddleOCR-VL pipeline usage](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md)
- [PaddleX PaddleOCR-VL full-pipeline tutorial](https://github.com/PaddlePaddle/PaddleX/blob/release/3.7/docs/pipeline_usage/tutorials/ocr_pipelines/PaddleOCR-VL.en.md)
- [Official PaddleOCR-VL 1.6 GGUF artifacts](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/tree/main)
- [PaddleOCR-VL 1.6 technical report](https://arxiv.org/abs/2606.03264)
