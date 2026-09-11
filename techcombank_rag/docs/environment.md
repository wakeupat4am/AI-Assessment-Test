# Server environment inspection

Kiểm tra ngày 07/09/2026 trên `cecs_server_1`, trước khi tạo code/cài dependency.

## Hệ thống

- SSH user yêu cầu: `24anh.nvd`; shell thực tế báo `whoami=ubuntu`.
- Working directory ban đầu: `/home/ubuntu`.
- Kernel: Linux 5.15.0-190-generic x86_64.
- Python: 3.11.5 tại `/home/ubuntu/miniconda3/bin/python3`.
- CPU: 2 × Intel Xeon Gold 6242, 64 logical CPU.
- RAM: 251 GiB tổng, khoảng 230 GiB available lúc kiểm tra; swap 8 GiB.
- Filesystem `/home/ubuntu`: 6.5 TB, khoảng 1.2 TB available (83% used).

## GPU lúc kiểm tra

6 × NVIDIA RTX A5000, mỗi GPU 24,564 MiB; driver 535.309.01, CUDA 12.2.

- GPU 0: 19,251 MiB, 100% utilization.
- GPU 1: 9,632 MiB, 66% utilization.
- GPU 2–5: khoảng 12 MiB, 0% utilization.

Các PID GPU đang chạy thuộc namespace/user khác nên `ps` không nhìn thấy. Baseline mặc định chọn GPU 2–3 và không chạm tiến trình trên GPU 0–1.

## Qwen inspection

- Không có process khớp Qwen/vLLM/Ollama/llama.cpp và không có listening endpoint thấy được dưới user hiện tại.
- `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B` chỉ 12 KB (ref, không có weights), không phải model hoàn chỉnh.
- Model hoàn chỉnh do người dùng chỉ ra:
  `/home/ubuntu/llms-conformity-rq1/multimodal_rq1/models/Qwen3.5-9B`.
- Model gồm 4 safetensors shard và tokenizer/config, tổng khoảng 19 GB.
- Môi trường chạy đã được project đó dùng:
  - Python: `/home/ubuntu/llms-conformity-rq1/multimodal_rq1/.venv-transformers/bin/python`
  - Extra dependencies: `/home/ubuntu/llms-conformity-rq1/multimodal_rq1/.deps-qwen35`
  - Torch 2.5.1+cu121
  - Transformers 5.16.1
- Một vLLM 0.5.4 cũ tồn tại trong `server_experiments/.venv`, nhưng quá cũ cho Qwen3.5 và không được dùng.
- Serving được chọn: adapter HTTP OpenAI-compatible nhỏ trong `src/llm/qwen_server.py`, dùng Transformers pipeline và model path local. API: `http://127.0.0.1:8000/v1`.

## Python packages trước khi cài

Base Conda không có `torch`, `transformers`, `vllm`, `fitz`, `faiss`, `sentence_transformers`, `openai` hoặc `python-dotenv`. Vì vậy dependency RAG được cài trong `/home/ubuntu/techcombank_rag/.venv`; môi trường chung không bị sửa.

## Data inspection

Báo cáo không có sẵn trong `/home/ubuntu` ở lần tìm đầu. Bản tiếng Việt chính thức, cập nhật, được lấy từ website Techcombank và đặt tại `data/raw/techcombank-bao-cao-thuong-nien-2025-vie-update.pdf` (khoảng 27 MB, 197 PDF sheets).

## Inference smoke test và đo thời gian

Đã chạy adapter bằng `nohup ./scripts/serve_qwen.sh > qwen_server.log 2>&1 &`. Endpoint `/v1/models` trả `Qwen/Qwen3.5-9B`; `QwenClient.health_check()` trả `OK`.

- Qwen model load: khoảng 25 giây.
- Health check + generation `OK`: khoảng 2.0 giây.
- GPU sau khi load ở lần đo cuối: GPU 2 khoảng 9,882 MiB; GPU 3 khoảng 12,824 MiB.
- Ingestion: 79.981 giây được đo bên trong pipeline; 87.396 giây wall time ở lần đầu, gồm tải/cache embedding model.
- Corpus: 197 PDF sheet → 393 printed-page fragment → 446 chunk, vector dimension 384.
- Page mapping: 380/393 fragment phát hiện số trực tiếp; 13 suy ra từ dominant offset `+1`; 0 override.
- Single-turn batch CASA: 6.743 giây, gồm một lần deterministic validation phát hiện citation chưa chứa số và yêu cầu Qwen sửa.
- Single-turn lợi nhuận trước thuế: 3.233 giây.
- Multi-turn: 6.770 giây cho lượt đầu; 4.219 giây cho rewrite `2024` + grounded refusal.
- Grounded refusal ngoài miền: 2.896 giây.
- Unit test: 10 test pass trong 6.48 giây; test không tải model nặng.

Latency là phép đo warm endpoint trên tải máy tại thời điểm kiểm tra, không phải benchmark throughput. Mỗi process CLI vẫn tải embedding model từ local cache một lần khi khởi tạo retriever.

## Smoke-test output

```text
Tỷ lệ CASA của Techcombank năm 2025 là 40,4% [tr. 5].
Lợi nhuận trước thuế năm 2025 là 32.538 tỷ đồng [tr. 257].
Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
```

Ba output trên đều có `citation_valid=true`; câu thứ ba có `refused=true`.
