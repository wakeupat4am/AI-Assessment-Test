# Baseline decisions

## A. Baseline này làm gì

Pipeline ingest đúng một Annual Report PDF, tách các spread thành trang in, giữ provenance, chunk từng trang, tạo multilingual dense embeddings, lưu FAISS index, rewrite follow-up bằng Qwen, trả lời từ top-k evidence và kiểm tra citation sau sinh.

## B. Những gì cố ý chưa làm

Không OCR, không layout/table reconstruction, không glossary extraction, không query expansion, không BM25/hybrid, không RRF, không reranker, không calculator, không query decomposition, không LLM judge, không fine-tuning, không web UI và không deployment. Các phần này sẽ làm baseline khó giải thích và che khuất nguyên nhân lỗi.

## C. Vì sao dense retrieval là baseline

Một multilingual embedding model cho phép hỏi tiếng Việt bằng cách diễn đạt khác tài liệu mà vẫn giữ pipeline nhỏ: một lần encode corpus, một `IndexFlatIP`, một lần encode query. Đây là điểm đối chứng rõ ràng cho các thử nghiệm retrieval sau.

## D. Vì sao chưa rerank

Reranker thêm model, latency và hyperparameter. Trước hết cần đo recall/grounding của retriever đơn; nếu không, không thể biết cải thiện đến từ candidate generation hay reranking.

## E. Vì sao chưa BM25/hybrid

BM25 có thể mạnh với mã chỉ tiêu và viết tắt, nhưng hybrid cần thêm index, normalization và fusion. Baseline dense trước tạo mốc đo sạch cho thí nghiệm BM25 + dense và Reciprocal Rank Fusion sau này.

## F. Vì sao không fine-tuning

Annual Report là nguồn bằng chứng thay đổi theo năm. Fine-tuning không thay thế provenance, có thể ghi nhớ dữ kiện cũ và không giải quyết citation. Qwen chỉ dùng cho hiểu ngôn ngữ/reasoning trên evidence.

## G. Citation theo trang in

PDF có 197 sheet nhưng 393 vùng trang in: sheet đầu là bìa, mỗi sheet landscape tiếp theo chứa trang trái/phải. Parser tách theo hình học, phát hiện số ở mép ngoài phần đầu/cuối trang, suy ra offset từ nhiều anchor đồng thuận và ghi `mapping_method`. Chunk giữ `pdf_page_index`, `pdf_page`, `page_part`, `printed_page`; chỉ `printed_page` được phép xuất dưới dạng `[tr. N]`. Mapping đầy đủ có thể audit và override thủ công.

Sau generation, regex Python trích `[tr. N]` và `[tr. N, M]`. Mọi số trang phải thuộc tập evidence đã đưa cho Qwen. Với câu trả lời có số liệu, validator còn yêu cầu số đó xuất hiện nguyên văn trong text của một trang được cite. Nếu sai/thiếu, hệ thống thử sửa một lần; nếu vẫn sai, trả warning an toàn và đánh dấu `citation_valid=false`. Cách kiểm tra này deterministic, không phải LLM judge.

## H. Cách refusal hiện tại hoạt động

Nếu top-1 cosine score thấp hơn `MIN_RETRIEVAL_SCORE`, code từ chối trước khi gọi Qwen. Nếu score qua ngưỡng, system prompt vẫn yêu cầu Qwen tự đánh giá đủ bằng chứng và dùng đúng câu từ chối. Đây là cơ chế baseline, chưa phải calibrated classifier.

## I. Failure modes đã biết

- Text extraction tuyến tính có thể làm méo bảng nhiều cột.
- Một số glyph trong PDF có encoding lỗi dù nhìn đúng trên trang.
- Page offset inference đúng với cấu trúc hiện tại nhưng cần re-audit nếu thay PDF.
- Dense-only có thể bỏ sót viết tắt/chỉ tiêu exact-match.
- Một global threshold không tối ưu cho mọi loại câu hỏi.
- Citation validator kiểm tra trang và sự xuất hiện nguyên văn của số liệu, nhưng chưa chứng minh quan hệ ngữ nghĩa giữa nhãn và giá trị; bảng nhiều cột bị làm phẳng vẫn có thể gây metric binding sai.
- Câu hỏi dùng tên chỉ tiêu mơ hồ (ví dụ không nói rõ tỷ lệ hay số dư) có thể cần người dùng làm rõ; baseline không có glossary/intent classifier.
- Query rewrite có thể làm lệch ý nếu lịch sử dài hoặc mơ hồ.
- Qwen adapter baseline xử lý tuần tự và không phải production server.

## J. Track A nên làm tiếp

Sau khi có số đo baseline: layout-aware parsing, table-preserving extraction, glossary extraction và phát hiện trang in tốt hơn. Mỗi thay đổi phải được đánh giá riêng để biết tác động.

## K. Track B nên làm tiếp

Thử theo thứ tự: glossary-aware query expansion; BM25 + dense hybrid; Reciprocal Rank Fusion; reranking; calculator tool; query decomposition. Không phần nào trong danh sách này được cài ở baseline hiện tại.

## Thí nghiệm kế tiếp được đề xuất

So sánh A/B cùng bộ câu hỏi và cùng answer prompt:

- Baseline: dense retrieval với query gốc/standalone.
- Improvement: mở rộng query bằng glossary cho các viết tắt như CASA, CIBG, RBG, N/N trước dense retrieval.

Đo retrieval hit/recall theo trang vàng, citation validity, refusal và latency. Chỉ triển khai khi được phê duyệt.
