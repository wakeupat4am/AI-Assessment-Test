# A3.1 public answer comparison — B0 vs HyDE variants

> All systems use the same 10 public questions and the unchanged answer-generation layer. Automatic answer accuracy is diagnostic, not exact-match accuracy.

## Full metrics

| System | Auto answer accuracy | Hit@1 | Hit@5 | Hit@10 | Recall@5 | Recall@10 | Citation P | Citation R | Citation valid | Refusal | p50 s | p95 s | LLM tokens | Cost USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 (one-shot) | 0.4000 | 0.5556 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0.6667 | 1.0000 | 1.0000 | 5.007 | 6.536 | 36077 | 0.000000 |
| B3a (HyDE-only) | 0.4000 | 0.4444 | 0.7778 | 0.8889 | 0.7778 | 0.8889 | 0.5556 | 0.5556 | 1.0000 | 1.0000 | 15.086 | 17.659 | 33399 | 0.000000 |
| B3b (query + HyDE fusion) | 0.4000 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0.6667 | 0.9000 | 0.9000 | 13.453 | 19.033 | 41681 | 0.000000 |
| B3c (safe fusion) | 0.4000 | 0.5556 | 0.6667 | 1.0000 | 0.6667 | 1.0000 | 0.5556 | 0.5556 | 1.0000 | 1.0000 | 12.562 | 17.474 | 40085 | 0.000000 |

## Outputs for the 10 public questions

### sq-01 — Tính đến năm 2025, Techcombank có bao nhiêu chi nhánh và phòng giao dịch, và hiện diện tại bao nhiêu tỉnh thành?

Gold: 302 chi nhánh và phòng giao dịch, tại 29 trên 34 tỉnh thành trên toàn quốc.

Gold pages: 4

- B0 (one-shot): Tính đến năm 2025, Techcombank có 302 chi nhánh và phòng giao dịch hiện diện tại 29 tỉnh thành trên toàn quốc [tr. 27].
  - citations=27; automatic_score=False

- B3a (HyDE-only): Tính đến năm 2025, Techcombank có 302 chi nhánh và phòng giao dịch trên toàn quốc [tr. 27]. Hiện tại, ngân hàng này hiện diện tại 29 tỉnh, thành phố [tr. 27].
  - citations=27; automatic_score=False

- B3b (query + HyDE fusion): Không thể tạo câu trả lời với trích dẫn hợp lệ từ bằng chứng được truy xuất.
  - citations=none; automatic_score=False

- B3c (safe fusion): Tính đến năm 2025, Techcombank có 302 chi nhánh và phòng giao dịch trên toàn quốc [tr. 27]. Hiện tại, ngân hàng này hiện diện tại 29 tỉnh, thành phố [tr. 27].
  - citations=27; automatic_score=False

### sq-02 — Năm 2025, Techcombank được S&P Global Ratings và Fitch Ratings xếp hạng tín nhiệm ở mức nào?

Gold: S&P Global Ratings nâng xếp hạng lên mức BB với triển vọng Ổn định. Fitch Ratings lần đầu tiên xếp hạng tín nhiệm dài hạn ở mức BB- với triển vọng Tích cực.

Gold pages: 4

- B0 (one-shot): Năm 2025, Techcombank được S&P Global Ratings xếp hạng tín nhiệm ở mức BB và Fitch Ratings xếp hạng tín nhiệm dài hạn ở mức BB- [tr. 43].
  - citations=43; automatic_score=True

- B3a (HyDE-only): Năm 2025, Techcombank được S&P Global Ratings xếp hạng tín nhiệm ở mức BB với triển vọng Ổn định và được Fitch Ratings xếp hạng tín nhiệm dài hạn ở mức BB- với triển vọng "Tích cực" [tr. 43].
  - citations=43; automatic_score=True

- B3b (query + HyDE fusion): Năm 2025, Techcombank được S&P Global Ratings xếp hạng tín nhiệm ở mức BB với triển vọng Ổn định và được Fitch Ratings xếp hạng tín nhiệm dài hạn ở mức BB- với triển vọng "Tích cực" [tr. 43].
  - citations=43; automatic_score=True

- B3c (safe fusion): Năm 2025, Techcombank được S&P Global Ratings xếp hạng tín nhiệm ở mức BB với triển vọng Ổn định và được Fitch Ratings xếp hạng tín nhiệm dài hạn ở mức BB- với triển vọng "Tích cực" [tr. 43].
  - citations=43; automatic_score=True

### sq-03 — Tổng tài sản của Techcombank tại ngày 31/12/2025 là bao nhiêu?

Gold: 1.192 nghìn tỷ đồng, tăng 21,8% so với năm trước.

Gold pages: 5

- B0 (one-shot): Tổng tài sản của Techcombank tại ngày 31/12/2025 là 1.192 nghìn tỷ đồng [tr. 5].
  - citations=5; automatic_score=False

- B3a (HyDE-only): Tổng tài sản của Techcombank tại ngày 31/12/2025 là 1.192 nghìn tỷ đồng [tr. 5].
  - citations=5; automatic_score=False

- B3b (query + HyDE fusion): Tổng tài sản của Techcombank tại ngày 31/12/2025 là 1.192 nghìn tỷ đồng [tr. 5].
  - citations=5; automatic_score=False

- B3c (safe fusion): Tổng tài sản của Techcombank tại ngày 31/12/2025 là 224.331.619 triệu đồng [tr. 379].
  - citations=379; automatic_score=False

### sq-04 — Tỷ lệ CASA của Techcombank năm 2025 là bao nhiêu?

Gold: 40,4%, dẫn đầu ngành.

Gold pages: 5

- B0 (one-shot): Tỷ lệ CASA của Techcombank năm 2025 là 40,4% [tr. 53].
  - citations=53; automatic_score=False

- B3a (HyDE-only): Tỷ lệ CASA của Techcombank năm 2025 là 40,4% [tr. 48].
  - citations=48; automatic_score=False

- B3b (query + HyDE fusion): Tỷ lệ CASA của Techcombank năm 2025 là 40,4% [tr. 48].
  - citations=48; automatic_score=False

- B3c (safe fusion): Tỷ lệ CASA của Techcombank năm 2025 là 40,4% [tr. 48].
  - citations=48; automatic_score=False

### sq-05 — Tỷ lệ nợ xấu của Techcombank năm 2025 là bao nhiêu?

Gold: 1,13%, thuộc nhóm chất lượng tài sản tốt nhất thị trường.

Gold pages: 5

- B0 (one-shot): Tỷ lệ nợ xấu của Techcombank năm 2025 là 1,13% [tr. 5].
  - citations=5; automatic_score=False

- B3a (HyDE-only): Tỷ lệ nợ xấu của Techcombank năm 2025 là 1,13% [tr. 48].
  - citations=48; automatic_score=False

- B3b (query + HyDE fusion): Tỷ lệ nợ xấu của Techcombank năm 2025 là 1,13% [tr. 5].
  - citations=5; automatic_score=False

- B3c (safe fusion): Tỷ lệ nợ xấu của Techcombank năm 2025 là 1,13% [tr. 5].
  - citations=5; automatic_score=False

### sq-06 — Tổng thu nhập hoạt động năm 2025 của Techcombank là bao nhiêu, và tốc độ tăng trưởng kép giai đoạn 2018–2025 của chỉ tiêu này là bao nhiêu?

Gold: 53,4 nghìn tỷ đồng, với tốc độ tăng trưởng kép giai đoạn 2018–2025 là 16,5%.

Gold pages: 5

- B0 (one-shot): Tổng thu nhập hoạt động của Techcombank năm 2025 là 53,4 nghìn tỷ đồng với tốc độ tăng trưởng kép giai đoạn 2018–2025 đạt 16,5% [tr. 5].
  - citations=5; automatic_score=True

- B3a (HyDE-only): Tổng thu nhập hoạt động năm 2025 của Techcombank là 53,4 nghìn tỷ đồng với tốc độ tăng trưởng kép giai đoạn 2018–2025 là 16,5% [tr. 5].
  - citations=5; automatic_score=True

- B3b (query + HyDE fusion): Tổng thu nhập hoạt động năm 2025 của Techcombank là 53,4 nghìn tỷ đồng với tốc độ tăng trưởng kép giai đoạn 2018–2025 đạt 16,5% [tr. 5].
  - citations=5; automatic_score=True

- B3c (safe fusion): Tổng thu nhập hoạt động năm 2025 của Techcombank là 53,4 nghìn tỷ đồng với tốc độ tăng trưởng kép giai đoạn 2018–2025 là 16,5% [tr. 5].
  - citations=5; automatic_score=True

### sq-07 — Dư nợ vay của Khối Ngân hàng Bán lẻ năm 2025 là bao nhiêu và tăng bao nhiêu phần trăm so với năm trước?

Gold: 328,1 nghìn tỷ đồng, tăng 26,9% so với năm trước.

Gold pages: 59

- B0 (one-shot): Dư nợ vay của Khối Ngân hàng Bán lẻ năm 2025 là 328,1 Nghìn tỷ đồng, tăng 26,9% so với năm trước [tr. 59].
  - citations=59; automatic_score=True

- B3a (HyDE-only): Dư nợ vay của Khối Ngân hàng Bán lẻ năm 2025 là 328,1 Nghìn tỷ đồng [tr. 59], tăng 26,9% so với năm trước [tr. 59].
  - citations=59; automatic_score=True

- B3b (query + HyDE fusion): Dư nợ vay của Khối Ngân hàng Bán lẻ năm 2025 là 328,1 Nghìn tỷ đồng, tăng 26,9% so với năm trước [tr. 59].
  - citations=59; automatic_score=True

- B3c (safe fusion): Dư nợ vay của Khối Ngân hàng Bán lẻ năm 2025 là 328,1 Nghìn tỷ đồng và tăng 26,9% so với năm trước [tr. 59].
  - citations=59; automatic_score=True

### sq-08 — Theo danh mục thuật ngữ viết tắt của báo cáo, RBG là viết tắt của khối nào?

Gold: Khối Ngân hàng Bán lẻ.

Gold pages: 387

- B0 (one-shot): Theo danh mục thuật ngữ viết tắt trong Báo cáo thường niên năm 2025, RBG là viết tắt của "Khối Ngân hàng Bán lẻ" [tr. 387].
  - citations=387; automatic_score=False

- B3a (HyDE-only): Theo danh mục thuật ngữ viết tắt của Báo cáo thường niên Techcombank năm 2025, RBG là viết tắt của "Khối Ngân hàng Bán lẻ" [tr. 387].
  - citations=387; automatic_score=False

- B3b (query + HyDE fusion): Theo danh mục thuật ngữ viết tắt của Báo cáo thường niên năm 2025, RBG là viết tắt của "Khối Ngân hàng Bán lẻ" [tr. 387].
  - citations=387; automatic_score=False

- B3c (safe fusion): Theo danh mục thuật ngữ viết tắt của Báo cáo thường niên Techcombank năm 2025, RBG là viết tắt của Khối Ngân hàng Bán lẻ [tr. 387].
  - citations=387; automatic_score=False

### sq-09 — Theo danh mục thuật ngữ viết tắt của báo cáo, CASA là viết tắt của thuật ngữ gì?

Gold: Tiền gửi không kỳ hạn.

Gold pages: 386

- B0 (one-shot): Theo danh mục thuật ngữ viết tắt của Báo cáo thường niên Techcombank năm 2025, CASA là viết tắt của "Tiền gửi không kỳ hạn" [tr. 386].
  - citations=386; automatic_score=False

- B3a (HyDE-only): Theo danh mục thuật ngữ viết tắt trong Báo cáo thường niên, CASA là viết tắt của "Tiền gửi không kỳ hạn" [tr. 386].
  - citations=386; automatic_score=False

- B3b (query + HyDE fusion): Theo danh mục thuật ngữ viết tắt trong Báo cáo thường niên, CASA là viết tắt của "Tiền gửi không kỳ hạn" [tr. 386].
  - citations=386; automatic_score=False

- B3c (safe fusion): Theo danh mục thuật ngữ viết tắt của Báo cáo thường niên Techcombank năm 2025, CASA là viết tắt của "Tiền gửi không kỳ hạn" [tr. 386].
  - citations=386; automatic_score=False

### sq-10 — Techcombank dự báo lợi nhuận trước thuế năm 2027 là bao nhiêu?

Gold: Không có trong báo cáo. Báo cáo thường niên 2025 không đưa ra dự báo lợi nhuận cho năm 2027; hệ thống cần từ chối trả lời thay vì suy đoán.

Gold pages: none

- B0 (one-shot): Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=True

- B3a (HyDE-only): Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=True

- B3b (query + HyDE fusion): Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=True

- B3c (safe fusion): Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=True
