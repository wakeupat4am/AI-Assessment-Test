# A3.1 public answer comparison — B0 vs B1a vs B1b vs B2

> All systems use the same 10 public questions and the unchanged answer-generation layer. Automatic answer accuracy is diagnostic, not exact-match accuracy.

## Full metrics

| System | Auto answer accuracy | Hit@1 | Hit@5 | Hit@10 | Recall@5 | Recall@10 | Citation P | Citation R | Citation valid | Refusal | p50 s | p95 s | LLM tokens | Cost USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.4000 | 0.5556 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0.6667 | 1.0000 | 1.0000 | 5.282 | 7.420 | 36077 | 0.000000 |
| B1a | 0.4000 | 0.5556 | 0.7778 | 0.8889 | 0.7778 | 0.8889 | 0.6667 | 0.6667 | 0.9000 | 0.9000 | 5.340 | 8.343 | 42271 | 0.000000 |
| B1b | 0.3000 | 0.5556 | 0.7778 | 0.7778 | 0.7778 | 0.7778 | 0.6667 | 0.6667 | 0.9000 | 0.8000 | 8.394 | 10.869 | 45981 | 0.000000 |
| B2 | 0.3000 | 0.2222 | 0.6667 | 0.8889 | 0.6667 | 0.8889 | 0.5000 | 0.5556 | 1.0000 | 0.9000 | 5.089 | 7.483 | 33344 | 0.000000 |

## Outputs for the 10 public questions

### sq-01 — Tính đến năm 2025, Techcombank có bao nhiêu chi nhánh và phòng giao dịch, và hiện diện tại bao nhiêu tỉnh thành?

Gold: 302 chi nhánh và phòng giao dịch, tại 29 trên 34 tỉnh thành trên toàn quốc.

Gold pages: 4

- B0: Tính đến năm 2025, Techcombank có 302 chi nhánh và phòng giao dịch hiện diện tại 29 tỉnh thành trên toàn quốc [tr. 27].
  - citations=27; automatic_score=False

- B1a: Tính đến năm 2025, Techcombank có 302 chi nhánh và phòng giao dịch hiện diện tại 29 tỉnh, thành phố [tr. 27].
  - citations=27; automatic_score=False

- B1b: Tính đến năm 2025, Techcombank có 302 chi nhánh và phòng giao dịch hiện diện tại 29 tỉnh, thành phố [tr. 27].
  - citations=27; automatic_score=False

- B2: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=False; rounds=2; stop=max_rounds_reached_insufficient

### sq-02 — Năm 2025, Techcombank được S&P Global Ratings và Fitch Ratings xếp hạng tín nhiệm ở mức nào?

Gold: S&P Global Ratings nâng xếp hạng lên mức BB với triển vọng Ổn định. Fitch Ratings lần đầu tiên xếp hạng tín nhiệm dài hạn ở mức BB- với triển vọng Tích cực.

Gold pages: 4

- B0: Năm 2025, Techcombank được S&P Global Ratings xếp hạng tín nhiệm ở mức BB và Fitch Ratings xếp hạng tín nhiệm dài hạn ở mức BB- [tr. 43].
  - citations=43; automatic_score=True

- B1a: Năm 2025, Techcombank được S&P Global Ratings xếp hạng tín nhiệm ở mức BB và Fitch Ratings xếp hạng tín nhiệm dài hạn ở mức BB- [tr. 43].
  - citations=43; automatic_score=True

- B1b: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=False

- B2: Năm 2025, Techcombank được S&P Global Ratings xếp hạng tín nhiệm ở mức BB và Fitch Ratings xếp hạng tín nhiệm dài hạn ở mức BB- [tr. 43].
  - citations=43; automatic_score=True; rounds=1; stop=sufficient_after_round_1

### sq-03 — Tổng tài sản của Techcombank tại ngày 31/12/2025 là bao nhiêu?

Gold: 1.192 nghìn tỷ đồng, tăng 21,8% so với năm trước.

Gold pages: 5

- B0: Tổng tài sản của Techcombank tại ngày 31/12/2025 là 1.192 nghìn tỷ đồng [tr. 5].
  - citations=5; automatic_score=False

- B1a: Tổng tài sản của Techcombank tại ngày 31/12/2025 là 1.192 nghìn tỷ đồng [tr. 5].
  - citations=5; automatic_score=False

- B1b: Tổng tài sản của Techcombank tại ngày 31/12/2025 là 1.192 nghìn tỷ đồng [tr. 5].
  - citations=5; automatic_score=False

- B2: Tổng tài sản của Techcombank tại ngày 31/12/2025 là 224.331.619 triệu đồng [tr. 379].
  - citations=379; automatic_score=False; rounds=1; stop=sufficient_after_round_1

### sq-04 — Tỷ lệ CASA của Techcombank năm 2025 là bao nhiêu?

Gold: 40,4%, dẫn đầu ngành.

Gold pages: 5

- B0: Tỷ lệ CASA của Techcombank năm 2025 là 40,4% [tr. 53].
  - citations=53; automatic_score=False

- B1a: Không thể tạo câu trả lời với trích dẫn hợp lệ từ bằng chứng được truy xuất.
  - citations=none; automatic_score=False

- B1b: Không thể tạo câu trả lời với trích dẫn hợp lệ từ bằng chứng được truy xuất.
  - citations=none; automatic_score=False

- B2: Tỷ lệ tiền gửi không kỳ hạn (CASA) của Techcombank năm 2025 là 16,6% so với năm 2024 [tr. 121].
  - citations=121; automatic_score=False; rounds=1; stop=sufficient_after_round_1

### sq-05 — Tỷ lệ nợ xấu của Techcombank năm 2025 là bao nhiêu?

Gold: 1,13%, thuộc nhóm chất lượng tài sản tốt nhất thị trường.

Gold pages: 5

- B0: Tỷ lệ nợ xấu của Techcombank năm 2025 là 1,13% [tr. 5].
  - citations=5; automatic_score=False

- B1a: Tỷ lệ nợ xấu của Techcombank năm 2025 là 1,13% [tr. 5].
  - citations=5; automatic_score=False

- B1b: Tỷ lệ nợ xấu của Techcombank năm 2025 là 1,13% [tr. 5].
  - citations=5; automatic_score=False

- B2: Tỷ lệ nợ xấu của Techcombank năm 2025 là 1,13% [tr. 5].
  - citations=5; automatic_score=False; rounds=1; stop=sufficient_after_round_1

### sq-06 — Tổng thu nhập hoạt động năm 2025 của Techcombank là bao nhiêu, và tốc độ tăng trưởng kép giai đoạn 2018–2025 của chỉ tiêu này là bao nhiêu?

Gold: 53,4 nghìn tỷ đồng, với tốc độ tăng trưởng kép giai đoạn 2018–2025 là 16,5%.

Gold pages: 5

- B0: Tổng thu nhập hoạt động của Techcombank năm 2025 là 53,4 nghìn tỷ đồng với tốc độ tăng trưởng kép giai đoạn 2018–2025 đạt 16,5% [tr. 5].
  - citations=5; automatic_score=True

- B1a: Tổng thu nhập hoạt động năm 2025 của Techcombank là 53,4 nghìn tỷ đồng với tốc độ tăng trưởng kép giai đoạn 2018–2025 đạt 16,5% [tr. 5].
  - citations=5; automatic_score=True

- B1b: Tổng thu nhập hoạt động năm 2025 của Techcombank là 53,4 nghìn tỷ đồng với tốc độ tăng trưởng kép giai đoạn 2018–2025 đạt 16,5% [tr. 5].
  - citations=5; automatic_score=True

- B2: Tổng thu nhập hoạt động của Techcombank năm 2025 là 53.391.125 triệu đồng [tr. 300], với tốc độ tăng trưởng kép giai đoạn 2018–2025 đạt 16,5% [tr. 5].
  - citations=300, 5; automatic_score=False; rounds=1; stop=sufficient_after_round_1

### sq-07 — Dư nợ vay của Khối Ngân hàng Bán lẻ năm 2025 là bao nhiêu và tăng bao nhiêu phần trăm so với năm trước?

Gold: 328,1 nghìn tỷ đồng, tăng 26,9% so với năm trước.

Gold pages: 59

- B0: Dư nợ vay của Khối Ngân hàng Bán lẻ năm 2025 là 328,1 Nghìn tỷ đồng, tăng 26,9% so với năm trước [tr. 59].
  - citations=59; automatic_score=True

- B1a: Dư nợ vay của Khối Ngân hàng Bán lẻ năm 2025 là 328,1 Nghìn tỷ đồng [tr. 59] và tăng 26,9% so với năm trước [tr. 59].
  - citations=59; automatic_score=True

- B1b: Dư nợ vay của Khối Ngân hàng Bán lẻ năm 2025 là 328,1 Nghìn tỷ đồng [tr. 59] và tăng 26,9% so với năm trước [tr. 59].
  - citations=59; automatic_score=True

- B2: Dư nợ vay của Khối Ngân hàng Bán lẻ năm 2025 là 328,1 Nghìn tỷ đồng, tăng 26,9% so với năm trước [tr. 59].
  - citations=59; automatic_score=True; rounds=1; stop=sufficient_after_round_1

### sq-08 — Theo danh mục thuật ngữ viết tắt của báo cáo, RBG là viết tắt của khối nào?

Gold: Khối Ngân hàng Bán lẻ.

Gold pages: 387

- B0: Theo danh mục thuật ngữ viết tắt trong Báo cáo thường niên năm 2025, RBG là viết tắt của "Khối Ngân hàng Bán lẻ" [tr. 387].
  - citations=387; automatic_score=False

- B1a: Theo danh mục thuật ngữ viết tắt của Báo cáo thường niên Techcombank năm 2025, RBG là viết tắt của Khối Ngân hàng Bán lẻ [tr. 387].
  - citations=387; automatic_score=False

- B1b: Theo danh mục thuật ngữ viết tắt của Báo cáo thường niên Techcombank năm 2025, RBG là viết tắt của Khối Ngân hàng Bán lẻ [tr. 387].
  - citations=387; automatic_score=False

- B2: Theo danh mục thuật ngữ viết tắt trong Báo cáo thường niên năm 2025, RBG là viết tắt của "Khối Ngân hàng Bán lẻ" [tr. 387].
  - citations=387; automatic_score=False; rounds=1; stop=sufficient_after_round_1

### sq-09 — Theo danh mục thuật ngữ viết tắt của báo cáo, CASA là viết tắt của thuật ngữ gì?

Gold: Tiền gửi không kỳ hạn.

Gold pages: 386

- B0: Theo danh mục thuật ngữ viết tắt của Báo cáo thường niên Techcombank năm 2025, CASA là viết tắt của "Tiền gửi không kỳ hạn" [tr. 386].
  - citations=386; automatic_score=False

- B1a: Theo danh mục thuật ngữ viết tắt của Báo cáo thường niên Techcombank năm 2025, CASA là viết tắt của "Tiền gửi không kỳ hạn" [tr. 386].
  - citations=386; automatic_score=False

- B1b: Theo danh mục thuật ngữ viết tắt của Báo cáo thường niên Techcombank năm 2025, CASA là viết tắt của "Tiền gửi không kỳ hạn" [tr. 386].
  - citations=386; automatic_score=False

- B2: Theo danh mục thuật ngữ viết tắt trong Báo cáo thường niên, CASA là viết tắt của "Tiền gửi không kỳ hạn" [tr. 386].
  - citations=386; automatic_score=False; rounds=1; stop=sufficient_after_round_1

### sq-10 — Techcombank dự báo lợi nhuận trước thuế năm 2027 là bao nhiêu?

Gold: Không có trong báo cáo. Báo cáo thường niên 2025 không đưa ra dự báo lợi nhuận cho năm 2027; hệ thống cần từ chối trả lời thay vì suy đoán.

Gold pages: none

- B0: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=True

- B1a: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=True

- B1b: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=True

- B2: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=True; rounds=2; stop=max_rounds_reached_insufficient
