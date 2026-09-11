# Public answer comparison — B1a vs B1b vs B2

> All systems use the same 10 public questions and the unchanged answer-generation layer. Automatic answer accuracy is diagnostic, not exact-match accuracy.

## Full metrics

| System | Auto answer accuracy | Hit@1 | Hit@5 | Hit@10 | Recall@5 | Recall@10 | Citation P | Citation R | Citation valid | Refusal | p50 s | p95 s | LLM tokens | Cost USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1a | 0.3000 | 0.3333 | 0.4444 | 0.6667 | 0.4444 | 0.6667 | 0.3333 | 0.3333 | 1.0000 | 0.8000 | 4.829 | 10.113 | 34035 | 0.000000 |
| B1b | 0.3000 | 0.3333 | 0.4444 | 0.6667 | 0.4444 | 0.6667 | 0.3333 | 0.3333 | 1.0000 | 0.8000 | 7.719 | 11.352 | 37488 | 0.000000 |
| B2 | 0.3000 | 0.3333 | 0.3333 | 0.7778 | 0.3333 | 0.7778 | 0.3333 | 0.3333 | 1.0000 | 0.7000 | 3.956 | 7.301 | 27218 | 0.000000 |

## Outputs for the 10 public questions

### sq-01 — Tính đến năm 2025, Techcombank có bao nhiêu chi nhánh và phòng giao dịch, và hiện diện tại bao nhiêu tỉnh thành?

Gold: 302 chi nhánh và phòng giao dịch, tại 29 trên 34 tỉnh thành trên toàn quốc.

Gold pages: 4

- B1a: Tính đến năm 2025, Techcombank có 302 chi nhánh và phòng giao dịch hiện diện tại 29 trên 34 tỉnh thành trên toàn quốc [tr. 4].
  - citations=4; automatic_score=True

- B1b: Tính đến năm 2025, Techcombank có 302 chi nhánh và phòng giao dịch hiện diện tại 29 trên 34 tỉnh thành trên toàn quốc [tr. 4].
  - citations=4; automatic_score=True

- B2: Tính đến năm 2025, Techcombank có mạng lưới hoạt động rộng khắp với 302 chi nhánh và phòng giao dịch tại 29 trên 34 tỉnh thành trên toàn quốc [tr. 4].
  - citations=4; automatic_score=True; rounds=1; stop=sufficient_after_round_1

### sq-02 — Năm 2025, Techcombank được S&P Global Ratings và Fitch Ratings xếp hạng tín nhiệm ở mức nào?

Gold: S&P Global Ratings nâng xếp hạng lên mức BB với triển vọng Ổn định. Fitch Ratings lần đầu tiên xếp hạng tín nhiệm dài hạn ở mức BB- với triển vọng Tích cực.

Gold pages: 4

- B1a: Năm 2025, Techcombank được S&P Global Ratings xếp hạng tín nhiệm ở mức BB với triển vọng Ổn định và được Fitch Ratings xếp hạng tín nhiệm dài hạn ở mức BB- với triển vọng "Tích cực" [tr. 4].
  - citations=4; automatic_score=True

- B1b: Năm 2025, Techcombank được S&P Global Ratings xếp hạng tín nhiệm ở mức BB với triển vọng Ổn định và được Fitch Ratings xếp hạng tín nhiệm dài hạn ở mức BB- với triển vọng "Tích cực" [tr. 4].
  - citations=4; automatic_score=True

- B2: Năm 2025, Techcombank được S&P Global Ratings nâng xếp hạng tín nhiệm lên mức BB với triển vọng Ổn định và lần đầu tiên được Fitch Ratings xếp hạng tín nhiệm dài hạn ở mức BB- với triển vọng "Tích cực" [tr. 4].
  - citations=4; automatic_score=True; rounds=1; stop=sufficient_after_round_1

### sq-03 — Tổng tài sản của Techcombank tại ngày 31/12/2025 là bao nhiêu?

Gold: 1.192 nghìn tỷ đồng, tăng 21,8% so với năm trước.

Gold pages: 5

- B1a: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=False

- B1b: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=False

- B2: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=False; rounds=1; stop=sufficient_after_round_1

### sq-04 — Tỷ lệ CASA của Techcombank năm 2025 là bao nhiêu?

Gold: 40,4%, dẫn đầu ngành.

Gold pages: 5

- B1a: Tỷ lệ CASA của Techcombank năm 2025 là 40,4% [tr. 49].
  - citations=49; automatic_score=False

- B1b: Tỷ lệ CASA của Techcombank năm 2025 là 40,4% [tr. 49].
  - citations=49; automatic_score=False

- B2: Tỷ lệ CASA của Techcombank năm 2025 là 40,4% [tr. 49].
  - citations=49; automatic_score=False; rounds=1; stop=sufficient_after_round_1

### sq-05 — Tỷ lệ nợ xấu của Techcombank năm 2025 là bao nhiêu?

Gold: 1,13%, thuộc nhóm chất lượng tài sản tốt nhất thị trường.

Gold pages: 5

- B1a: Tỷ lệ nợ xấu (NPL) của Techcombank năm 2025 là 1,13% [tr. 15].
  - citations=15; automatic_score=False

- B1b: Tỷ lệ nợ xấu (NPL) của Techcombank năm 2025 là 1,13% [tr. 15].
  - citations=15; automatic_score=False

- B2: Tỷ lệ nợ xấu (NPL) của Techcombank năm 2025 là 1,13% [tr. 15].
  - citations=15; automatic_score=False; rounds=1; stop=sufficient_after_round_1

### sq-06 — Tổng thu nhập hoạt động năm 2025 của Techcombank là bao nhiêu, và tốc độ tăng trưởng kép giai đoạn 2018–2025 của chỉ tiêu này là bao nhiêu?

Gold: 53,4 nghìn tỷ đồng, với tốc độ tăng trưởng kép giai đoạn 2018–2025 là 16,5%.

Gold pages: 5

- B1a: Tổng thu nhập hoạt động năm 2025 của Techcombank là 28,3 nghìn tỷ đồng với tốc độ tăng trưởng 12,8% so với cùng kỳ năm trước [tr. 59]. Báo cáo thường niên không cung cấp thông tin về tốc độ tăng trưởng kép giai đoạn 2018–2025 của chỉ tiêu này.
  - citations=59; automatic_score=False

- B1b: Tổng thu nhập hoạt động năm 2025 của Techcombank là 28,3 nghìn tỷ đồng với tốc độ tăng trưởng 12,8% so với cùng kỳ năm trước [tr. 59]. Báo cáo thường niên không cung cấp thông tin về tốc độ tăng trưởng kép giai đoạn 2018–2025 của chỉ tiêu này.
  - citations=59; automatic_score=False

- B2: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=False; rounds=2; stop=max_rounds_reached_insufficient

### sq-07 — Dư nợ vay của Khối Ngân hàng Bán lẻ năm 2025 là bao nhiêu và tăng bao nhiêu phần trăm so với năm trước?

Gold: 328,1 nghìn tỷ đồng, tăng 26,9% so với năm trước.

Gold pages: 59

- B1a: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=False

- B1b: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=False

- B2: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=False; rounds=1; stop=sufficient_after_round_1

### sq-08 — Theo danh mục thuật ngữ viết tắt của báo cáo, RBG là viết tắt của khối nào?

Gold: Khối Ngân hàng Bán lẻ.

Gold pages: 387

- B1a: Theo danh mục thuật ngữ viết tắt trong báo cáo, RBG là viết tắt của "Khối Ngân hàng Bán lẻ" [tr. 2].
  - citations=2; automatic_score=False

- B1b: Theo danh mục thuật ngữ viết tắt trong báo cáo, RBG là viết tắt của "Khối Ngân hàng Bán lẻ" [tr. 2].
  - citations=2; automatic_score=False

- B2: Theo danh mục thuật ngữ viết tắt trong Báo cáo thường niên, RBG là viết tắt của "Khối Ngân hàng Bán lẻ" [tr. 2].
  - citations=2; automatic_score=False; rounds=1; stop=sufficient_after_round_1

### sq-09 — Theo danh mục thuật ngữ viết tắt của báo cáo, CASA là viết tắt của thuật ngữ gì?

Gold: Tiền gửi không kỳ hạn.

Gold pages: 386

- B1a: Theo danh mục thuật ngữ viết tắt trong báo cáo, CASA là viết tắt của "Tiền gửi không kỳ hạn" [tr. 386].
  - citations=386; automatic_score=False

- B1b: Theo danh mục thuật ngữ viết tắt trong báo cáo, CASA là viết tắt của "Tiền gửi không kỳ hạn" [tr. 386].
  - citations=386; automatic_score=False

- B2: CASA là viết tắt của "Tiền gửi không kỳ hạn" theo danh mục thuật ngữ viết tắt trong Báo cáo thường niên [tr. 386].
  - citations=386; automatic_score=False; rounds=1; stop=sufficient_after_round_1

### sq-10 — Techcombank dự báo lợi nhuận trước thuế năm 2027 là bao nhiêu?

Gold: Không có trong báo cáo. Báo cáo thường niên 2025 không đưa ra dự báo lợi nhuận cho năm 2027; hệ thống cần từ chối trả lời thay vì suy đoán.

Gold pages: none

- B1a: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=True

- B1b: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=True

- B2: Không tìm thấy đủ thông tin trong Báo cáo thường niên để trả lời câu hỏi này.
  - citations=none; automatic_score=True; rounds=2; stop=max_rounds_reached_insufficient
