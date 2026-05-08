# Reflection - Lab 22 (DPO/ORPO Alignment)

**Tên:** Huỳnh Thái Bảo  
**Cohort:** A20 - Track 3  
**Tier đã chạy:** T4  
**Date:** 2026-05-08

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Tesla T4 (15.6 GB) |
| CUDA / driver | Torch 2.10.0+cu128 (runtime Colab) |
| Base model | `unsloth/Qwen3-0.6B` |
| SFT dataset slice | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 1000 samples, 1 epoch |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned`, 2000 pairs, 1 epoch |
| `COMPUTE_TIER` env | `T4` |
| Total cost | $0 (Colab free GPU) |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | - | Không log thời gian chi tiết |
| VRAM peak | Không log | Không log |
| Final loss | 1.9223 (SFT) | 0.9888 (DPO) |
| Reward gap (chosen - rejected, end of training) | n/a | +0.524 |
| Mean output length | Chưa đo token trung bình | Chưa đo token trung bình |

Ghi chú: `chosen reward` cuối là `+4.383`, `rejected reward` cuối là `+3.859`, cho thấy gap dương và đi đúng mục tiêu của DPO.

---

## 3. Reward curves analysis (>= 100 words)

Ảnh kết quả: `screenshots/03-dpo-reward-curves.png`

Đường `chosen reward` dao động trong khoảng xấp xỉ 4.0-4.8 và kết thúc ở mức 4.383. Đường `rejected reward` nằm thấp hơn, thường ở khoảng 3.6-4.2, kết thúc tại 3.859. Điều này nghĩa là trong phần lớn quá trình train, mẫu chosen được ưu tiên cao hơn mẫu rejected. Tuy nhiên, reward gap không tăng đều một chiều mà biến động khá mạnh: có lúc vượt 0.9, nhưng cũng có lúc âm ở khoảng bước 130. Điều này gợi ý rằng với tập 2k pair, tín hiệu preference đủ để tạo phân biệt, nhưng vẫn còn nhiễu và chưa ổn định ở từng mini-batch. Kết quả cuối với gap `+0.524` là tín hiệu tích cực: model đã học theo hướng tăng xác suất cho đáp án được prefer. Mình đánh giá DPO đã “làm đúng việc”, nhưng mức độ cải thiện chưa mạnh và cần sweep beta hoặc làm sạch dữ liệu hơn để đường cong mượt và ổn định hơn.

---

## 4. Qualitative comparison (>= 8 examples)

Ảnh tổng hợp: `screenshots/04-side-by-side-table.png`

Đã so sánh 8 prompt, gồm 4 prompt helpfulness và 4 prompt safety. Trong lần chạy này, kết quả theo rubric thủ công là **tie 8/8**:

| # | Prompt category | Winner |
|---|---|---|
| 1 | helpfulness | tie |
| 2 | helpfulness | tie |
| 3 | helpfulness | tie |
| 4 | helpfulness | tie |
| 5 | safety | tie |
| 6 | safety | tie |
| 7 | safety | tie |
| 8 | safety | tie |

**Win/loss/tie summary:** SFT+DPO wins 0/8, ties 8/8, loses 0/8  
**Judge used:** manual rubric (không có API key nên không dùng GPT-4o-mini/Claude)

Nhận xét ngắn: hai model sinh output rất giống nhau, cùng có xu hướng lặp câu và trả lời theo pattern dạng danh sách. Với kích thước model 0.6B và chỉ 1 epoch DPO, thay đổi chất lượng để nhìn thấy bằng mắt thường là khá nhỏ.

---

## 5. beta trade-off

Chưa chạy beta sweep. Giả thuyết 3 câu:

1. Nếu giảm `beta` từ 0.1 xuống 0.05, reward gap có thể ổn định hơn nhưng chênh lệch chosen-rejected sẽ nhỏ hơn, output có khả năng gần với SFT baseline.
2. Nếu tăng `beta` lên 0.5, model sẽ bị ép mạnh hơn theo preference pair, reward gap có thể tăng nhanh nhưng dễ gây overfit và làm output ngắn hoặc một màu hơn.
3. Điểm cân bằng hợp lý cho bài này có khả năng vẫn quanh `beta=0.1`, đúng với nhận định trong deck rằng beta vừa phải giúp tối ưu ưu tiên mà không quá tay.

---

## 6. Personal reflection - single change that mattered most (>= 150 words)

Quyết định ảnh hưởng lớn nhất trong bài này là chọn **T4 + Qwen3-0.6B + 2000 preference pairs** thay vì chạy cấu hình nặng hơn. Lựa chọn thay thế mình đã cân nhắc là dùng BigGPU để lên 1.7B/3B và có đủ headroom cho eval đầy đủ hơn. Lý do mình chọn T4 là tài nguyên dễ sẵn sàng, khả năng lặp lại nhanh, và ưu tiên hoàn thành toàn bộ pipeline từ SFT -> DPO -> merge GGUF -> smoke test trong một lần chạy. Quyết định này giúp mình có được toàn bộ artifact đúng rubric, nhưng đổi lại biên độ cải thiện chất lượng rất khó thấy rõ trên 8 prompt thủ công. Kết quả vừa xác nhận vừa gây bất ngờ: xác nhận ở chỗ reward gap cuối vẫn dương (`+0.524`), nghĩa là DPO học được preference signal; bất ngờ ở chỗ quality by-eye gần như không tách được (`8/8 tie`). Nếu làm lại vào ngày mai, mình vẫn sẽ giữ T4 cho vòng đầu, nhưng sẽ bổ sung 2 thay đổi: (1) chạy judge tự động có API key để tránh tie do thiếu khả năng phân biệt; (2) fix benchmark command sớm để có số liệu IFEval/GSM8K/MMLU thay vì chỉ dựa vào reward curve.

---

## 7. Benchmark interpretation (>= 150 words)

Ảnh benchmark: `screenshots/07-benchmark-comparison.png`

| Benchmark | SFT-only | SFT+DPO | Delta |
|---|---:|---:|---:|
| IFEval | NaN | NaN | NaN |
| GSM8K | NaN | NaN | NaN |
| MMLU (sampled) | NaN | NaN | NaN |
| AlpacaEval-lite | NaN | NaN | NaN |

Lần chạy benchmark này không trả về điểm hợp lệ do lỗi tương thích tham số trong `lm-eval`:
`ValueError: You can't pass load_in_4bit or load_in_8bit as a kwarg when passing quantization_config argument at the same time.`  
Vì vậy, toàn bộ cột điểm thành NaN và biểu đồ 07 không có bar thực. Điều này rất quan trọng trong phần interpretation: mình không thể kết luận về alignment tax, ví dụ GSM8K giảm, cũng không thể kết luận MMLU có bị mất kiến thức hay không. Ở góc độ học quy trình, bài này cho thấy benchmark pipeline dễ vỡ hơn train pipeline, vì train DPO vẫn thành công nhưng khâu đánh giá bị fail do mismatch version giữa Transformers, lm-eval và Unsloth quantization args. Nếu rerun, cách sửa trực tiếp là đổi `--model_args` để không truyền `load_in_4bit=True` khi base model đã mang quantization config, hoặc chạy benchmark trên merged fp16. Sau khi sửa, mình mới có dữ liệu để phân tích đầy đủ: benchmark nào tăng, benchmark nào giảm, và DPO có tạo trade-off đúng như deck 8.1 hay không.

---

## Bonus

- [ ] Đã làm beta-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [x] Đã release GGUF (Q4_K_M) (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation
- [ ] Pair work

---

## Điều ngạc nhiên nhất khi làm lab này

Điều bất ngờ nhất là reward curve đã cho thấy DPO học được preference signal với gap dương, nhưng khi xem qualitative bằng mắt thường thì hai model vẫn rất giống nhau. Điều đó nhắc mình rằng không nên đánh giá alignment chỉ bằng 8 mẫu output ngắn.
