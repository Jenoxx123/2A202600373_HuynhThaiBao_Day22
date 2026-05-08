# Reflection - Lab 22 (DPO/ORPO Alignment)

**Tên:** Huỳnh Thái Bảo  
**Cohort:** A20 - Track 3  
**Tier đã chạy:** T4  
**Date:** 2026-05-08

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Tesla T4, 15.6 GB |
| Runtime | Google Colab |
| Base model | `unsloth/Qwen3-0.6B` |
| SFT method | QLoRA, LoRA `r=16`, `alpha=32`, 1 epoch |
| SFT dataset | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 1000 samples |
| Preference dataset | `argilla/ultrafeedback-binarized-preferences-cleaned`, 2000 pairs |
| DPO config | `beta=0.1`, learning rate `5e-7`, `loss_type=sigmoid` |
| Cost | $0, Colab free tier |

Preference data sau khi format có median prompt 87 tokens, median chosen 400 tokens, median rejected 278 tokens. Với `MAX_LEN=1024`, có `96.2%` pair fit, nên mức truncate tương đối thấp và tín hiệu chosen/rejected vẫn đủ dùng cho DPO.

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training examples | 1000 SFT samples | 2000 preference pairs |
| Training steps | 125 | 250 |
| Final loss | 1.9223 | 0.9888 |
| End chosen reward | n/a | +4.381 |
| End rejected reward | n/a | +3.858 |
| End reward gap | n/a | +0.523 |
| Qualitative win/loss/tie | n/a | 0 win / 0 loss / 8 tie |

Kết quả chính là DPO train thành công và reward gap cuối dương. Tuy nhiên, phần so sánh output thủ công chưa cho thấy cải thiện rõ bằng mắt thường.

---

## 3. Reward Curves Analysis

Ảnh kết quả: `submission/screenshots/03-dpo-reward-curves.png`

Reward curve là phần đáng chú ý nhất của lab này vì nó cho thấy DPO có học được preference signal hay chỉ đang tạo ra một khoảng cách reward giả. Ở cuối training, `chosen reward` đạt `+4.381`, `rejected reward` đạt `+3.858`, và reward gap là `+0.523`. Điều này đúng với mục tiêu cơ bản của DPO: response được chọn nên có implicit reward cao hơn response bị loại. Quan trọng hơn, notebook diagnostic in ra kết luận `INTENDED`, tức là chosen reward tăng và gap vẫn dương, không rơi vào trường hợp xấu mà deck §3.4 cảnh báo như reward gap tăng chỉ vì rejected reward tụt mạnh hơn.

Tuy vậy, đường cong không hoàn toàn mượt. Với chỉ 2000 preference pairs, batch size per device 1 và gradient accumulation 8, mỗi điểm log vẫn chịu ảnh hưởng lớn từ mini-batch cụ thể. Vì vậy có những đoạn reward gap dao động thay vì tăng đều. Mình không xem đây là lỗi training, mà là dấu hiệu rằng dữ liệu preference còn nhỏ và model 0.6B có khả năng phân biệt hạn chế. Nếu so với mục tiêu alignment thực tế, kết quả này mới chứng minh rằng pipeline DPO hoạt động đúng hướng, chưa chứng minh rằng chất lượng hội thoại đã cải thiện rõ. Bước tiếp theo hợp lý sẽ là chạy beta sweep với `0.05`, `0.1`, `0.5` và so sánh thêm win-rate judge, vì chỉ nhìn reward gap có thể đánh giá quá lạc quan.

---

## 4. Qualitative Comparison

Ảnh tổng hợp: `submission/screenshots/04-side-by-side-table.png`

Mình so sánh 8 prompt cố định trong notebook, gồm 4 prompt helpfulness và 4 prompt safety:

| Group | Prompt count | Result |
|---|---:|---|
| Helpfulness | 4 | 4 tie |
| Safety | 4 | 4 tie |
| Overall | 8 | 8 tie |

**Win/loss/tie summary:** SFT+DPO wins `0/8`, loses `0/8`, ties `8/8`.  
**Judge:** manual/fallback rubric vì không có OpenAI hoặc Anthropic API key trong runtime.

Nhận xét của mình là hai model trả lời khá giống nhau về cấu trúc. Cả hai thường dùng dạng danh sách hoặc giải thích ngắn, và chưa có khác biệt đủ mạnh để chọn winner tự tin. Điều này hợp lý với cấu hình T4: base model nhỏ (`0.6B`), DPO chỉ 1 epoch, preference data là tiếng Anh trong khi SFT data là tiếng Việt. DPO đã tạo tín hiệu reward dương, nhưng tín hiệu đó chưa chuyển thành cải thiện qualitative rõ rệt trên 8 prompt tiếng Việt.

---

## 5. Beta Trade-Off

Mình chưa chạy beta sweep đầy đủ, nên phần này là giả thuyết dựa trên lần chạy `beta=0.1` và deck §3.2:

1. Nếu giảm `beta` xuống `0.05`, model sẽ bị kéo về reference/SFT mạnh hơn. Reward gap có thể nhỏ hơn, nhưng output có khả năng ổn định và ít overfit hơn.
2. Nếu tăng `beta` lên `0.5`, DPO sẽ ép model theo preference pair mạnh hơn. Reward gap có thể tăng nhanh hơn, nhưng rủi ro length hacking, output một màu, hoặc mất năng lực reasoning cũng cao hơn.
3. Với kết quả hiện tại, `beta=0.1` là lựa chọn cân bằng: đủ tạo gap dương `+0.523`, nhưng chưa đủ để tạo thay đổi qualitative rõ trên 8 prompt.

---

## 6. Personal Reflection - Single Change That Mattered Most

Quyết định ảnh hưởng lớn nhất trong bài này là chọn chạy toàn bộ pipeline trên T4 với `unsloth/Qwen3-0.6B`, thay vì cố dùng model lớn hơn. Lựa chọn thay thế mình đã cân nhắc là chuyển sang BigGPU để dùng model 1.7B hoặc lớn hơn, vì DPO cần cả policy và reference model nên VRAM tăng gần gấp đôi so với SFT. Nếu dùng model lớn hơn, có thể output tiếng Việt sẽ tốt hơn và khác biệt giữa SFT-only với SFT+DPO sẽ dễ thấy hơn. Nhưng mình chọn T4 vì mục tiêu chính của lab là hoàn thành end-to-end: SFT mini, prepare preference data, train DPO, so sánh output, export GGUF, smoke test và benchmark.

Quyết định này giúp mình học được nhiều hơn về pipeline alignment thực tế. Phần train chạy được, reward curve có gap dương, GGUF export thành công, và benchmark tạo được bảng so sánh. Đổi lại, chất lượng kết quả không quá ấn tượng: qualitative eval là `8/8 tie`, tức là DPO chưa tạo ra cải thiện rõ ràng trong những prompt mình kiểm tra. Điều bất ngờ là số training nhìn khá ổn (`DPO loss=0.9888`, gap `+0.523`) nhưng output vẫn gần như không tách biệt. Nếu làm lại, mình vẫn sẽ bắt đầu bằng T4 để debug pipeline nhanh, nhưng sau đó sẽ ưu tiên hai thay đổi: dùng preference data tiếng Việt hoặc hybrid Việt-Anh, và chạy judge tự động với API key để giảm bias khi chấm thủ công.

---

## 7. Benchmark Interpretation

Ảnh benchmark: `submission/screenshots/07-benchmark-comparison.png`

| Benchmark | SFT-only | SFT+DPO | Delta |
|---|---:|---:|---:|
| IFEval | 0.000 | 0.000 | +0.000 |
| GSM8K | 1.000 | 1.000 | +0.000 |
| MMLU | 0.439 | 0.439 | +0.000 |
| AlpacaEval-lite | NaN | NaN | NaN |

Không có benchmark nào tăng hoặc giảm trong lần chạy này: IFEval, GSM8K và MMLU đều có delta `+0.000`. AlpacaEval-lite bị `NaN` vì không có API key để dùng judge, nên không thể tính win-rate. Vì các limit trong notebook T4 đang rất nhỏ (`IFEval=2`, `GSM8K=2`, `MMLU=2`) nên mình không nên kết luận quá mạnh rằng DPO không ảnh hưởng gì. Với sample nhỏ, một hoặc hai câu có thể làm điểm số nhìn cực đoan, ví dụ GSM8K đạt `1.000` cho cả hai model nhưng không có nghĩa là model giỏi toán tổng quát.

Theo framing alignment tax của deck §8.1, điều mình cần tìm là benchmark nào giảm sau DPO, đặc biệt là GSM8K hoặc MMLU, vì chat alignment đôi khi lấy bớt năng lực reasoning hoặc kiến thức nền. Ở lần chạy này chưa thấy alignment tax: GSM8K không giảm, MMLU không giảm, nhưng cũng không thấy instruction-following cải thiện vì IFEval vẫn `0.000`. Cách đọc đúng là DPO ở cấu hình này tạo được tín hiệu training nội bộ nhưng chưa tạo thay đổi measurable trên benchmark nhỏ. Nếu rerun nghiêm túc hơn, mình sẽ tăng limit benchmark, chạy AlpacaEval-lite với judge, và so sánh thêm output length để kiểm tra liệu DPO có làm model trả lời ngắn hơn, an toàn hơn, hay chỉ giữ nguyên hành vi SFT.

---

## Bonus

- [ ] Đã làm beta sweep
- [ ] Đã push DPO adapter lên HuggingFace Hub
- [x] Đã export GGUF Q4_K_M
- [ ] Đã link W&B run public
- [ ] Đã làm cross-judge comparison
- [ ] Đã làm `BONUS-CHALLENGE.md`
- [ ] Pair work

---

## Điều Ngạc Nhiên Nhất

Điều ngạc nhiên nhất là reward curve có vẻ tích cực hơn qualitative eval. DPO tạo được reward gap dương và notebook diagnostic xem đây là hướng train đúng, nhưng khi nhìn 8 câu trả lời tiếng Việt thì SFT-only và SFT+DPO vẫn gần như hòa nhau. Bài lab này nhắc mình rằng alignment không nên được đánh giá bằng một con số duy nhất: reward curve, judge eval, benchmark và đọc output thật đều cần đi cùng nhau.
