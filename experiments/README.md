# TVCFormer — Experimental Reproduction and Analysis

Repository này phục vụ quá trình **tái thực nghiệm, đánh giá và phân tích mở rộng** mô hình TVCFormer (*Time-Variable Coupling Correlation Graph Transformer*) cho bài toán dự báo chuỗi thời gian đa biến dài hạn (*Long-Term Multivariate Time Series Forecasting — L-MTSF*).

## 1. Project Objectives

Các mục tiêu chính của dự án gồm:

- Tìm hiểu kiến trúc và nguyên lý hoạt động của TVCFormer.
- Tái thực nghiệm mô hình trên các bộ dữ liệu benchmark được sử dụng trong bài báo gốc.
- Đánh giá hiệu quả dự báo bằng hai chỉ số MSE và MAE.
- Phân tích ảnh hưởng của các siêu tham số quan trọng, đặc biệt là tỷ lệ giữ lại cạnh `kr` và số bộ lọc mẫu tương quan `M`.
- So sánh kết quả với các phương pháp dự báo liên quan.
- Mở rộng thực nghiệm trên bộ dữ liệu Exchange_rate để khảo sát khả năng áp dụng mô hình trên dữ liệu tỷ giá hối đoái.

## 2. Paper and Source Code

- **Paper:** *TVC Former: A Transformer-Based Long-Term Multivariate Time Series Forecasting Method Using Time-Variable Coupling Correlation Graph*
- **Authors:** Zhenyu Liu et al.
- **Journal:** Knowledge-Based Systems, Volume 314, 2025, Article 113147.
- **DOI:** [10.1016/j.knosys.2025.113147](https://doi.org/10.1016/j.knosys.2025.113147)
- **Official implementation:** [IFengSource/TVCFormer](https://github.com/IFengSource/TVCFormer)

## 3. Datasets

Thực nghiệm sử dụng sáu bộ dữ liệu chuỗi thời gian đa biến.

| Dataset | Domain | Sampling Frequency | Variables |
|---|---|---|---:|
| ETTh1 | Electricity transformer temperature | Hourly | 7 |
| ETTh2 | Electricity transformer temperature | Hourly | 7 |
| ETTm1 | Electricity transformer temperature | 15 minutes | 7 |
| ETTm2 | Electricity transformer temperature | 15 minutes | 7 |
| Weather | Weather conditions | 10 minutes | 21 |
| Exchange_rate | Exchange rates | Daily | 8 |

Năm bộ dữ liệu ETTh1, ETTh2, ETTm1, ETTm2 và Weather được sử dụng để đánh giá mô hình trên các bộ dữ liệu benchmark. Exchange_rate được bổ sung cho phần phân tích mở rộng.

## 4. Experimental Configuration

Cấu hình thực nghiệm được xây dựng dựa trên các siêu tham số được trình bày trong báo cáo và bài báo gốc.

### 4.1. Model Hyperparameters

| Hyperparameter | Symbol / Argument | Value |
|---|---|---:|
| Input sequence length | `seq_len` | 336 |
| Prediction length | `pred_len` | 96, 192, 336, 720 |
| Temporal encoder layers | `l` | 2 |
| Patch length | `P` | 12 |
| Patch stride | `S` | 6 |
| Number of local windows | `L` | 7 |
| Hidden dimension | `d_model` | 32 |
| Number of pattern filters | `M` | 10 |
| Convolution kernel size | `CK` | 3 |
| Graph retention ratio | `kr` | 0.5 |
| Number of GNN layers | — | 1 |
| Number of attention heads | — | 8 |
| Attention MLP dimension | — | 128 |
| Embedding dropout | — | 0.5 |
| Batch size | `batch_size` | 128 |
| Maximum epochs | `train_epochs` | 100 |
| Early stopping patience | `patience` | 10 |
| Learning rate | `learning_rate` | 0.001 |

**Lưu ý:** Bảng trên thể hiện cấu hình tham chiếu. Cần đối chiếu với file cấu hình và lệnh chạy thực tế để xác nhận các giá trị được sử dụng trong từng thí nghiệm.

### 4.2. Experimental Environment

| Component | Configuration |
|---|---|
| Programming language | Python |
| Deep learning framework | PyTorch |
| Graph learning library | Deep Graph Library (DGL) |
| Data processing | NumPy, Pandas, Scikit-learn |
| Device | CPU |

## 5. Hyperparameter Analysis

Hai siêu tham số `kr` và `M` được quan tâm trong quá trình phân tích ảnh hưởng đến hiệu quả dự báo.

### 5.1. Graph Retention Ratio (`kr`)

`kr` điều khiển tỷ lệ giữ lại các cặp biến ứng viên trong quá trình xây dựng đồ thị.

- Nếu `kr` quá thấp, một số quan hệ hữu ích có thể bị loại bỏ ngay từ bước khởi tạo.
- Nếu `kr` quá cao, đồ thị có thể giữ lại nhiều kết nối ít đóng góp cho dự báo.
- Theo kết quả khảo sát trên ETTm1 được trình bày trong báo cáo, `kr = 0.5` là lựa chọn phù hợp trong các cấu hình được đánh giá.

### 5.2. Number of Pattern Filters (`M`)

`M` xác định số lượng bộ lọc dùng để thu nhận các mẫu tương quan cục bộ.

Theo phân tích trong báo cáo, vùng `M = 6–10` cho sai số tương đối thấp; `M = 10` là lựa chọn được sử dụng theo cấu hình bài báo gốc.

### 5.3. Convolution Kernel Size (`CK`)

`CK` xác định kích thước kernel trong phép tích chập được sử dụng ở cơ chế xử lý và làm thưa đồ thị. Giá trị tham chiếu trong cấu hình là `CK = 3`.

## 6. Running Experiments

File thực thi được sử dụng trong phần tái thực nghiệm:

`reproduce_tvcformer.py`

Trước khi chạy, hãy kiểm tra:

1. Môi trường Python và các thư viện phụ thuộc đã được cài đặt.
2. Dữ liệu được đặt đúng đường dẫn mà chương trình yêu cầu.
3. Các tham số trong file cấu hình phù hợp với bộ dữ liệu và prediction length cần chạy.
4. Thiết bị chạy và tài nguyên tính toán đáp ứng yêu cầu của chương trình.

Nếu file sử dụng cấu hình mặc định, có thể chạy bằng:

```bash
python reproduce_tvcformer.py
```

Lệnh trên chỉ phù hợp nếu file thực thi không yêu cầu tham số bắt buộc khác. Cách chọn dataset và prediction length cần dựa trên các tham số thực sự được hỗ trợ trong mã nguồn.

## 7. Evaluation Metrics

Mô hình được đánh giá bằng hai chỉ số:

- **Mean Squared Error (MSE):** Sai số bình phương trung bình giữa giá trị dự báo và giá trị thực tế.
- **Mean Absolute Error (MAE):** Sai số tuyệt đối trung bình giữa giá trị dự báo và giá trị thực tế.

Giá trị MSE và MAE càng thấp thì sai số dự báo càng nhỏ.

## 8. Experimental Results

Kết quả cần được tổng hợp theo từng bộ dữ liệu và prediction length. Bảng dưới đây là mẫu để cập nhật sau khi thực nghiệm hoàn tất.

| Dataset | Prediction Length | MSE | MAE |
|---|---:|---:|---:|
| ETTh1 | 96 / 192 / 336 / 720 | — | — |
| ETTh2 | 96 / 192 / 336 / 720 | — | — |
| ETTm1 | 96 / 192 / 336 / 720 | — | — |
| ETTm2 | 96 / 192 / 336 / 720 | — | — |
| Weather | 96 / 192 / 336 / 720 | — | — |
| Exchange_rate | 96 / 192 / 336 / 720 | — | — |

Các dấu `—` biểu thị kết quả chưa được cập nhật, không phải giá trị đo được.

## 9. Extended Analysis: Exchange_rate

Exchange_rate là bộ dữ liệu được bổ sung nhằm khảo sát khả năng áp dụng TVCFormer trên dữ liệu tỷ giá hối đoái.

Phần phân tích mở rộng tập trung vào:

- Khảo sát đặc điểm và mức độ biến động của chuỗi dữ liệu.
- Đánh giá sai số dự báo ở các prediction lengths được thực nghiệm.
- Trực quan hóa giá trị thực tế và giá trị dự báo.
- Phân tích kết quả, ưu điểm và hạn chế của mô hình trên bộ dữ liệu bổ sung.

Kết quả thực nghiệm sẽ được sử dụng làm cơ sở cho các nhận xét và kết luận trong báo cáo.

## 10. Project Status

- [x] Xác định các bộ dữ liệu thực nghiệm.
- [x] Tổng hợp cấu hình siêu tham số tham chiếu.
- [ ] Hoàn tất thực nghiệm trên toàn bộ bộ dữ liệu.
- [ ] Tổng hợp kết quả MSE và MAE.
- [ ] Hoàn thiện biểu đồ và phân tích kết quả.
- [ ] Hoàn thiện so sánh với các mô hình cơ sở.

*Trạng thái cần được cập nhật theo tiến độ thực tế của dự án.*

---

**Academic project:** TVCFormer reproduction, experimental evaluation, and extended analysis.