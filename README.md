# TVCFormer: Long-Term Multivariate Time Series Forecasting

Repository phục vụ **đọc hiểu, tái thực nghiệm và phân tích mở rộng** mô hình TVCFormer trong bài toán *Long-Term Multivariate Time Series Forecasting (L-MTSF)*.

## 1. Overview

Long-Term Multivariate Time Series Forecasting (L-MTSF) là bài toán dự báo giá trị tương lai của nhiều biến dựa trên dữ liệu lịch sử. Một thách thức quan trọng là mối quan hệ giữa các biến có thể thay đổi theo thời gian, khiến các phương pháp sử dụng tương quan cố định khó nắm bắt đầy đủ những phụ thuộc cục bộ.

**TVCFormer** kết hợp Transformer và Graph Neural Network (GNN), sử dụng *Time-Variable Coupling Correlation Graph (TVC Graph)* để khai thác phụ thuộc theo thời gian và tương quan giữa các biến.

## 2. Paper Information

- **Title:** TVC Former: A Transformer-Based Long-Term Multivariate Time Series Forecasting Method Using Time-Variable Coupling Correlation Graph
- **Authors:** Zhenyu Liu et al.
- **Journal:** Knowledge-Based Systems, Volume 314, 2025, Article 113147
- **DOI:** [10.1016/j.knosys.2025.113147](https://doi.org/10.1016/j.knosys.2025.113147)
- **Official Repository:** [IFengSource/TVCFormer](https://github.com/IFengSource/TVCFormer)
- **Dataset:** [Download datasets](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)

## 3. Key Contributions

- **TVC Graph:** Biểu diễn tương quan cục bộ giữa các đoạn chuỗi thời gian thuộc những biến khác nhau.
- **Adaptive Graph Construction:** Kết hợp Pearson correlation và Top-K selection để xác định các kết nối quan trọng.
- **Graph Sparsification:** Sử dụng cơ chế *Pattern Capture-Fusion Sparsification* nhằm tinh giản đồ thị và chọn lọc thông tin.
- **Time-Variable Joint Encoding:** Kết hợp Transformer để học đặc trưng thời gian và GNN để tổng hợp thông tin từ các nút liên quan trên đồ thị.

## 4. Project Objectives

- Phân tích bài toán, cơ sở lý thuyết và kiến trúc TVCFormer.
- Khám phá dữ liệu và chuẩn bị môi trường thực nghiệm.
- Tái thực nghiệm mô hình dựa trên mã nguồn chính thức.
- Đánh giá hiệu quả dự báo bằng các chỉ số Mean Squared Error (MSE) và Mean Absolute Error (MAE).
- So sánh kết quả với các mô hình liên quan như Informer và Autoformer.
- Phân tích mở rộng trên bộ dữ liệu Exchange_rate.

## 5. Problem Formulation

Cho chuỗi thời gian đa biến đầu vào:

\[
X \in \mathbb{R}^{C \times T_s}
\]

Trong đó, \(C\) là số biến và \(T_s\) là độ dài chuỗi đầu vào. Mục tiêu là dự báo \(T_p\) bước thời gian tiếp theo:

\[
\hat{Y}=f(X;\theta), \qquad
\hat{Y}\in\mathbb{R}^{C \times T_p}
\]

Trong đó, \(f\) là mô hình dự báo và \(\theta\) là tập tham số cần học.

TVCFormer chia chuỗi đầu vào thành các patches, xây dựng TVC Graph để mô hình hóa tương quan cục bộ, sau đó kết hợp Transformer và GNN để trích xuất đặc trưng và tạo dự báo.

## 6. Technologies

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Graph Learning:** Deep Graph Library (DGL)
- **Data Processing:** NumPy, Pandas, Scikit-learn
- **Evaluation Metrics:** MSE, MAE

## 7. Experiments and Extended Analysis

Các nội dung thực nghiệm của dự án bao gồm:

- Khảo sát và tiền xử lý dữ liệu chuỗi thời gian đa biến.
- Thực nghiệm dự báo trên các bộ dữ liệu được hỗ trợ bởi mã nguồn TVCFormer.
- Đánh giá kết quả theo các prediction lengths khác nhau.
- So sánh hiệu quả dự báo với các phương pháp liên quan.
- Phân tích mở rộng trên **Exchange_rate** nhằm khảo sát khả năng áp dụng mô hình trên bộ dữ liệu bổ sung.
- So sánh với 1 bài báo khác tương đương

## 8. Expected Outcomes

- Làm rõ nguyên lý hoạt động và kiến trúc của TVCFormer.
- Tái thực nghiệm và đánh giá hiệu quả dự báo của mô hình.
- Phân tích kết quả, ưu điểm và hạn chế.
- Đề xuất các hướng phân tích mở rộng dựa trên kết quả thực nghiệm.

---

*Academic project on TVCFormer reproduction, experimental evaluation, and extended analysis.*
