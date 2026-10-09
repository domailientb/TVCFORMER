# TVCFORMER: A TRANSFORMER-BASED LONG-TERM MULTIVARIATE TIME SERIES FORECASTING METHOD USING TIME-VARIABLE COUPLING CORRELATION GRAPH

Repository này phục vụ tái thực nghiệm, phân tích mở rộng và đọc hiểu bài báo TVCFormer trong khuôn khổ bài tập môn học.

## 1. GIỚI THIỆU CHUNG 

**Long-Term Multivariate Time Series Forecasting** là bài toán dự báo các giá trị tương lai dựa trên dữ liệu lịch sử của nhiều biến được quan sát theo thời gian. Bài toán có nhiều ứng dụng thực tế như dự báo nhu cầu điện năng, dự báo thời tiết, giao thông và giám sát các hệ thống công nghiệp.

Một thách thức quan trọng là mối quan hệ giữa các biến có thể thay đổi theo từng khoảng thời gian. Vì vậy, mô hình cần khai thác đồng thời **temporal dependencies** và **inter-variable correlations**.

Bài báo **TVCFormer** kết hợp **Transformer** và **Graph Neural Network (GNN)**, sử dụng **Time-Variable Coupling Correlation Graph (TVC Graph)** để mô hình hóa các mối quan hệ tương quan cục bộ giữa các biến theo từng khoảng thời gian.

Project tập trung vào việc tìm hiểu phương pháp, tái thực nghiệm mô hình theo thiết lập phù hợp với bài báo, đánh giá kết quả và so sánh với các phương pháp khác trong bài toán forecasting.

## 2. THÔNG TIN BÀI BÁO

**Tên bài báo:** TVC Former: A transformer-based long-term multivariate time series forecasting method using time-variable coupling correlation graph

**Tác giả:** Zhenyu Liu, Yuan Feng, Hui Liu, Ruining Tang, Bo Yang, Donghao Zhang, Weiqiang Jia và Jianrong Tan

**Tạp chí:** Knowledge-Based Systems

**Năm:** 2025

**Volume:** 314

**Article:** 113147

**DOI:** 10.1016/j.knosys.2025.113147

**Mã nguồn chính thức:** https://github.com/IFengSource/TVCFormer

**Dataset:** https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

## 3. MỤC TIÊU CỦA PROJECT

Project được thực hiện với các mục tiêu chính:

1. Tìm hiểu bài toán, vai trò và các ứng dụng thực tế
2. Đọc hiểu bài báo TVCFormer, bao gồm phát biểu bài toán, ý tưởng, đóng góp chính và phương pháp được đề xuất.
3. Chuẩn bị, khám phá và phân tích các bộ dữ liệu được sử dụng trong thực nghiệm
4. Tìm hiểu kiến trúc và cách hoạt động của mô hình TVCFormer
5. Cài đặt môi trường, chạy lại mô hình và tái thực nghiệm các thiết lập phù hợp với bài báo
6. Đánh giá kết quả dự báo bằng các chỉ số phù hợp và trực quan hóa kết quả thực nghiệm
7. So sánh TVCFormer với một số thuật toán khác cùng giải quyết bài toán dự báo chuỗi thời gian
8. Phân tích mở rộng để làm rõ ưu điểm, hạn chế và khả năng ứng dụng của mô hình.

## 4. PHÁT BIỂU BÀI TOÁN

Theo bài báo, multivariate time series được biểu diễn bằng ma trận
\(Z \in \mathbb{R}^{C \times T}\), trong đó \(C\) là số biến và \(T\)
là số bước thời gian.

Với ma trận dữ liệu lịch sử đầu vào
\(X \in \mathbb{R}^{C \times T_s}\), TVCFormer xây dựng hàm dự báo
\(\hat{Y}=f(X\mid\theta)\) để dự đoán chuỗi mục tiêu
\(Y \in \mathbb{R}^{C \times T_p}\), trong đó \(T_s\) là độ dài chuỗi
đầu vào và \(T_p\) là độ dài dự báo.

Để mô hình hóa các mối quan hệ tương quan cục bộ giữa các biến, dữ liệu
đầu vào được chia thành các local window-level subsequences và tiếp tục
được phân chia thành patches. Từ các subsequences này, mô hình xây dựng
Time-Variable Coupling Correlation Graph (TVC Graph), kết hợp Graph
Neural Network và Transformer để khai thác thông tin tương quan giữa
các biến và temporal dependencies.

## 5. ĐÓNG GÓP CHÍNH CỦA BÀI BÁO

### - Time-Variable Coupling Correlation Graph: Đề xuất **TVC Graph** để biểu diễn các local inter-variable correlation patterns trong multivariate time series. Mỗi node đại diện cho một local subsequence, giúp mô hình khai thác các mối quan hệ tương quan tại những khoảng thời gian khác nhau.

### - Dynamic Graph Construction: Đề xuất cơ chế xây dựng và tối ưu TVC Graph dựa trên statistical similarity, kết hợp với **Pattern Capture-Fusion Sparsification**. Cơ chế này sử dụng learnable parameters để điều chỉnh cấu trúc đồ thị phù hợp với đặc điểm dữ liệu và forecasting task.

### - Time-Variable Joint-Encoding Framework: Đề xuất **Time-Variable Joint-Encoding Framework**, kết hợp Transformer và Graph Neural Network (GNN). Transformer khai thác temporal dependencies, trong khi GNN học các mối quan hệ giữa các nodes trên TVC Graph.
