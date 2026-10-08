# TVCFORMER: A TRANSFORMER-BASED LONG-TERM MULTIVARIATE TIME SERIES FORECASTING METHOD USING TIME-VARIABLE COUPLING CORRELATION GRAPH

Repository này phục vụ tái thực nghiệm, phân tích mở rộng và đọc hiểu bài báo TVCFormer trong khuôn khổ bài tập môn học.

## 1. GIỚI THIỆU CHUNG 

Dự báo chuỗi thời gian đa biến dài hạn (Long-Term Multivariate Time Series Forecasting) là bài toán dự đoán các giá trị trong tương lai dựa trên nhiều biến quan sát theo thời gian. Đây là một bài toán quan trọng và được ứng dụng trong nhiều lĩnh vực thực tế, ví dụ như:

- Dự báo nhu cầu và tiêu thụ điện năng
- Dự báo thời tiết
- Giao thông
- Kinh tế và tài chính
- Quản lý năng lượng
- Các hệ thống IoT và công nghiệp
- ....

Một thách thức quan trọng là mối quan hệ giữa các biến có thể thay đổi theo từng khoảng thời gian. Vì vậy, việc sử dụng một cấu trúc tương quan cố định có thể không phản ánh đầy đủ các đặc điểm cục bộ và động của dữ liệu.

Để giải quyết vấn đề này, bài báo "TVCFormer" kết hợp **Transformer** và **Graph Neural Network (GNN)**. Mô hình xây dựng **Time-Variable Coupling Correlation Graph (TVC Graph)** để mô hình hóa mối quan hệ tương quan cục bộ, thay đổi theo thời gian giữa các biến, trong khi Transformer được sử dụng để học các phụ thuộc theo thời gian.

Trong project này, mô hình TVCFormer được **tái thực nghiệm** dựa trên mã nguồn, dữ liệu và thiết lập do tác giả cung cấp. Kết quả sẽ được đánh giá, phân tích và so sánh với một số phương pháp dự báo chuỗi thời gian khác.

## 2. THÔNG TIN BÀI BÁO

**Tên bài báo:** TVC Former: A transformer-based long-term multivariate time series forecasting method using time-variable coupling correlation graph

**Tác giả:**

- Zhenyu Liu
- Yuan Feng
- Hui Liu
- Ruining Tang
- Bo Yang
- Donghao Zhang
- Weiqiang Jia
- Jianrong Tan

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

TVCFormer tập trung vào bài toán **dự báo chuỗi thời gian đa biến dài hạn**. Với một chuỗi thời gian gồm nhiều biến được quan sát trong một khoảng thời gian lịch sử, mục tiêu là dự đoán giá trị của các biến tại những thời điểm trong tương lai.

Bài toán yêu cầu mô hình đồng thời khai thác hai loại thông tin:

- **Sự phụ thuộc theo thời gian** của từng biến.
- **Mối quan hệ giữa các biến** trong chuỗi thời gian.

Trong thực tế, mối quan hệ giữa các biến có thể thay đổi theo từng giai đoạn. Do đó, việc sử dụng một cấu trúc tương quan cố định trên toàn bộ chuỗi dữ liệu có thể chưa mô hình hóa đầy đủ các đặc điểm cục bộ và thay đổi theo thời gian.

TVCFormer giải quyết vấn đề này bằng cách mô hình hóa mối quan hệ giữa các biến ở mức **local window-level subsequences** thông qua **Time-Variable Coupling Correlation Graph (TVC Graph)**. Cách tiếp cận này cho phép mô hình khai thác các mối quan hệ tương quan cục bộ và kết hợp chúng với khả năng học phụ thuộc dài hạn của Transformer.

## 5. ĐÓNG GÓP CHÍNH CỦA BÀI BÁO

Bài báo đưa ra ba đóng góp chính:

1. **Đề xuất TVC Graph** để mô hình hóa các mẫu tương quan cục bộ giữa các biến trong chuỗi thời gian đa biến. Mỗi nút trong đồ thị biểu diễn một subsequence cục bộ của một biến.

2. **Đề xuất phương pháp xây dựng TVC Graph động**, gồm khởi tạo đồ thị dựa trên độ tương đồng thống kê và cơ chế **Pattern Capture-Fusion Sparsification** sử dụng các tham số có thể học để tối ưu cấu trúc đồ thị theo từng nhiệm vụ.

3. **Đề xuất Time-Variable Joint-Encoding Framework**, kết hợp Transformer và Graph Neural Network. Transformer được sử dụng để học các phụ thuộc dài hạn trong từng biến, trong khi GNN khai thác các mối quan hệ tương quan cục bộ giữa các biến thông qua TVC Graph.
