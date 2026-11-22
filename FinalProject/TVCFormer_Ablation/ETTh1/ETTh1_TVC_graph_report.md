# Trích xuất TVC Graph và phân tích cấu trúc trên ETTh1

## Quy trình thực hiện

Đã tải các checkpoint TVCFormer được lựa chọn dựa trên MSE của tập validation trong thí nghiệm thay thế giá trị trung bình trên ETTh1. Không thực hiện huấn luyện hoặc cập nhật tham số. Đồ thị được trích xuất bằng cách đưa toàn bộ các batch hoàn chỉnh của cửa sổ dữ liệu test ETTh1 qua từng checkpoint. Bộ tải dữ liệu test sử dụng thiết lập chính thức `drop_last=True`; số lượng mẫu được liệt kê bên dưới.

Đối với mỗi cạnh của đồ thị, ID nguồn và ID đích được lấy modulo 49 để loại bỏ độ lệch do batch; ma trận kề của từng mẫu được cộng dồn và chia cho tổng số cửa sổ test. ID của các nút được sắp xếp theo thứ tự biến trước: HUFL-T1..T7, HULL-T1..T7, MUFL-T1..T7, MULL-T1..T7, LUFL-T1..T7, LULL-T1..T7, OT-T1..T7. Ma trận 7×7 là giá trị trung bình của khối nút thời gian 7×7 tương ứng, bao gồm cả các giá trị 0 đối với những cạnh không xuất hiện. Các phần tử trên đường chéo được loại bỏ khỏi ma trận kết nối giữa các biến.

## Hình ảnh

- Hình 1: [Bản đồ nhiệt đồ thị nút thời gian 49×49](figure1_graph49_heatmaps.png)
- Hình 2: [Bản đồ nhiệt kết nối giữa các biến 7×7](figure2_variable_heatmaps.png)
- Hình 3: [So sánh giữa các khoảng dự báo](figure3_horizon_comparison.png)

## Các batch đồ thị trên tập test

| Khoảng dự báo | Số cửa sổ được tổng hợp | Checkpoint |
|---:|---:|---|
| 96 | 2784 | `ETTh1_TVCFormer_ablation_pl96_seed2021` |
| 192 | 2688 | `ETTh1_TVCFormer_ablation_pl192_seed2021` |
| 336 | 2528 | `ETTh1_TVCFormer_ablation_pl336_seed2021` |
| 720 | 2144 | `ETTh1_TVCFormer_ablation_pl720_seed2021` |

## Các cạnh kết nối giữa các biến mạnh nhất và yếu nhất

Các giá trị dưới đây là trọng số khối trung bình lớn nhất và nhỏ nhất ở mỗi khoảng dự báo. Chúng mô tả mức độ kết nối được mô hình học, không phải mức độ quan trọng của đặc trưng.

| Khoảng dự báo | Cặp biến kết nối mạnh nhất | Trọng số | Cặp biến kết nối yếu nhất | Trọng số |
|---:|---|---:|---|---:|
| 96 | HUFL–LUFL | 0.422242 | MUFL–LULL | 0.0318559 |
| 192 | HUFL–LUFL | 0.420587 | MUFL–LULL | 0.0314353 |
| 336 | HUFL–LUFL | 0.423537 | MUFL–LULL | 0.0341359 |
| 720 | HUFL–MUFL | 0.423009 | MUFL–LULL | 0.0414913 |

## Mức độ ổn định giữa các khoảng dự báo

Hệ số tương quan Pearson so sánh toàn bộ 49 trọng số kết nối giữa các biến sau khi chuyển mỗi ma trận 7×7 thành một vector. Các giá trị gần 1 cho thấy các mẫu phân bố trọng số cạnh tương đối giống nhau. Đây là kết quả mang tính mô tả; do độ lớn trọng số cạnh được lấy trung bình trên các cửa sổ test, điều này không đồng nghĩa với việc đồ thị của từng cửa sổ giống hệt nhau.

| Các khoảng dự báo | Hệ số tương quan Pearson của ma trận trọng số 7×7 | Độ tương đồng Jaccard của tập cạnh được chọn (Top 30%) |
|---|---:|---:|
| 96 và 192 | 0.9996 | 1.0000 |
| 192 và 336 | 0.9991 | 1.0000 |
| 336 và 720 | 0.9946 | 1.0000 |

## Bảng 1. Năm cặp biến có kết nối mạnh nhất ở mỗi khoảng dự báo

| Khoảng dự báo | Thứ hạng | Cặp biến | Trọng số khối trung bình |
|---:|---:|---|---:|
| 96 | 1 | HUFL–LUFL | 0.422242 |
| 96 | 2 | LUFL–LULL | 0.419946 |
| 96 | 3 | HUFL–MUFL | 0.413430 |
| 96 | 4 | HULL–LULL | 0.413036 |
| 96 | 5 | MUFL–LUFL | 0.412675 |
| 192 | 1 | HUFL–LUFL | 0.420587 |
| 192 | 2 | HUFL–MUFL | 0.417662 |
| 192 | 3 | LUFL–LULL | 0.416772 |
| 192 | 4 | HULL–LULL | 0.413810 |
| 192 | 5 | LULL–OT | 0.411284 |
| 336 | 1 | HUFL–LUFL | 0.423537 |
| 336 | 2 | HUFL–MUFL | 0.423107 |
| 336 | 3 | HULL–LULL | 0.418851 |
| 336 | 4 | LUFL–LULL | 0.418475 |
| 336 | 5 | MUFL–LUFL | 0.411936 |
| 720 | 1 | HUFL–MUFL | 0.423009 |
| 720 | 2 | HUFL–LUFL | 0.422063 |
| 720 | 3 | LUFL–LULL | 0.417089 |
| 720 | 4 | HULL–LULL | 0.415353 |
| 720 | 5 | MUFL–LUFL | 0.414475 |

## Bậc có trọng số của các biến

Bậc có trọng số là tổng các phần tử trên hàng tương ứng của ma trận trọng số trung bình 7×7 đối xứng; chỉ số này tóm tắt cường độ kết nối trung bình của một biến với tất cả các biến còn lại.

| Khoảng dự báo | Biến có bậc có trọng số cao nhất | Trọng số | Biến có bậc có trọng số thấp nhất | Trọng số |
|---:|---|---:|---|---:|
| 96 | LUFL | 1.928820 | OT | 1.268909 |
| 192 | HULL | 1.905094 | OT | 1.278464 |
| 336 | LUFL | 1.922128 | OT | 1.284403 |
| 720 | LUFL | 1.958860 | OT | 1.273807 |

## Diễn giải

Đồ thị được tạo động cho từng cửa sổ test từ tập kết nối giữa các kênh được lựa chọn bằng hệ số tương quan Pearson của TVC, sau đó được chuyển đổi thành trọng số cạnh giữa các nút thời gian thông qua bộ sinh đồ thị. Ma trận 49×49 thể hiện vị trí tập trung của các cạnh kết nối giữa các phân đoạn thời gian; giá trị trung bình của các khối 7×7 tóm tắt mức độ kết nối giữa các biến. Bậc có trọng số là tổng trọng số kết nối trung bình của một biến với sáu biến còn lại. Tham khảo hai tệp `variable_weighted_degree.csv` và `top_variable_edges_by_horizon.csv` để xem đầy đủ thứ hạng.

Các trọng số này thể hiện mức độ kết nối bên trong TVC Graph. Chúng không đo lường mối quan hệ nhân quả hoặc mức độ đóng góp của một biến vào độ chính xác dự báo; để đánh giá mức độ nhạy cảm của mô hình trong dự báo, cần sử dụng thí nghiệm thay thế giá trị trung bình riêng biệt.