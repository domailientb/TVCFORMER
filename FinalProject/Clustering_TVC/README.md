# Clustering-TVCFormer

Dự án này triển khai bộ chọn cụm theo từng cửa sổ lịch sử (*per-history-window clustering selector*) và quy trình thực nghiệm có khả năng tái lập. Bộ chọn chỉ sử dụng dữ liệu lịch sử đã quan sát của từng mẫu, chỉ tính hệ số Pearson cho các cặp biến thuộc danh sách ứng viên và tạo ra các cạnh đồ thị đối xứng tương thích với Graph Generator và GraphSAGE hiện có của TVCFormer.

Các thành phần tối ưu như NumPy K-Means, ghép cạnh theo batch, precompute có đánh chỉ mục và các phương thức tham chiếu để kiểm tra tính tương đương (*parity reference paths*) vẫn được giữ lại.

## Cấu hình bộ dữ liệu

Năm cấu hình thực nghiệm được lưu trong `configs/`:

- `ETTh1.yaml`, `ETTh2.yaml`, `ETTm1.yaml`, `ETTm2.yaml`
- `Weather.yaml`

Các cấu hình sử dụng chung bộ thiết lập huấn luyện đã được xác nhận ở phase 5: sequence length 336, batch size 128, Adam learning rate 0.001, tối đa 100 epochs, patience 5, cùng các thiết lập về model width và graph.

Chỉ các thông tin về tên dataset, đường dẫn/tệp dữ liệu đầu vào và số lượng input channels khác nhau.

Dataset Weather sử dụng khóa viết thường `weather` theo yêu cầu của data provider và có 21 biến đầu vào. Đường dẫn `checkpoints`, nếu được sử dụng để lưu checkpoint, sẽ nằm bên trong thư mục dự án này.

## Chạy trên một dataset

Chạy các lệnh sau từ thư mục gốc của repository bằng Git Bash:

```bash
bash FinalProject/Clustering_TVC/scripts/ETTh1.sh
bash FinalProject/Clustering_TVC/scripts/ETTh2.sh
bash FinalProject/Clustering_TVC/scripts/ETTm1.sh
bash FinalProject/Clustering_TVC/scripts/ETTm2.sh
bash FinalProject/Clustering_TVC/scripts/Weather.sh
```

Mỗi script lần lượt chạy 3 seeds: 2021, 2023 và 2025, với 4 prediction horizons: 96, 192, 336 và 720.

Có thể truyền thêm các tùy chọn để giới hạn lần chạy phục vụ phát triển hoặc kiểm thử, ví dụ:

```bash
bash FinalProject/Clustering_TVC/scripts/ETTh1.sh --seeds 2021 --pred_lens 96
```

Khi chạy lại script, chương trình sẽ tiếp tục từ những horizon chưa hoàn thành. Một kết quả CSV chỉ được xem là hoàn thành khi chứa đầy đủ test metrics hợp lệ (giá trị hữu hạn), best epoch, runtime và sample count, đồng thời log tương ứng của dataset và seed phải có completion marker xác nhận hoàn thành.

Các horizon bị lỗi sẽ được ghi nhận, nhưng những horizon còn lại vẫn tiếp tục chạy. Ở lần gọi tiếp theo, chương trình sẽ tự động chạy lại các kết quả bị thiếu hoặc không hợp lệ.

Mỗi dataset và seed sử dụng một log duy nhất tại:

`logs/<dataset>_seed<seed>.log`

Log được ghi nối tiếp (*append-only*), giữ lại các lần chạy trước, bao gồm cả lỗi. Cả standard output và standard error đều được lưu lại.

## Kết quả đầu ra

- Kết quả theo từng horizon: `results/clustering_suite_runs_v2.csv` (ghi nối tiếp).
- Các file CSV và Markdown tổng hợp mean/std có timestamp: `results/`.
- Các biểu đồ có timestamp: `figures/`.
- Log thực nghiệm riêng cho từng dataset và seed: `logs/`.
- Log smoke test: `logs/smoke/`; các CSV kết quả smoke test cũng được lưu theo dạng append-only.

Các thống kê tổng hợp chỉ bao gồm những lần chạy đã hoàn thành. Độ lệch chuẩn được tính theo **sample standard deviation** và để trống khi chưa có ít nhất 2 seeds hoàn thành.

Bảng kết quả theo từng horizon bao gồm:

- Validation MSE/MAE tại epoch có validation MSE tốt nhất.
- Test MSE/MAE.
- Best epoch.
- Train time, validation time, test time và total time.
- Trạng thái thực nghiệm (*status*).

Các kết quả thăm dò trước đây vẫn được lưu riêng.

Checkpoint được lưu trong `checkpoints/` nếu bật cơ chế lưu checkpoint. Runner của full training hiện giữ best state trong bộ nhớ và ghi lại epoch tương ứng trong dòng kết quả, thay vì lưu checkpoint ra file.

## Kiểm tra và xác thực

Chạy các lệnh sau bằng PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest FinalProject\Clustering_TVC\tests -q -p no:cacheprovider

.\.venv\Scripts\python.exe .\FinalProject\Clustering_TVC\exp_clustering.py --config_yaml .\FinalProject\Clustering_TVC\configs\smoke_etth1.yaml
```

Smoke run được giới hạn có chủ đích ở 1 train batch và 1 test batch. Nó kiểm tra tính tích hợp của selector, đầu ra graph, tính đối xứng và quá trình forward/backward.

Smoke run **không phải kết quả đánh giá trên toàn bộ dataset**.

Để đánh giá đầy đủ, hãy chạy các script dataset tương ứng, bao gồm tổng cộng 60 lượt đánh giá dataset/seed/horizon.

Không có benchmark hay profiling script lịch sử nào bị xóa. Các file này vẫn được giữ lại vì log và báo cáo của chúng là nguồn đối chiếu (*provenance*) cho những kết quả tối ưu đã được đo lường trước đây.