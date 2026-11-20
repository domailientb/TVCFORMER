# Siamese_TVC

Nghiên cứu thay đổi candidate support và edge weights của TVC Graph bằng encoder Siamese cho các cửa sổ lịch sử. Transformer, GraphSAGE, joint encoding, decoder, loss MSE, data loader và split được tái sử dụng từ `official/`. Không chỉnh `official/`.

## Dataset và cấu hình

Danh sách giữ đúng theo `FinalProject/Clustering_TVC/`: ETTh1, ETTh2, ETTm1, ETTm2 và Weather. Các YAML dataset chỉ chứa data key, đường dẫn và số kênh; `configs/common.yaml` giữ hyperparameter dùng chung, còn `configs/E0.yaml`, `E1.yaml`, `E2.yaml` định nghĩa khác biệt thực nghiệm.

| Dataset | Data key | Kênh | CSV mặc định |
|---|---|---:|---|
| ETTh1 | ETTh1 | 7 | `experiments/datasets/ETT-small/ETTh1.csv` |
| ETTh2 | ETTh2 | 7 | `experiments/datasets/ETT-small/ETTh2.csv` |
| ETTm1 | ETTm1 | 7 | `experiments/datasets/ETT-small/ETTm1.csv` |
| ETTm2 | ETTm2 | 7 | `experiments/datasets/ETT-small/ETTm2.csv` |
| Weather | weather | 21 | `experiments/datasets/weather.csv` |

Cả năm dùng input length 336, split/scaler chính thức, batch 128, Adam lr 0.001, tối đa 100 epoch và patience 5. Chưa đặt hyperparameter riêng dựa trên test. `analyze_datasets.py` xuất thống kê phân phối, Pearson và cosine hình dạng của các cửa sổ lịch sử căn chỉnh (các cửa sổ chỉ lấy từ train portion) để hỗ trợ phân tích; kết quả đó không tự động đổi cấu hình và không được xem là nhãn quan hệ vật lý.

Encoder dùng chung kết hợp Conv1D trên cửa sổ đã chuẩn hóa theo từng đoạn, sáu descriptor (mean, log-std, slope, độ biến thiên sai phân, độ lớn sai phân và lag-1) cùng thống kê TVC. Chuẩn hóa CNN theo cửa sổ giảm ảnh hưởng scale; descriptor/statistics vẫn cung cấp mức và độ lớn biến thiên. Candidate similarity so sánh embedding của hai biến tại các time-node cùng vị trí.

## E0/E1/E2

- **E0:** model TVCFormer nguyên bản, Pearson Top-K, không tạo Siamese model.
- **E1:** Siamese Top-K và gate cố định ở nhánh Siamese.
- **E2:** hợp Pearson Top-K và Siamese Top-K, sau đó rank fusion Borda giữ đúng số cạnh Top-K của baseline; gate sigmoid học từ forecasting loss trộn cosine thống kê và cosine embedding.

Support chọn rời rạc nên không nhận gradient trong một forward. Forecast loss cập nhật Siamese encoder/projection, gate E2 và PCFS qua edge weights đã chọn. Embedding rank có thể đổi support ở các forward kế tiếp. E2 chọn các cặp tốt nhất trong union nhưng giữ số cặp bằng Pearson Top-K để mật độ graph không đổi theo overlap hay thành phần batch. Candidate cặp kênh được đối xứng hóa; self-loop tuân theo cấu hình TVC gốc. Dữ liệu tương lai/test không tham gia tạo graph.

## Môi trường

Dùng môi trường Python của repo đã cài PyTorch, DGL, PyYAML, NumPy, pandas, scikit-learn, Matplotlib và pytest. Các lệnh bên dưới chạy từ root repo trong Git Bash trên Windows; script tự chọn `.venv/Scripts/python.exe` hoặc `python`.

Chạy kiểm thử trước benchmark:

```bash
./.venv/Scripts/python.exe -m pytest FinalProject/Siamese_TVC/tests -q
```

## Chạy benchmark

Mỗi lệnh chạy một dataset với bốn horizon 96/192/336/720, ba seed 2021/2023/2025 và E0/E1/E2. Ví dụ chạy riêng:

```bash
bash FinalProject/Siamese_TVC/scripts/ETTh1.sh
bash FinalProject/Siamese_TVC/scripts/ETTh2.sh
bash FinalProject/Siamese_TVC/scripts/ETTm1.sh
bash FinalProject/Siamese_TVC/scripts/ETTm2.sh
bash FinalProject/Siamese_TVC/scripts/Weather.sh
```

Chạy toàn bộ 180 lượt:

```bash
bash FinalProject/Siamese_TVC/scripts/run_all.sh
```

Trong PowerShell, dùng launcher native này để tránh chuyển đổi đường dẫn qua Git Bash:

```powershell
.\FinalProject\Siamese_TVC\scripts\run_all.ps1
```

Lệnh trên mặc định resume: chỉ bỏ qua lượt có hàng `completed` trong raw CSV và checkpoint tương ứng. Lượt lỗi được ghi `failed` cùng traceback vào log và CSV; các lượt còn lại vẫn chạy. Khi tất cả dataset script kết thúc, mã thoát khác 0 báo rằng cần xem các lỗi. Chạy lại cùng lệnh để tiếp tục các lượt thiếu/lỗi.

Có thể giới hạn lượt khi cần kiểm tra thủ công; đây không phải benchmark đầy đủ:

```bash
bash FinalProject/Siamese_TVC/scripts/ETTh1.sh --seeds 2021 --horizons 96 --epochs 1 --patience 1 --batch_size 16
```

Điều chỉnh tài nguyên mà không sửa YAML:

```bash
bash FinalProject/Siamese_TVC/scripts/run_all.sh --batch_size 64 --gpu 0 --use_gpu true
```

CPU: thêm `--use_gpu false`. Đường dẫn gốc dữ liệu khác: thêm `--root_path /path/to/data-root`; tên CSV trong từng YAML vẫn được dùng. Có thể chọn output CSV mới bằng `--results_csv FinalProject/Siamese_TVC/results/my_run.csv`. CSV smoke cũ `results/benchmark.csv` không bị ghi đè; full runs mặc định ghi `benchmark_full_runs.csv`.

Mỗi dataset/seed có một append-only log: `logs/ETTh1_seed2021.log` (tương tự 15 log). Log chứa cấu hình, run id, thời gian, train/validation loss từng epoch, best epoch, test MSE/MAE, thời gian và trạng thái/error. Early stopping/checkpoint chỉ dựa vào validation; test chạy sau checkpoint selection.

## Kết quả và phân tích

- `results/benchmark_full_runs.csv`: từng lượt, có status completed/failed.
- `results/benchmark_full_summary.csv`: mean, sample std qua seed và paired improvement (%) E1/E2 so với E0 cho từng dataset/horizon.
- `results/<Dataset>/`: metric/prediction và graph snapshot theo run.
- `checkpoints/`: checkpoint theo run id.
- `logs/`: 15 log dataset/seed.

Tạo lại bảng tổng hợp từ CSV đã có:

```bash
./.venv/Scripts/python.exe FinalProject/Siamese_TVC/summarize_results.py
```

Thống kê dataset (chỉ train split):

```bash
./.venv/Scripts/python.exe FinalProject/Siamese_TVC/analyze_datasets.py
```

Phân tích graph của các run đã lưu; truyền các file graph E0/E1/E2 tương ứng để so sánh:

```bash
./.venv/Scripts/python.exe FinalProject/Siamese_TVC/analyze_graph.py \
  FinalProject/Siamese_TVC/results/ETTh1/ETTh1_E0_pl96_seed2021_graph.npz \
  FinalProject/Siamese_TVC/results/ETTh1/ETTh1_E1_pl96_seed2021_graph.npz \
  FinalProject/Siamese_TVC/results/ETTh1/ETTh1_E2_pl96_seed2021_graph.npz \
  --channels 7 --time_nodes 7 \
  --output_dir FinalProject/Siamese_TVC/analysis/ETTh1_pl96_seed2021
```

Phân tích xuất mật độ/degree/weight, symmetry, self-loop, tỷ lệ candidate Siamese-only/shared/Pearson-only, embedding spread/cosine, gate, overlap và hình heatmap. Các cạnh có Pearson thấp là candidate cần xem xét, không mặc nhiên là quan hệ vật lý hữu ích. So sánh ổn định qua seed cần dùng cùng dataset/horizon và graph snapshots tương ứng.

Không có benchmark đầy đủ mới được khẳng định trong tài liệu này. Chỉ kết luận cải thiện sau khi cả 180 lượt hoàn thành và kiểm tra từng dataset/horizon cùng độ biến thiên seed, graph diagnostics và chi phí.
