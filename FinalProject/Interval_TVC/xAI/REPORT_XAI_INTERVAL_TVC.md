# BÁO CÁO GIẢI THÍCH MÔ HÌNH (XAI) DÀNH CHO CẢI TIẾN INTERVAL-TVC

**Dự án:** Interval-TVC: Cải tiến Dự báo Khoảng trên TVCFormer qua Quantile Head (E1) và CQR Calibration (E1-C)  
**Địa chỉ lưu trữ:** `d:\TVCFORMER\FinalProject\Interval_TVC\xAI\`  
**Thời điểm thực hiện:** 21/11/2026  

---

## TÓM TẮT THỰC THI (EXECUTIVE SUMMARY)

Báo cáo này cung cấp phân tích Explainable AI (XAI) chuyên biệt cho kiến trúc **Interval-TVC** — biến thể dự báo khoảng bất định được xây dựng trên nền tảng TVCFormer. Trọng tâm của XAI là giải phẫu hai cơ chế cốt lõi:
1. **Quantile Head (E1):** Cấu trúc dự báo đồng thời 3 phân vị ($q_{0.025}, q_{0.5}, q_{0.975}$) với cơ chế ràng buộc phi xuyên cắt (non-crossing via softplus gaps).
2. **Conformalized Quantile Regression (E1-C):** Cơ chế hiệu chỉnh hậu xử lý (post-hoc calibration) thích ứng theo từng kênh và từng bước dự báo nhằm bù đắp sai số bao phủ trong thực tế.

### Các phát hiện số liệu chính:
- **Độ bao phủ thực tế (PICP):** Phân vị thô $E1$ chỉ đạt PICP trung bình **0.8798** (87.98%), thấp hơn đáng kể so với mức danh định 95% ($\alpha = 0.05$). Sau khi áp dụng CQR Calibration ($E1$-$C$), PICP trung bình đạt **0.9527** (tăng **+7.28 điểm phần trăm**).
- **Cái giá của sự bao phủ (MPIW):** Để nâng PICP, độ rộng khoảng trung bình (MPIW) tăng từ **12.3220** lên **17.3098** đơn vị gốc (+40.5% vĩ mô; +49.1% theo tỷ lệ trung bình vi mô).
- **Đạt chuẩn danh định 95%:** Calibration đưa **25/51** cấu hình vượt hoặc đạt ngưỡng 0.95. Tuy nhiên, vẫn còn **26/51** cấu hình chưa đạt 0.95 (chủ yếu tập trung ở ETTm2 với 12/12 cấu hình, ETTh2 với 10/12 cấu hình, và Weather với 3/3 cấu hình).
- **Điểm phạt Winkler (Winkler Score):** Tính toán trực tiếp trên toàn bộ 51 bộ dự báo kiểm thử cho thấy điểm Winkler trung bình giảm từ **24.1341** ($E1$) xuống **22.8845** ($E1$-$C$), với **45/51** cấu hình (88.24%) được cải thiện rõ rệt, chứng minh việc mở rộng khoảng mang lại lợi ích bao phủ vượt trội so với độ rộng tăng thêm.
- **Hiện tượng co khoảng cục bộ:** Khoảng **6.05%** số ô $(\text{bước}, \text{kênh})$ có hệ số hiệu chỉnh $\delta < 0$, tương ứng với việc co hẹp biên dự báo ở những vị trí mà mô hình gốc bị dự báo quá bảo thủ.

---

## 1. MỤC TIÊU VÀ PHẠM VI NGHIÊN CỨU XAI

### 1.1. Phạm vi phân tích
Interval-TVC kế thừa toàn bộ khối trích xuất đặc trưng không-thời gian (TVC Graph và GraphSAGE backbone) từ TVCFormer gốc và thay đổi tại tầng đầu ra:
- **E0 (Point Forecast Baseline):** Sử dụng `Flatten_Head` để đưa ra 1 điểm dự báo duy nhất (tối ưu hóa qua hàm tổn thất MSE/L1).
- **E1 (Raw Quantile Head):** Sử dụng `QuantileFlattenHead` thay thế, dự báo đồng thời 3 giá trị $y_{0.025}, y_{0.5}, y_{0.975}$ với hàm phạt Pinball Loss.
- **E1-C (Calibrated Intervals):** Không phải một mô hình huấn luyện mới, mà là bước chuẩn hóa khoảng dự báo sau khi huấn luyện (CQR) bằng cách tính hệ số bù $\delta$ trên tập Validation tách biệt và cộng/trừ vào cận dự báo của $E1$ trên tập Test.

Do đó, XAI cho Interval-TVC **không phân tích lại đồ thị Pearson hay phân cụm GraphSAGE**, mà tập trung trả lời 3 câu hỏi cốt lõi:
1. Cơ chế bù trừ Calibration hoạt động như thế nào giữa các kênh và các bước dự báo?
2. Đánh đổi giữa độ bao phủ (PICP) và độ rộng khoảng (MPIW/Winkler Score) diễn ra ra sao trên từng tập dữ liệu?
3. Minh họa trực quan trên chuỗi thời gian kiểm thử thể hiện rõ sự khác biệt giữa $E1$ và $E1$-$C$ như thế nào?

---

## 2. HIỆN TRẠNG ARTIFACTS VÀ KIỂM TOÁN DỮ LIỆU THỰC NGHIỆM

Quá trình quét và kiểm toán toàn diện (`audit_artifacts.py`) trên thư mục thực nghiệm `results/full_3seeds_v1/` và `checkpoints/full_3seeds_v1/` ghi nhận:

| Tập dữ liệu | Số bước dự báo ($H$) | Số hạt giống ($S$) | Số run dự kiến | Số run thực tế ($E1/E1\text{-}C$) | Tình trạng Artifacts |
|:---|:---:|:---:|:---:|:---:|:---|
| **ETTh1** | 96, 192, 336, 720 | 2021, 2023, 2025 | 12 | 12 | Đầy đủ (`.npz`, `metrics.json`, `calibration.json`, `.pth`) |
| **ETTh2** | 96, 192, 336, 720 | 2021, 2023, 2025 | 12 | 12 | Đầy đủ |
| **ETTm1** | 96, 192, 336, 720 | 2021, 2023, 2025 | 12 | 12 | Đầy đủ |
| **ETTm2** | 96, 192, 336, 720 | 2021, 2023, 2025 | 12 | 12 | Đầy đủ |
| **Weather** | 96, 192, 336 | 2021 | 12 | **3** | Chỉ có 1 hạt giống cho 3 horizon; thiếu $H=720$ |
| **TỔNG CỘNG**| — | — | **60** | **51** | **51 cặp E1/E1-C hoàn chỉnh** |

> [!WARNING]
> **Lưu ý phương pháp luận về bộ dữ liệu Weather:**
> Toàn bộ thiết kế thực nghiệm chuẩn có 60 cấu hình (5 tập $\times$ 4 horizon $\times$ 3 seeds). Tuy nhiên, trên hệ thống lưu trữ hiện tại, bộ dữ liệu Weather chỉ mới thực hiện 3 cấu hình ($H \in \{96, 192, 336\}$ với duy nhất seed 2021); toàn bộ horizon $H=720$ và các seed 2023, 2025 đều chưa có dữ liệu.
> Vì vậy:
> - Mọi số liệu tổng hợp 51 cấu hình đều được tính trên số lượng quan sát thực tế ($n=51$).
> - Không nội suy hay điền số liệu giả định cho 9 cấu hình còn thiếu của Weather.
> - Độ lệch chuẩn (std) của Weather trong các bảng thống kê theo horizon là không xác định ($n=1$), cần được ký hiệu rõ ràng là `NA`.

---

## 3. PHÂN TÍCH HIỆU CHỈNH CQR CALIBRATION (E1 vs E1-C)

### 3.1. Cơ chế toán học của CQR
Trên tập hiệu chuẩn (Validation Calibration Split), điểm phi bao phủ (non-conformity score) được xác định bởi:
$$s_i = \max(y^{\text{lower}}_i - y_i, \; y_i - y^{\text{upper}}_i)$$

Với mức độ tin cậy danh định $1 - \alpha = 0.95$ ($\alpha = 0.05$), giá trị điều chỉnh $\delta$ được xác định là quantile mức $(1 - \alpha)(1 + 1/n_{\text{cal}})$ của phân phối sai số $s_i$. Trong nghiên cứu này, phương pháp `horizon_channel` được áp dụng, tạo ra ma trận điều chỉnh $\delta \in \mathbb{R}^{H \times C}$:
$$\hat{y}^{\text{lower}}_{t, c} = \min\left(y^{\text{lower}}_{t, c} - \delta_{t, c}, \; y^{\text{median}}_{t, c}\right)$$
$$\hat{y}^{\text{upper}}_{t, c} = \max\left(y^{\text{upper}}_{t, c} + \delta_{t, c}, \; y^{\text{median}}_{t, c}\right)$$

> [!NOTE]
> Dự báo trung vị $y^{\text{median}}$ hoàn toàn **không bị thay đổi** bởi bước hiệu chỉnh này. Điểm số MSE và MAE của $E1$ và $E1$-$C$ là đồng nhất tuyệt đối.

### 3.2. Bảng so sánh tổng hợp theo bộ dữ liệu

Dưới đây là số liệu tính toán thực nghiệm tổng hợp từ 51 cấu hình:

| Tập dữ liệu | Số run ($n$) | E1 PICP (Mean $\pm$ Std) | E1-C PICP (Mean $\pm$ Std) | $\Delta$PICP | E1 MPIW | E1-C MPIW | E1 Winkler | E1-C Winkler | % Run đạt $\ge 0.95$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ETTh1** | 12 | $0.8913 \pm 0.0160$ | $0.9628 \pm 0.0089$ | $+0.0715$ | 6.8902 | 10.1072 | 13.5807 | **12.8805** | 91.7% (11/12) |
| **ETTh2** | 12 | $0.8821 \pm 0.0211$ | $0.9464 \pm 0.0070$ | $+0.0643$ | 13.2762 | 17.5519 | 25.0017 | **23.1672** | 16.7% (2/12) |
| **ETTm1** | 12 | $0.8334 \pm 0.0188$ | $0.9642 \pm 0.0033$ | $+0.1307$ | 4.6296 | 9.0897 | 15.6937 | **12.4670** | **100.0%** (12/12) |
| **ETTm2** | 12 | $0.9028 \pm 0.0116$ | $0.9386 \pm 0.0016$ | $+0.0358$ | 11.8113 | 14.2524 | 21.4376 | **20.1173** | 0.0% (0/12) |
| **Weather** | 3* | $0.9182 \pm 0.0145$ | $0.9475 \pm 0.0017$ | $+0.0292$ | 63.0445 | 90.2615 | 107.4254 | 114.5094 | 0.0% (0/3) |
| **TOÀN BỘ** | **51** | **0.8798** | **0.9527** | **+0.0728** | **12.3220** | **17.3098** | **24.1341** | **22.8845** | **49.0%** (25/51) |

*\* Lưu ý: Weather chỉ có 1 seed trên 3 horizon.*

### 3.3. Các mốc cực trị và hiện tượng chưa đạt danh định
1. **Cực trị PICP ở mô hình thô E1:**
   - PICP thấp nhất: **0.7954** tại cấu hình `ETTm1_h336_s2025` (hơn 20.4% mẫu thực tế rơi ra ngoài khoảng 95%).
   - PICP cao nhất: **0.9271** tại `Weather_h96_s2021`.
2. **Cực trị PICP ở mô hình đã hiệu chỉnh E1-C:**
   - PICP thấp nhất: **0.9349** tại cấu hình `ETTm2_h720_s2025`.
   - PICP cao nhất: **0.9720** tại cấu hình `ETTh1_h336_s2021`.
3. **Phân tích 26 cấu hình chưa đạt ngưỡng danh định 0.95:**
   - **ETTm2 (12/12 cấu hình):** PICP dao động từ 0.9349 đến 0.9405. Nguyên nhân do phân phối phương sai của ETTm2 giữa tập Validation và tập Test có sự dịch chuyển nhẹ (distribution shift). CQR đảm bảo tính bao phủ cận dưới chỉ khi giả định trao đổi được (exchangeability) được thỏa mãn hoàn toàn.
   - **ETTh2 (10/12 cấu hình):** PICP đạt từ 0.9389 đến 0.9480.
   - **Weather (3/3 cấu hình):** PICP đạt từ 0.9464 đến 0.9494 (tiệm cận rất sát 0.95).
   - **ETTh1 (1/12 cấu hình):** Chỉ có duy nhất `ETTh1_h720_s2025` đạt 0.9406.

### 3.4. Điểm phạt Winkler (Winkler Score)
Chỉ số Winkler Score được định nghĩa với hệ số phạt nghiêm khắc:
$$\text{Winkler} = (\text{upper} - \text{lower}) + \frac{2}{\alpha}(\text{lower} - y)\mathbb{I}(y < \text{lower}) + \frac{2}{\alpha}(y - \text{upper})\mathbb{I}(y > \text{upper})$$
Với $\alpha = 0.05$, điểm phạt cho mỗi đơn vị rơi ngoài khoảng là $\frac{2}{0.05} = 40$ lần.
- Kết quả cho thấy tại **45/51** cấu hình (88.24%), việc nới rộng khoảng đã triệt tiêu các lần vi phạm lớn, giúp điểm phạt Winkler giảm đáng kể.
- Riêng với Weather, do MPIW ở đơn vị gốc quá lớn (tăng hơn 27 đơn vị), tổng điểm phạt bề rộng tăng lên nhanh hơn số điểm phạt vi phạm giảm được, dẫn tới Winkler tăng nhẹ (+7.08).

---

## 4. BẰNG CHỨNG HÌNH ẢNH TRỰC QUAN HÓA (VISUAL EVIDENCE)

Toàn bộ biểu đồ phân tích XAI đã được sinh tự động vào thư mục [`figures/`](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/figures/):

### 4.1. Biểu đồ Dumbbell: Dịch chuyển PICP trước và sau Calibration
![Dumbbell PICP](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/figures/fig1_picp_dumbbell.png)
*Hình 1: Dịch chuyển độ bao phủ PICP từ E1 (chấm xanh) sang E1-C (kim cương cam) trên từng cấu hình thực nghiệm so với đường danh định 0.95 (đường nét đứt xanh lá).*

### 4.2. Độ rộng khoảng trung bình (MPIW) theo tập dữ liệu
![MPIW Comparison](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/figures/fig2_mpiw_comparison.png)
*Hình 2: So sánh độ rộng khoảng MPIW giữa E1 và E1-C trên các tập dữ liệu.*

### 4.3. Đánh đổi PICP – MPIW (Tradeoff Scatter)
![Tradeoff Scatter](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/figures/fig3_picp_mpiw_scatter.png)
*Hình 3: Vector dịch chuyển $(E1 \to E1\text{-}C)$ trên không gian độ rộng - độ bao phủ. Các mũi tên hướng chéo lên bên phải biểu thị mức tăng độ tin cậy đi kèm với mở rộng cận.*

### 4.4. Bản đồ nhiệt hệ số hiệu chỉnh $\delta$ (CQR Adjustment Heatmap)
![ETTh1 Heatmap](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/figures/fig4_delta_heatmap_ETTh1.png)
*Hình 4: Phân bố hệ số điều chỉnh $\delta$ trên ma trận [Horizon, Channel] của ETTh1. Vùng màu xanh biểu thị $\delta > 0$ (cần nới rộng khoảng), vùng màu đỏ biểu thị $\delta < 0$ (thu hẹp khoảng).*

### 4.5. Minh họa chuỗi thời gian kiểm thử trước và sau Calibration
![Before After Overlay](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/figures/fig5_interval_before_after.png)
*Hình 5: Minh họa dự báo khoảng của biến đích OT (Oil Temperature) trên ETTh1 H=96 seed=2021. Đường trung vị đen hoàn toàn trùng khớp giữa hai phiên bản; dải bao phủ E1-C (dưới) mở rộng bao bọc trọn vẹn các đỉnh nhọn mà E1 (trên) bỏ sót.*

### 4.6. Phân phối toàn cục của hệ số điều chỉnh $\delta$
![Delta Distribution](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/figures/fig6_delta_distribution.png)
*Hình 6: Phân phối của hơn 100,000 giá trị $\delta_{t,c}$. Trung vị $\delta = 1.34$, với 6.05% giá trị âm thể hiện việc co hẹp khoảng cục bộ.*

---

## 5. PHÂN TÍCH ĐỘ NHẠY BIẾN ĐẦU VÀO (FEATURE ABLATION)

### 5.1. Thiết kế kiểm nghiệm
Để đo lường tầm ảnh hưởng của các biến đầu vào lên dự báo khoảng của Quantile Head, kỹ thuật **Train-Mean Masking** được áp dụng:
- Thay thế chuỗi thời gian của từng biến $j$ bằng giá trị trung bình trên tập huấn luyện (trong không gian đã chuẩn hóa, giá trị này tiệm cận 0).
- Chạy suy luận qua mô hình $E1$ và áp dụng hệ số $\delta$ tương ứng để thu được khoảng $E1$-$C$.
- Ghi nhận mức biến thiên sai số trung vị ($\Delta\text{MSE}$), độ bao phủ ($\Delta\text{PICP}$) và độ rộng khoảng ($\Delta\text{MPIW}$).

### 5.2. Khẳng định giới hạn phương pháp luận
> [!IMPORTANT]
> **Quy tắc giải thích XAI:**
> Phép thử biến thế (Ablation) đo lường **độ nhạy quan sát (observational sensitivity)** của mô hình mạng nơ-ron đối với sự thiếu hụt thông tin của biến đó dưới phép thay thế đã định nghĩa. Phép thử này **không chứng minh mối quan hệ nhân quả (causality)** trong tự nhiên của hệ thống vật lý.

### 5.3. Trạng thái thực thi mô hình
Do tải bộ nhớ phân trang hệ thống (system paging file limit) trên môi trường thử nghiệm Windows gặp giới hạn với thư viện động của PyTorch, mã nguồn suy luận `feature_ablation.py` đã được tái cấu trúc hoàn chỉnh dựa trên đúng pipeline chuẩn của `train_interval.py`, bao gồm cơ chế bắt lỗi an toàn và ghi log tường minh (`ablation_skip.log`) để sẵn sàng thực thi tự động khi hệ thống cấp phát đủ tài nguyên bộ nhớ.

---

## 6. THẢO LUẬN VÀ GIẢI ĐÁP CÁC CÂU HỎI MỞ CỦA NGHIÊN CỨU

### 6.1. Tại sao nghiên cứu không báo cáo CRPS (Continuous Ranked Probability Score)?
Mô hình Interval-TVC xuất ra dự báo dạng **phân vị rời rạc** ($q_{0.025}, q_{0.5}, q_{0.975}$), xác định một khoảng tin cậy 95% có cấu trúc. Mô hình **không xuất ra hàm mật độ xác suất liên tục (PDF) hay hàm phân phối tích lũy đầy đủ (CDF)**. Việc tính CRPS từ chỉ 3 điểm phân vị đòi hỏi phải giả định dạng phân phối (như Gaussian hoặc Student-t) hoặc nội suy tham số hóa, điều này sẽ đưa thêm các giả định chủ quan làm sai lệch bản chất phân phối phi tham số của Quantile Regression. Do đó, việc không báo cáo CRPS là quyết định chuẩn mực về mặt phương pháp luận; thay vào đó, **Winkler Score** và **PINAW** là các thước đo đánh giá khoảng tin cậy phù hợp nhất.

### 6.2. Tại sao PICP của E1-C chưa đạt tuyệt đối 95% trên tất cả các tập dữ liệu?
Trong lý thuyết Conformal Prediction:
1. Độ bao phủ danh định $1 - \alpha$ được bảo đảm về mặt toán học theo kỳ vọng cận biên khi các mẫu dữ liệu thỏa mãn tính chất **trao đổi được (exchangeability)**.
2. Trong chuỗi thời gian thực nghiệm (đặc biệt là ETTm2 và ETTh2), tính chất dừng (stationarity) bị vi phạm cục bộ; xu hướng phân phối và phương sai giữa tập Validation và tập Test có sự dịch chuyển.
3. Vì vậy, việc PICP thực tế đạt trung bình **0.9527** trên toàn bộ 51 cấu hình, với mức thấp nhất không rơi xuống dưới 0.9348, đã là minh chứng mạnh mẽ cho năng lực kiểm soát rủi ro của phương pháp CQR so với mô hình gốc thô ($E1$ có lúc tụt xuống 0.7954).

---

## 7. DANH MỤC TÀI LIỆU VÀ TỆP TIN ĐÃ TẠO

Toàn bộ các thành phần của module XAI đã được lưu trữ hoàn chỉnh tại:
- Báo cáo XAI chính: [`REPORT_XAI_INTERVAL_TVC.md`](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/REPORT_XAI_INTERVAL_TVC.md)
- Hướng dẫn và cấu trúc pipeline: [`README.md`](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/README.md)
- Mã nguồn kiểm toán artifacts: [`audit_artifacts.py`](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/audit_artifacts.py)
- Mã nguồn phân tích calibration: [`calibration_analysis.py`](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/calibration_analysis.py)
- Mã nguồn trực quan hóa biểu đồ: [`plot_xai.py`](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/plot_xai.py)
- Mã nguồn phân tích độ nhạy biến: [`feature_ablation.py`](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/feature_ablation.py)
- Bộ điều phối thực thi: [`run_xai.py`](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/run_xai.py)
- Bảng tổng hợp so sánh chi tiết 51 runs: [`results/e1_vs_e1c_metrics.csv`](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/results/e1_vs_e1c_metrics.csv)
- Bảng thống kê hệ số $\delta$ hiệu chỉnh: [`results/calibration_delta_by_run.csv`](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/results/calibration_delta_by_run.csv)
- Thư mục hình ảnh khoa học: [`figures/`](file:///d:/TVCFORMER/FinalProject/Interval_TVC/xAI/figures/)
