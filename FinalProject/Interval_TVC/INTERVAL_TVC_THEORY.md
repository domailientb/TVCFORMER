# Interval-TVC: cơ sở lý thuyết, kiến trúc và công thức

## 1. Phạm vi và cách đọc tài liệu

Tài liệu này phân tích implementation trong `FinalProject/Interval_TVC/` và backbone TVCFormer được nạp qua `official_bridge.py`. Nội dung mô tả code hiện có, không khẳng định hiệu quả thực nghiệm. E0 là baseline point forecast; E1 là mô hình dự báo ba quantile; E1-C là các khoảng E1 sau hiệu chỉnh hậu nghiệm bằng CQR.

Các công thức được gắn nhãn **lý thuyết** khi phát biểu tính chất thống kê tổng quát, và **implementation** khi biểu diễn phép tính mà code thực hiện. Những tính chất conformal tiêu chuẩn dựa trên exchangeability không tự động trở thành bảo đảm cho chuỗi thời gian có cửa sổ chồng lấp.

## 2. Bài toán dự báo điểm và dự báo khoảng

Cho chuỗi đa biến lịch sử \(\mathbf X_t\in\mathbb R^C\). Với cửa sổ đầu vào độ dài \(T\), mô hình nhận \(\mathbf X_{t-T+1:t}\in\mathbb R^{T\times C}\) và dự báo \(H\) bước tiếp theo \(\mathbf Y_t=\mathbf X_{t+1:t+H}\in\mathbb R^{H\times C}\). Trên batch, shape đầu vào là \([B,T,C]\), target là \([B,H,C]\).

Point forecasting học một giá trị \(\hat y_{b,h,c}\) cho từng mẫu \(b\), bước dự báo \(h\), kênh \(c\), thường tối thiểu hóa sai số bình phương. Nó không trực tiếp mô tả độ bất định hay phân vị có điều kiện của đáp ứng. Prediction interval trả về hai biên \([\ell_{b,h,c},u_{b,h,c}]\), ngoài một ước lượng trung tâm; độ rộng và tỷ lệ bao phủ thực nghiệm là hai khía cạnh cần xem xét đồng thời.

Trong Interval-TVC, E1 trả về quantile thấp \(0.025\), median \(0.5\), quantile cao \(0.975\), tạo khoảng danh nghĩa 95%. Đây là các marginal quantile theo từng tọa độ bước dự báo–kênh, không phải một vùng tin cậy đồng thời xác suất 95% cho toàn bộ tensor \([H,C]\). Động cơ dùng quantile là dự báo phân bố có điều kiện thay vì chỉ dự báo một điểm; code không chứng minh rằng khoảng hẹp hơn hoặc bao phủ tốt hơn baseline.

## 3. Kiến trúc và luồng dữ liệu

### 3.1. Phần kế thừa và phần thay đổi

`IntervalTVCFormer` khởi tạo `TVCFormer.Model(configs)` rồi thay `detail_decoder` của backbone bằng `QuantileFlattenHead`. Cấu hình hiện tại yêu cầu đường dự báo chuẩn: `decomposition=False`, `na_head=False`, `last_value_residual=False`, `graph_gate=0`, không dùng pretrain head. Các lớp xử lý patch/token, encoder Transformer, GraphSAGE, graph generator, chuẩn hóa RevIN và cách chọn graph được tái sử dụng. Thay đổi chức năng chính nằm ở prediction head và loss huấn luyện E1.

### 3.2. TVC graph và encoder

Input \(\mathbf X\in\mathbb R^{B\times T\times C}\) được chuyển thành \(\mathbf Z\in\mathbb R^{B\times C\times T}\). Khi không có graph selector khác, code tính Pearson giữa cặp kênh trên context hiện tại và chọn các cặp theo `subgraph_keep_rate` (cùng tùy chọn self-cycle của backbone). Tập cạnh được chọn là batch-dependent. GraphGenerator lấy thống kê cục bộ của các kênh được chọn tại các time-node, tính cosine giữa vector thống kê cho mọi cặp time-node của cạnh đó, rồi dùng các tích chập 2D có kernel đối xứng, ReLU, tích chập 1×1, InstanceNorm và ReLU để tạo edge weights thưa. Những cạnh này được chuyển sang sparse COO; GraphSAGE dùng chúng để truyền thông tin giữa các local token.

Song song, context được padding ở đầu hoặc cuối để chia thành \(L\) time-node; mỗi time-node tạo các thống kê mean và standard deviation. Statistics đi qua LayerNorm và GraphGenerator. Giá trị chuỗi sau padding (và RevIN nếu bật) được cắt thành các detail patch. Linear patch embedding biến mỗi patch độ dài \(P\) thành vector chiều \(d\). Mỗi channel/time-node có local token được tạo từ detail token; positional encoding được cộng và dropout áp dụng. Mỗi encoder layer trước hết chạy temporal Transformer trên các token, sau đó GraphSAGE cập nhật local token theo graph; detail token được ghép lại. Backbone trả detail representation để prediction head giải mã.

Với ký hiệu \(L\) là `time_node_num`, \(P\) là `patch_len`, \(S\) là stride, \(M\) là số patch trên mỗi time-node và \(d\) là `d_model`, backbone tạo \(LM\) patch token cùng local token theo cấu trúc encoder. Số đặc trưng đầu vào của flatten head cho mỗi channel là \(D=LMd\). Representation decoder trước flatten có trục batch, channel, embedding, patch; sau flatten có shape \([B,C,D]\). Graph không thay đổi dự báo thành một output theo node graph: decoder vẫn cho dự báo theo channel và horizon.

### 3.3. Quantile prediction head và thứ tự quantile

Head tuyến tính (hoặc một head cho từng channel khi `individual=True`) xuất \(3H\) tham số thô trên mỗi channel. Sau reshape/permute, raw có shape \([B,C,H,3]\). Với mỗi tọa độ, raw cuối cùng được diễn giải là \((a,m,b)\): hai gap thô và median thô. Softplus \(\operatorname{sp}(x)=\log(1+e^x)\) tạo gap dương:

\[
\begin{aligned}
g^-_{b,h,c}&=\operatorname{sp}(a_{b,h,c}),\\
q_{0.025,b,h,c}&=m_{b,h,c}-g^-_{b,h,c},\\
q_{0.5,b,h,c}&=m_{b,h,c},\\
q_{0.975,b,h,c}&=m_{b,h,c}+\operatorname{sp}(b_{b,h,c}).
\end{aligned}\tag{1}
\]

Ở đây \(a,b,m\) là tham số chưa ràng buộc của head, \(g^-\) và \(g^+\) là gap dưới/trên. Do softplus dương, code bảo đảm \(q_{0.025}\le q_{0.5}\le q_{0.975}\) bằng tham số hóa; không cần penalty crossing. Hai gap không bị ép bằng nhau, nên khoảng có thể bất đối xứng quanh median. Head trả shape \([B,H,C,3]\), theo thứ tự lower, median, upper. Nếu RevIN bật, mỗi quantile được denormalize bằng cùng thống kê instance/channel.

Quantile crossing là hiện tượng dự báo các mức \(q_{\tau_1}>q_{\tau_2}\) dù \(\tau_1<\tau_2\). Tham số hóa (1) loại bỏ crossing giữa đúng ba quantile của code. Nó không áp đặt ràng buộc trơn theo horizon, hay bất kỳ quan hệ thứ tự nào giữa channel khác nhau.

## 4. Quantile regression và pinball loss

Quantile mức \(\tau\in(0,1)\) của biến ngẫu nhiên \(Y\) là một ngưỡng \(q_\tau\) sao cho xác suất nằm dưới ngưỡng xấp xỉ \(\tau\) (định nghĩa tổng quát dùng hàm phân phối và có thể có nhiều nghiệm). Quantile regression học quantile có điều kiện \(Q_\tau(Y\mid X)\). Median là quantile 0.5 và là dự báo trung tâm bền vững theo tiêu chuẩn sai số tuyệt đối; code vẫn báo MSE/MAE của median để mô tả point forecast.

Đặt sai số \(e=y-q\). Pinball/check loss lý thuyết cho một quantile là:

\[
\rho_\tau(e)=\max\{\tau e,(\tau-1)e\}
=\begin{cases}\tau e,&e\ge0,\\(1-\tau)(-e),&e<0.\end{cases}
\tag{2}
\]

Nếu dự báo thấp hơn quan sát (\(e>0\)), loss tỷ lệ \(\tau\); nếu dự báo cao (\(e<0\)), tỷ lệ \(1-\tau\). Tối thiểu hóa kỳ vọng loss tại mỗi mức \(\tau\) cho một quantile có điều kiện.

**Implementation.** Với \(N=BHC\) tọa độ và ba mức \(\mathcal Q=(0.025,0.5,0.975)\), code tính đều trung bình trên cả tọa độ và quantile:

\[
\mathcal L_{E1}=\frac{1}{3N}\sum_{b=1}^{B}\sum_{h=1}^{H}\sum_{c=1}^{C}\sum_{\tau\in\mathcal Q}
\max\left(\tau(y_{b,h,c}-q_{\tau,b,h,c}), (\tau-1)(y_{b,h,c}-q_{\tau,b,h,c})\right).
\tag{3}
\]

Không có trọng số khác nhau giữa ba quantile; đây là loss duy nhất trong E1 và code không cộng contrastive/triplet loss. E0 thay bằng \(\mathcal L_{E0}=\frac{1}{BHC}\sum(y-\hat y)^2\). Vì objective và output khác nhau, median E1 không được tối ưu trực tiếp theo MSE.

## 5. Conformalized Quantile Regression (CQR)

### 5.1. Nguyên lý và điều kiện bảo đảm

Split conformal cố định mô hình đã fit, chấm điểm một tập calibration không dùng để fit mô hình, lấy phân vị hữu hạn mẫu của score, rồi nới tập dự báo. Với các đơn vị calibration và test exchangeable, quy tắc phân vị thứ tự cho bảo đảm coverage marginal hữu hạn mẫu ít nhất \(1-\alpha\) (ngoại trừ xử lý trường hợp nhóm quá nhỏ trong code). CQR dùng hai biên quantile làm interval ban đầu và score không phù hợp đo mức target nằm ngoài khoảng.

Các cửa sổ chuỗi thời gian ở đây có phụ thuộc thời gian và có thể chồng lấp. Chia theo thời gian giúp tránh trộn ngẫu nhiên tương lai/quá khứ và code chèn gap để target ranges của monitor và calibration tách nhau; việc này không làm dữ liệu trở thành exchangeable. Vì vậy E1-C trong dự án là hiệu chỉnh thực nghiệm hậu xử lý; không nên tuyên bố bảo đảm conformal chuẩn 95% cho chuỗi này.

### 5.2. Chia validation và cách tạo calibration scores

Giả sử validation có \(n_v\) forecast windows, horizon \(H\), và `monitor_fraction` \(r\). Code đặt \(n_m=\lfloor r n_v\rfloor\), monitor indices \(0,\ldots,n_m-1\), bỏ \(H\) windows tiếp theo, rồi lấy calibration indices bắt đầu ở \(n_m+H\). Calibration cần tối thiểu `min_calibration_windows` windows (mặc định 100). Monitor dùng early stopping; calibration chỉ dùng sau khi đã khôi phục checkpoint có validation monitor loss tốt nhất.

Từ quantiles dự báo cho calibration windows, code inverse-scale lower/upper và target trước khi tính score. Với \(i\) là calibration window, \(h\) horizon step, \(c\) channel:

\[
s_{i,h,c}=\max\{\ell_{i,h,c}-y_{i,h,c},\ y_{i,h,c}-u_{i,h,c}\}.
\tag{4}
\]

Score âm khi target nằm trong khoảng và có khoảng cách tới biên; score dương khi target nằm ngoài. Đây là signed CQR score, khác score không âm \(\max(\ell-y,y-u,0)\) thường dùng trong một số trình bày CQR. Vì score có thể âm, hiệu chỉnh code có thể thu hẹp interval nếu quantile calibration score tương ứng âm; đây là hành vi được phép bởi phép tính cụ thể.

### 5.3. Nhóm score, phân vị hữu hạn mẫu và nới biên

Code hỗ trợ bốn kiểu nhóm: `global` gộp tất cả \((i,h,c)\); `channel` lấy score theo \(c\), gộp \((i,h)\); `horizon` lấy theo \(h\), gộp \((i,c)\); `horizon_channel` lấy theo cặp \((h,c)\), gộp qua \(i\). Với mỗi nhóm có \(n_g\) score, thứ hạng được tính:

\[
k_g=\left\lceil(n_g+1)(1-\alpha)\right\rceil,\qquad
\delta_g=s_{(k_g),g},
\tag{5}
\]

trong đó \(s_{(k_g),g}\) là phần tử thứ \(k_g\) sau khi sắp xếp tăng dần. Implementation dùng order statistic với index zero-based \(k_g-1\); nếu \(k_g>n_g\), code báo lỗi nhóm quá nhỏ thay vì clip thứ hạng. Mặc định danh nghĩa \(1-\alpha=0.95\), method cấu hình chính là `horizon_channel`; tất cả bốn method vẫn được fit và báo cáo.

Trên test (hoặc bất kỳ dự báo nào được hiệu chỉnh), adjustment được broadcast theo nhóm tương ứng, giữ nguyên median \(m\):

\[
\ell'_{i,h,c}=\min\{\ell_{i,h,c}-\delta_{g(h,c)},m_{i,h,c}\},\qquad
u'_{i,h,c}=\max\{u_{i,h,c}+\delta_{g(h,c)},m_{i,h,c}\}.
\tag{6}
\]

Với `global`, \(\delta\) là một scalar; `channel` là \(\delta_c\); `horizon` là \(\delta_h\); `horizon_channel` là \(\delta_{h,c}\). Dùng `min/max` bảo đảm median vẫn nằm trong khoảng ngay cả khi \(\delta<0\). Code không cập nhật median, tham số mạng hoặc checkpoint sau calibration. Do đó không cần và không thực hiện train lại.

## 6. Quy trình huấn luyện, validation, inverse scaling và đánh giá

1. Dataset loader tạo các cửa sổ theo split thời gian; ETTh/ETTm dùng split chuẩn trong loader, Custom (Weather) dùng các biên theo tỷ lệ train/validation/test. `StandardScaler` được fit trên đoạn train và áp dụng cho các split; `scale=true` trong cấu hình. Batch lấy \(x\) và phần target cuối \(H\) của `batch_y`; marker thời gian trả kèm không được đưa vào model.
2. E0 gọi TVCFormer chuẩn; E1 gọi IntervalTVCFormer. Adam cập nhật theo MSE (E0) hoặc pinball (E1), sau `zero_grad`, `backward`, `step`; learning rate được điều chỉnh bằng lịch `adjust_learning_rate`. Mỗi epoch đánh giá loss trên validation monitor. Checkpoint lưu khi validation loss thấp hơn tốt nhất; early stopping theo `patience`.
3. Nạp checkpoint tốt nhất. Với E1, dự báo calibration split từ phần validation còn lại, inverse-scale các biên và target, fit bốn calibrator. Sau đó dự báo test, inverse-scale target và từng quantile theo scaler đã fit ở train. Không sử dụng test để fit calibration.
4. Tính point metrics từ median; interval metrics từ lower/upper. Lưu báo cáo theo đơn vị gốc, quantiles, split calibration, tham số adjustment, forecast arrays và hình minh họa. E1-C là kết quả hiệu chỉnh theo method cấu hình chính, mặc định `horizon_channel`.

Inverse scaling áp dụng scaler theo channel: nếu \(z\) là scaled value, `inverse_transform` khôi phục \(y\) theo tham số train-fitted của StandardScaler. Khi scaler affine, phép biến đổi đơn điệu dương nên thứ tự quantile và coverage không đổi; width được đổi về đơn vị gốc. Calibration được fit sau inverse scaling và test metrics cũng được tính sau inverse scaling.

## 7. Chỉ số đánh giá trong code

Đặt \(N=BHC\), target \(y\), median \(m\), lower \(\ell\), upper \(u\). Point metrics là:

\[
\operatorname{MSE}=\frac1N\sum(m-y)^2,\qquad
\operatorname{MAE}=\frac1N\sum|m-y|.
\tag{7}
\]

Đặt indicator bao phủ \(I_{b,h,c}=\mathbf1[\ell_{b,h,c}\le y_{b,h,c}\le u_{b,h,c}]\), width \(w=u-\ell\), nominal coverage \(p_0=1-\alpha\). Các phép tính implementation:

\[
\begin{aligned}
\operatorname{PICP}&=\frac1{BHC}\sum_{b,h,c}I_{b,h,c},\\
\operatorname{MPIW}&=\frac1{BHC}\sum_{b,h,c}w_{b,h,c},\\
R_c&=\max_{b,h}y_{b,h,c}-\min_{b,h}y_{b,h,c},\\
\operatorname{PINAW}&=\frac1C\sum_{c=1}^C\frac{\frac1{BH}\sum_{b,h}w_{b,h,c}}{R_c},\\
\operatorname{ACE}&=|\operatorname{PICP}-p_0|.
\end{aligned}\tag{8}
\]

Nếu \(R_c=0\), implementation đặt thành phần PINAW channel đó bằng 0. PICP là coverage tổng hợp trên mọi mẫu, horizon và channel; code cũng báo coverage theo channel, horizon step và horizon-channel. MPIW là độ rộng trung bình đơn vị gốc. PINAW chuẩn hóa width theo range của target test từng channel rồi lấy trung bình channel; đây không phải chuẩn hóa theo một range chung. ACE đo sai lệch tuyệt đối coverage tổng hợp so với danh nghĩa.

Coverage cao hơn thường đòi khoảng rộng hơn; độ rộng thấp hơn có lợi khi coverage giữ gần mức mục tiêu. Không nên diễn giải riêng PICP hoặc MPIW mà bỏ qua chỉ số còn lại. Pinball là objective huấn luyện E1; MSE/MAE là đánh giá median; PICP/MPIW/PINAW/ACE đánh giá khoảng. Đây là metric thực nghiệm trên test, không phải đảm bảo coverage.

## 8. Các phiên bản thí nghiệm

| Phiên bản | Đầu ra | Loss dùng train và chọn checkpoint | Hiệu chỉnh |
|---|---|---|---|
| E0 | \([B,H,C]\) point forecast | MSE | Không |
| E1 | \([B,H,C,3]\), quantile 0.025/0.5/0.975 | Mean pinball loss | Chưa hiệu chỉnh |
| E1-C | median E1 và interval điều chỉnh | Dùng checkpoint E1; không train riêng | Signed CQR trên validation calibration split |

E0 dùng cùng cấu trúc TVCFormer chuẩn trong code. E1 dùng cùng backbone nhưng đổi head và objective. E1-C không phải kiến trúc hay quá trình train khác; nó là hậu xử lý lower/upper của E1. So sánh chất lượng E0/E1 và E1/E1-C cần dựa trên kết quả paired cùng dữ liệu, horizon và seed; tài liệu code không cung cấp bằng chứng kết quả.

## 9. Giới hạn diễn giải

- Quantile dự báo là marginal theo từng vị trí horizon-channel; việc mọi vị trí đều có khoảng 95% không đồng nghĩa toàn bộ quỹ đạo đa biến đồng thời có xác suất bao phủ 95%.
- Tham số hóa softplus ngăn crossing giữa ba quantile, nhưng không bảo đảm quantile calibration hay phân phối được mô hình hóa chính xác.
- CQR finite-sample coverage chuẩn giả định exchangeability của calibration/test units. Chuỗi có phụ thuộc, drift và cửa sổ forecast chồng lấp; gap trong validation chỉ loại bỏ một vùng chồng lấp giữa monitor và calibration, không xóa phụ thuộc chuỗi. Không khẳng định bảo đảm coverage hữu hạn mẫu cho dữ liệu này.
- `horizon_channel` tạo một nhóm riêng cho từng tọa độ horizon-channel và cần đủ calibration scores. Code yêu cầu ít nhất 100 windows tổng thể, đồng thời còn yêu cầu thứ hạng hữu hạn mẫu không vượt số score của từng nhóm.
- Signed score có thể âm và adjustment âm có thể thu hẹp interval. Đây là hệ quả của implementation score (4), không phải lỗi dấu trong công thức.
- Validation monitor loss là MSE cho E0 và pinball cho E1; do objective khác nhau, giá trị validation loss giữa E0/E1 không phải so sánh cùng thang đo.
- Không suy luận nhân quả, quan hệ vật lý, tính đảm bảo coverage ngoài điều kiện, hoặc ưu thế chất lượng nếu không có phân tích và kết quả tương ứng.

## 10. Đề xuất hình cho báo cáo

1. **Sơ đồ kiến trúc tổng thể.** Vẽ input \([B,T,C]\), Pearson candidate pairs và TVC graph/GraphGenerator, patch/local token embedding, các lớp lặp Transformer → GraphSAGE, flatten quantile head, RevIN denorm, output \([B,H,C,3]\). Tô nổi head/loss là phần E1 bổ sung, ghi chú graph/backbone được tái sử dụng. Đặt ở phần phương pháp/kiến trúc.
2. **Minh họa quantile và prediction interval.** Trên một forecast window, vẽ target, median, vùng giữa \(q_{0.025}\) và \(q_{0.975}\); ghi rõ mức nominal 95%, không gọi là simultaneous trajectory interval. Đặt sau phần quantile regression.
3. **Lưu đồ calibration CQR.** Thể hiện validation chronological split → monitor, gap \(H\), calibration → checkpoint tốt nhất → inverse-scale → score (4) theo nhóm → order statistic (5) → nới/thu biên (6) → test evaluation. Tô test chỉ ở bước đánh giá để nhấn mạnh không fit calibrator trên test. Đặt ở phần CQR.

## 11. Bảng ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| \(B,T,C,H\) | Batch size, độ dài context, số kênh, prediction horizon |
| \(L,P,S,M,d\) | Số time-node, patch length, stride, patch mỗi time-node, chiều embedding |
| \(\mathbf X,\mathbf Y\) | Input context và target tương lai |
| \(b,h,c,i\) | Chỉ số mẫu batch, horizon step, channel, calibration window |
| \(\tau,\mathcal Q\) | Mức quantile và tập mức \((0.025,0.5,0.975)\) |
| \(q_\tau,m,\ell,u\) | Quantile mức \(\tau\), median, cận dưới, cận trên |
| \(a,b\) | Raw lower-gap và upper-gap parameter (chữ \(b\) ở đây không phải batch index) |
| \(g^-,g^+\) | Gap dương dưới và trên, được tạo bằng softplus |
| \(\rho_\tau\) | Pinball loss tại quantile \(\tau\) |
| \(\alpha,p_0\) | Mức sai số danh nghĩa và nominal coverage \(p_0=1-\alpha\) |
| \(s_{i,h,c}\) | Signed nonconformity score của CQR |
| \(n_g,k_g,\delta_g\) | Số score nhóm, thứ hạng hữu hạn mẫu, adjustment nhóm |
| \(I,w,R_c\) | Coverage indicator, interval width, range target channel c |
| MSE/MAE | Sai số bình phương trung bình / tuyệt đối trung bình của median |
| PICP/MPIW/PINAW/ACE | Coverage, width trung bình, width chuẩn hóa trung bình, sai lệch coverage tuyệt đối |

## 12. Danh sách công thức trọng tâm

1. Tham số hóa quantile có thứ tự: (1).
2. Pinball loss một quantile và mean loss triển khai: (2)–(3).
3. Signed CQR nonconformity score: (4).
4. Phân vị order statistic hữu hạn mẫu theo nhóm: (5).
5. Hiệu chỉnh lower/upper và giữ median: (6).
6. Point metrics, interval coverage và width: (7)–(8).

## 13. Tóm tắt thuật toán

```text
Input context x [B,T,C]
  → chọn channel pairs theo context hiện tại
  → thống kê local time-node và GraphGenerator tạo edge weights
  → patch embedding + local tokens
  → lặp temporal Transformer và GraphSAGE
  → flatten decoder tạo raw [B,C,H,3]
  → softplus gaps sắp thứ tự quantile, RevIN denormalize
Train E1 bằng mean pinball loss trên ba quantile
  → theo dõi pinball validation, lưu checkpoint tốt nhất
  → dự báo validation calibration sau chronological split và gap
  → inverse-scale, fit signed CQR adjustment theo nhóm
  → dự báo test, inverse-scale, áp adjustment mà không đổi median
  → đánh giá MSE/MAE và PICP/MPIW/PINAW/ACE
```

## 14. Tài liệu tham khảo

1. Koenker, R. & Bassett, G. (1978). “Regression Quantiles.” *Econometrica*, 46(1), 33–50. DOI: [10.2307/1913643](https://doi.org/10.2307/1913643). Nền tảng quantile regression và check/pinball loss.
2. Romano, Y., Patterson, E. & Candès, E. J. (2019). “Conformalized Quantile Regression.” *Advances in Neural Information Processing Systems 32*. [Bài báo NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html). Kết hợp quantile regression và split conformal.
3. Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. [DOI](https://doi.org/10.1007/b106715). Cơ sở conformal prediction và giả định exchangeability.
4. Shafer, G. & Vovk, V. (2008). “A Tutorial on Conformal Prediction.” *Journal of Machine Learning Research*, 9, 371–421. [JMLR](https://www.jmlr.org/papers/v9/shafer08a.html). Tổng quan conformal prediction.
5. TVCFormer backbone. Mã nguồn trong repository này: `official/models/TVCFormer.py`, `official/layers/TVCFormer_backbone.py`, `official/layers/TVC_Encoder.py`, `official/layers/GraphGenerator.py`, `official/layers/Graph_layers.py`.
6. Interval-TVC implementation. `FinalProject/Interval_TVC/models/interval_tvcformer.py`, `losses/quantile_loss.py`, `calibration/conformal.py`, `metrics/interval_metrics.py`, `train_interval.py`.
