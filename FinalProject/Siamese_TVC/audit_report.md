# Siamese_TVC: rà soát mã nguồn và thiết kế

**Phạm vi:** Giai đoạn 2, chỉ đọc mã và tạo báo cáo. Không sửa model, baseline hay cấu hình; không chạy benchmark. Kết luận dưới đây dựa trên implementation hiện tại, không dựa riêng vào README.

## Tóm tắt

Siamese_TVC hiện có encoder dùng chung thật sự, nhận hai loại đầu vào cho mỗi nút thời gian: hai thống kê TVC đã chuẩn hóa và sáu descriptor tính từ đoạn chuỗi cục bộ tương ứng. Embedding L2-normalized được so sánh bằng cosine, trộn với cosine thống kê bằng gate, sau đó đi qua PCFS/sparse graph và GraphSAGE. Forecasting MSE có đường gradient tới encoder và các tham số graph generator.

Thiết kế hiện tại đủ để chạy một nghiên cứu dự báo E1/E2 ở cấp mã, nhưng chưa đủ bằng chứng để nói embedding học được quan hệ chuỗi có ý nghĩa độc lập với mục tiêu dự báo. Quan trọng hơn, support cạnh vẫn cố định theo Pearson Top-K; Siamese chỉ điều chỉnh trọng số trong support đó. Chưa có benchmark E0/E1/E2 hoàn chỉnh, đánh giá embedding độc lập, đồ thị qua seed, hoặc unit tests trong thư mục Siamese_TVC.

## 1. Kiến trúc Siamese hiện tại

Trong `models/similarity.py`, `SiameseWindowSimilarity` có **một** `nn.Sequential` encoder được gọi trên toàn bộ tensor `[B,C,L,F]`. Không có hai module nhánh riêng; hai phía của phép so sánh dùng cùng tensor embedding, do đó trọng số được chia sẻ.

MLP hiện tại là `Linear(statistics_dim + 6, hidden_dim) -> GELU -> Linear(hidden_dim, embedding_dim)`. `WindowFeatures.forward` tính sáu descriptor từ từng cửa sổ: mean, log(std), độ dốc hai đầu, độ lệch chuẩn sai phân chia std, mean absolute difference chia std, và lag-1 correlation. Encoder nhận nối thống kê với các descriptor này. Đây là MLP phi tuyến trên đặc trưng đã trích xuất; nó không nhận chuỗi thô trực tiếp như một chuỗi thời gian qua Conv1D/Transformer. Nó có thể học tổ hợp phi tuyến của descriptor, nhưng khả năng biểu diễn bị giới hạn bởi sáu descriptor và hai thống kê.

Embedding được `F.normalize(..., dim=-1)` rồi cosine qua nhân ma trận trong `edge_weights`. Chuẩn hóa này khống chế độ dài vector, không ngăn nhiều embedding có cùng hướng/collapse.

## 2. Luồng dữ liệu và graph

Luồng thực tế:

1. `TVCFormer_backbone.forward` nhận `z` dạng `[B,C,T]` từ cửa sổ đầu vào. Pearson và Top-K tính trên `z`.
2. `get_statistics_full` đệm chuỗi rồi chia thành `time_node_num` khối bằng `torch.chunk`, tạo mean/std cho từng kênh và nút thời gian. `get_statistics_block` detach mean/std trước khi trả ra. Thống kê sau đó qua `statistic_norm`.
3. Vì `Model` trong `models/tvcformer_siamese.py` đặt `configs.sdei = 1`, backbone dựng `windows` bằng cách đệm cùng `z` rồi chia thành cùng số khối. Do đó cửa sổ Siamese là các đoạn cục bộ thật tương ứng với time-node; không phải future window. Sáu descriptor được tính từ các cửa sổ này.
4. `SiameseWindowSimilarity.forward` mã hóa mọi `[B,C,L]` cục bộ bằng shared MLP. `edge_weights` chỉ gather các kênh trong `index_vector`, tính ma trận cosine embedding `[B,K,L,L]` và cosine thống kê cùng shape.
5. Gate sigmoid trộn hai cosine. Ở mode `learned`, gate học được; `statistical` ép gate về 0 và `siamese` ép gate về 1. Hai mode ablation không đóng băng/thay đổi encoder.
6. `GraphGeneratorSiamese.forward` áp dụng `sparse_module` kế thừa từ graph generator chính thức (PCFS), đổi local weights thành sparse COO qua `subgraph_tensor_to_full_graph_sparse_tensor`, rồi trả index/weight cho encoder.
7. `TVC_Encoder.TSTEncoder.forward` tạo DGL graph từ COO và truyền edge weights cho `dgl_graph_for_trans`/`SAGEConv` ở từng lớp. Local tokens là node features; detail tokens được ghép lại sau GraphSAGE.

`get_statistics_full` và `cut_detail_patches` là hai luồng khác nhau: thống kê/nút dùng khối chia đều, còn detail patches được unfold trong từng khối. Việc Siamese dùng đúng khối của time-node là nhất quán với graph, dù descriptor không trực tiếp mã hóa từng patch nhỏ.

## 3. Support, đối xứng, self-loop và gradient

`top_k_channel_pairs_choose` được gọi trước graph generator, trong `TVCFormer_backbone.forward`. Support là rời rạc và không học được. Hàm chọn các hệ số Pearson lớn nhất trên nửa ma trận rồi phản chiếu support, nên các cặp thuận/nghịch cùng có mặt. Khi `graph_keep_self_cycles: false`, đường chéo kênh bị loại; config Siamese hiện đặt giá trị này thành false.

Do đó Siamese **không thể thêm cạnh cho cặp kênh nằm ngoài Pearson Top-K**. Ngoài ra, chọn giá trị Pearson lớn nhất thiên về tương quan dương; cặp có tương quan âm mạnh có thể bị loại dù có quan hệ nghịch đảo đáng kể. Đây là đặc tính kế thừa baseline, cần giữ cố định nếu so sánh trên cùng support, nhưng phải được nêu rõ khi diễn giải phạm vi cải tiến.

Với một cặp support, cosine embedding và cosine thống kê đều đối xứng theo hoán vị hai kênh; `get_conv_weights('symmetry')` tạo kernel đối xứng cho PCFS. Khi support đối xứng, sparse weights và graph dự kiến đối xứng. Tuy nhiên implementation không có assertion/kiểm tra tự động cho tính đối xứng sau sparse conversion. Self-loop thời gian trong ma trận `[L,L]` có thể xuất hiện từ similarity/PCFS; `graph_keep_self_cycles` chỉ kiểm soát self-loop ở mức kênh trong support, không loại đường chéo temporal. Config hiện không bật channel self-loop.

Gradient từ forecast loss đi theo đường `MSE -> forecast -> GraphSAGE (edge weights) -> sparse values -> PCFS -> mixed similarity -> embedding -> shared MLP/gate`. `index_vector`, Pearson sort/Top-K và các chỉ số sparse là quyết định rời rạc, không có gradient. `nonzero()` trong sparse conversion cũng làm support giá trị cạnh rời rạc theo vị trí khác 0. Điều này không chặn gradient của các giá trị cạnh đang tồn tại: sparse values được gather từ `local_weights`, còn DGL SAGEConv nhận chúng làm edge weights. `last_gate` bị detach chỉ để logging; giá trị gate dùng cho forward vẫn có gradient.

## 4. Huấn luyện và nguy cơ biểu diễn không hữu ích

`Exp_Siamese.train` dùng forecasting criterion từ `Exp_Main` (MSE), optimizer/scheduler chính thức, train loader và validation loss để early stopping/chọn checkpoint. Không có contrastive loss, positive/negative pair giả, hay cập nhật trên test. Loss dự báo có tác động trực tiếp tới encoder nếu cạnh có ảnh hưởng tới output. Mã hiện không đo gradient norm của encoder, do vậy điều này cần unit/smoke check thay vì chỉ suy luận từ graph code.

Forecast loss chỉ đảm bảo embedding được tối ưu theo dự báo trong pipeline này. Nó không đảm bảo cosine thể hiện quan hệ hình dạng độc lập, và có thể tạo embedding collapse hoặc gate gần hẳn 0/1 nếu dự báo vẫn tốt theo nhánh còn lại. L2 normalization không ngăn collapse. Hiện không có log phân phối cosine, embedding norm trước normalize, effective rank/variance, gate qua epoch, hay ổn định qua seed.

Kiểm tra độc lập nên được định nghĩa trước khi xem test: (a) phân phối cosine giữa cặp support và giữa các cặp không thuộc support; (b) độ phân tán embedding theo kênh/time-node, effective rank và tỷ lệ cặp cosine gần 1; (c) tương quan/độ ổn định ma trận cosine giữa seed; (d) trên validation, tạo cặp cửa sổ từ train/validation context không chồng lấn và đánh giá ranking theo tiêu chí hình dạng độc lập với Siamese (ví dụ khoảng cách trên sai phân đã chuẩn hóa hoặc DTW, chỉ dùng như phân tích hậu nghiệm, không làm nhãn huấn luyện); (e) ablation thống kê, Siamese và hybrid với cùng support. Không nên gọi cặp “tương đồng” chỉ vì Pearson cao rồi dùng chính nó làm nhãn xác nhận Siamese.

## 5. Leakage và tính công bằng

`Dataset_ETTh` trong `official/data_provider/data_loader.py` fit `StandardScaler` trên đoạn train rồi transform toàn bộ chuỗi. Validation/test có thể dùng `seq_len` ngữ cảnh ngay trước ranh giới split theo thiết kế forecasting; target tương lai vẫn thuộc split tương ứng. `Exp_Siamese.train` chỉ gọi train và val; `Exp_Siamese.test` chỉ chạy khi runner nhận `--do_test`. Runner không tự chạy test mỗi epoch. Vì vậy luồng được khảo sát không đưa nhãn test vào huấn luyện Siamese và graph chỉ nhìn input context của từng batch.

Tính công bằng giữa E0/E1/E2 chưa được chứng minh bằng pipeline benchmark: YAML hiện có chỉ là một cấu hình ETTh1 horizon 96; không có script benchmark/ma trận config trong thư mục. E0 phải dùng đúng split, batch, seed, optimizer, lr, epochs/patience, scaler và model hyperparameters; E1/E2 cần thay duy nhất graph similarity/gate tương ứng. Ghi nhận `Model.__init__` thay generator sau khi dựng official model, và `Exp_Siamese.train` chọn checkpoint chỉ bằng validation. Số tham số khác do encoder Siamese/gate là thay đổi thuộc can thiệp cần báo cáo. Runner seed Python/NumPy/PyTorch/CUDA và thử DGL seed; deterministic kernels chưa được bật, nên seed không đảm bảo bitwise deterministic.

## 6. Phát hiện theo mức độ

| Mức | File / class / function | Phát hiện và ảnh hưởng | Cách kiểm chứng / xử lý |
|---|---|---|---|
| Cao | `models/graph_generator.py:GraphGeneratorSiamese.forward`; `official/utils/correlation_utils.py:top_k_channel_pairs_choose` | Siamese chỉ học trọng số trong support Pearson Top-K; không khôi phục quan hệ phi tuyến bị Pearson loại. Top-K theo giá trị lớn nhất cũng có thể bỏ tương quan âm mạnh. | Báo cáo rõ estimand là reweighting trong Pearson support. Giữ support giống nhau cho E0/E2; nghiên cứu learned support/union chỉ ở ablation tách biệt để không làm lẫn hai thay đổi. |
| Cao | `models/similarity.py:SiameseWindowSimilarity`; `Exp_Siamese.train` | Forecast loss không xác nhận embedding mang ý nghĩa quan hệ độc lập; không có collapse/geometry diagnostics. | Thêm phân tích cosine, variance/effective rank, seed stability và phép đối chiếu hình dạng độc lập trên validation trước khi diễn giải. |
| Trung bình | `models/tvcformer_siamese.py:Model.__init__`; `official/layers/TVCFormer_backbone.py:forward` | Dùng cờ `sdei` để yêu cầu backbone chuẩn bị local windows; tên/cơ chế cờ còn gắn với module SDEI/contrastive cũ. Không thấy contrastive loss được dùng trong `Exp_Siamese`, nhưng phụ thuộc vào đường code legacy làm tích hợp khó hiểu. | Tách cờ nội bộ `return_windows`/kiểu graph generator hoặc ghi rõ adapter; bảo đảm không kích hoạt nhánh loss SDEI ngoài ý muốn. Không cần sửa baseline để làm điều này. |
| Trung bình | `models/graph_generator.py`; `official/utils/graph_utils.py`; `official/layers/TVC_Encoder.py` | Tính đối xứng được đảm bảo bởi cấu trúc dự kiến, không có assertion; sparse conversion và DGL weight alignment chưa được unit-test tại thư mục Siamese. | Test graph nhỏ: số node/edge, weights khớp COO, đối xứng, channel/temporal self-loop và backward gradient tới encoder/PCFS/gate. |
| Trung bình | `run_siamese.py`; `configs/ETTh1_pl96.yaml`; `exp_siamese.py` | Chỉ có cấu hình ETTh1/96 và runner đơn lẻ; chưa có E0/E1/E2 matrix, bốn horizon, nhiều seed, benchmark thời gian/memory, kiểm tra graph. | Bổ sung cấu hình/benchmark riêng ở giai đoạn tiếp theo; khóa tham số chung và lưu metadata từng run. Chưa thể kết luận fairness/toàn bộ tiêu chí yêu cầu chỉ từ một config. |
| Thấp | `models/similarity.py:WindowFeatures.forward` | Descriptor phụ thuộc sáu thống kê thủ công; không phải encoder chuỗi thô, nên có thể bỏ lỡ các mẫu hình khác nhau nhưng cùng descriptor. | Đối chiếu các cặp shape độc lập; nếu cần kiến trúc encoder chuỗi, xem đó là ablation mới, không thay đổi âm thầm cấu hình chính. |
| Thấp | `run_siamese.py:main` | Có lặp khóa `use_multi_gpu` và `devices` trong YAML; runner seed thông thường nhưng không bật chế độ deterministic. | Dọn config khi triển khai; báo cáo biến thiên qua seed và cấu hình phần cứng. |

## 7. Phần đúng nên giữ

- Shared encoder là một module dùng chung; embedding được chuẩn hóa L2 trước cosine.
- Positive/negative labels tùy tiện và test labels không được dùng.
- Siamese dùng local windows lấy trực tiếp từ input context; không lấy future target.
- Pearson Top-K giữ nguyên support giúp đối chứng graph reweighting có ngân sách cạnh tương đương baseline.
- Gate có ba chế độ statistical/Siamese/learned, thuận tiện cho ablation E0-like graph, E1 và E2.
- PCFS, sparse conversion, DGL/GraphSAGE và transformer backbone chính thức được tái sử dụng thay vì viết lại.
- Checkpoint được chọn từ validation; test chỉ chạy tường minh bằng `--do_test`.
- Scaler ETTh fit trên train split.

## 8. Phương án cuối cùng đề xuất

1. Giữ E0 baseline nguyên trạng. E1 dùng `siamese_gate_mode: siamese`; E2 dùng `learned`. Với E1/E2 giữ nguyên Pearson Top-K, PCFS, backbone, split và training protocol.
2. Trước benchmark, viết unit tests cho shape, đối xứng/self-loop, support cố định, sparse/DGL alignment và backward gradient; kiểm tra gradient norm của encoder/gate/PCFS khác None và hữu hạn.
3. Bổ sung log/audit đồ thị và embedding: phân phối edge weights, density/degree, cosine geometry, gate, graph similarity qua seed, thời gian graph/train/inference và peak GPU memory.
4. Tạo benchmark paired E0/E1/E2 trên ETTh1 với input 336, horizons 96/192/336/720, cùng nhiều seed và mọi cấu hình tối ưu hóa/early stop. Chọn checkpoint theo validation, chỉ mở test sau khi giao thức đã khóa.
5. Không gọi graph tốt hơn chỉ từ phân phối weights đẹp hoặc forecast metric đơn lẻ. Báo cả chênh lệch seed và phân tích cặp shape độc lập; nếu E1/E2 không cải thiện, báo cáo như kết quả âm hợp lệ.

## 9. Năm câu trả lời bắt buộc

1. **Siamese hiện đã thực sự học biểu diễn quan hệ giữa các đoạn chuỗi chưa?** Nó học embedding phi tuyến từ thống kê và sáu descriptor của từng local window, với gradient từ forecast loss. Chưa có bằng chứng hiện tại rằng embedding học quan hệ tổng quát/độc lập hoặc hữu ích hơn thống kê; cần diagnostics và thí nghiệm kiểm chứng.
2. **Embedding đã được dùng thực sự để tạo trọng số cạnh chưa?** Có. Cosine embedding được trộn với cosine thống kê cho mỗi cặp kênh được chọn, rồi qua PCFS để thành weights cho TVC Graph. Siamese không quyết định support cạnh.
3. **Gradient có truyền ngược đúng để học Siamese Encoder không?** Có đường gradient khả vi qua cosine, gate, PCFS, sparse edge values, DGL weighted GraphSAGE và forecast loss. Top-K/support/index selection không khả vi; điều này chặn học support mới nhưng không chặn học edge weights trong support. Chưa có test backward thực tế chứng minh gradient hữu hạn.
4. **Thiết kế đủ cơ sở chạy E0/E1/E2 chưa?** Đủ cơ sở ở mức triển khai ablation dự báo có kiểm soát, với điều kiện E1 là Siamese-only, E2 hybrid và giữ support Pearson cố định. Chưa đủ để khẳng định chất lượng biểu diễn hay fairness từ kết quả vì chưa có benchmark matrix, tests và diagnostics.
5. **Cần sửa gì trước Giai đoạn 3?** Ưu tiên thêm unit tests cho graph/gradient, làm rõ cờ `sdei` dùng để chuẩn bị windows, thêm diagnostics embedding/graph, tạo config và benchmark paired nhiều horizon/seed; sau đó kiểm tra leakage/checkpoint và chạy pipeline smoke test trước benchmark chính thức.

## 10. File đã đối chiếu

- `FinalProject/Siamese_TVC/models/similarity.py`
- `FinalProject/Siamese_TVC/models/graph_generator.py`
- `FinalProject/Siamese_TVC/models/tvcformer_siamese.py`
- `FinalProject/Siamese_TVC/exp_siamese.py`
- `FinalProject/Siamese_TVC/run_siamese.py`
- `FinalProject/Siamese_TVC/configs/ETTh1_pl96.yaml`
- `official/layers/TVCFormer_backbone.py`
- `official/layers/GraphGenerator.py`
- `official/layers/TVC_Encoder.py`
- `official/layers/Graph_layers.py`
- `official/utils/correlation_utils.py`
- `official/utils/normal_utils.py`
- `official/utils/patch_utils.py`
- `official/utils/graph_utils.py`
- `official/data_provider/data_loader.py`
- `official/exp/exp_main.py`
