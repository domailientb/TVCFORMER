# Ghi chú (VI): Xây dựng mô hình PatchTST gồm hai phần chính:
#    - Backbone: Chia sẻ trọng số Transformer giữa các biến để mã hóa các patch độc lập theo từng kênh.
#    - Head: Chuyển biểu diễn đã mã hóa thành đầu ra phù hợp với từng tác vụ như dự báo, hồi quy,
#      phân loại hoặc tái tạo dữ liệu trong quá trình tiền huấn luyện.
# Thành phần chính trong file: PatchTST, PatchTSTEncoder, TSTEncoder, TSTEncoderLayer,
#    PredictionHead, RegressionHead, ClassificationHead, PretrainHead.

__all__ = ['PatchTST']

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from ..models.layers.pos_encoding import *
from ..models.layers.basics import *
from ..models.layers.attention import *


# Mô hình PatchTST hoàn chỉnh, kết hợp Backbone và Head.
# head_type quyết định loại tác vụ mà mô hình thực hiện.
class PatchTST(nn.Module):
    """
    Mã hóa các patch theo từng kênh, dùng chung trọng số Transformer giữa các kênh.
    head_type chọn đầu ra cho tiền huấn luyện, dự báo, hồi quy hoặc phân loại.
    Output dimension:
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """

    def __init__(self, c_in: int, target_dim: int, patch_len: int, stride: int, num_patch: int,
                 n_layers: int = 3, d_model=128, n_heads=16, shared_embedding=True, d_ff: int = 256,
                 norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "gelu",
                 res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, head_dropout=0,
                 head_type="prediction", individual=False,
                 y_range: Optional[tuple] = None, verbose: bool = False, **kwargs):

        super().__init__()

        # Kiểm tra loại Head được lựa chọn.
        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], \
            'head type should be either pretrain, prediction, or regression'

        # Backbone: Mã hóa các patch bằng phép chiếu tuyến tính và Transformer Encoder.
        self.backbone = PatchTSTEncoder(
            c_in, num_patch=num_patch, patch_len=patch_len,
            n_layers=n_layers, d_model=d_model, n_heads=n_heads,
            shared_embedding=shared_embedding, d_ff=d_ff,
            attn_dropout=attn_dropout, dropout=dropout, act=act,
            res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
            pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs
        )

        # Head: Lựa chọn lớp xử lý đầu ra tương ứng với tác vụ.
        self.n_vars = c_in
        self.head_type = head_type

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len, head_dropout)
        elif head_type == "prediction":
            self.head = PredictionHead(
                individual, self.n_vars, d_model, num_patch, target_dim, head_dropout
            )
        elif head_type == "regression":
            self.head = RegressionHead(
                self.n_vars, d_model, target_dim, head_dropout, y_range
            )
        elif head_type == "classification":
            self.head = ClassificationHead(
                self.n_vars, d_model, target_dim, head_dropout
            )

    # Luồng xử lý chính: dữ liệu đầu vào -> Backbone -> Head -> kết quả đầu ra.
    def forward(self, z):
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """

        # Backbone tạo biểu diễn đặc trưng cho từng biến và từng patch.
        z = self.backbone(z)  # [bs x nvars x d_model x num_patch]

        # Head chuyển biểu diễn đặc trưng thành kết quả của tác vụ.
        z = self.head(z)

        # Kích thước đầu ra phụ thuộc vào head_type đã chọn.
        return z


# Head hồi quy: tổng hợp đặc trưng của các biến tại patch cuối cùng,
# sau đó ánh xạ thành một hoặc nhiều giá trị đầu ra.
class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """

        # Chỉ lấy biểu diễn của patch cuối cùng.
        x = x[:, :, :, -1]  # [bs x nvars x d_model]

        # Làm phẳng đặc trưng của tất cả biến thành một vector.
        x = self.flatten(x)  # [bs x (nvars * d_model)]

        # Dropout giúp hạn chế overfitting trước khi dự đoán.
        x = self.dropout(x)

        # Ánh xạ vector đặc trưng thành giá trị hồi quy.
        y = self.linear(x)  # [bs x output_dim]

        # Nếu có giới hạn đầu ra, đưa kết quả về khoảng y_range.
        if self.y_range:
            y = SigmoidRange(*self.y_range)(y)

        return y


# Head phân loại: sử dụng đặc trưng tại patch cuối cùng để dự đoán nhãn.
class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """

        # Lấy biểu diễn của patch cuối cùng.
        x = x[:, :, :, -1]  # [bs x nvars x d_model]

        # Gộp đặc trưng của các biến thành một vector.
        x = self.flatten(x)  # [bs x (nvars * d_model)]

        # Áp dụng Dropout trước khi phân loại.
        x = self.dropout(x)

        # Tạo điểm số dự đoán cho các lớp.
        y = self.linear(x)  # [bs x n_classes]

        return y


# Head dự báo: chuyển biểu diễn của toàn bộ patch thành chuỗi giá trị tương lai.
# individual=True: mỗi biến có một Linear riêng.
# individual=False: tất cả biến dùng chung một Linear.
class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len,
                 head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten

        # Số chiều đặc trưng sau khi gộp d_model và num_patch.
        head_dim = d_model * num_patch

        if self.individual:
            # Tạo các lớp xử lý riêng cho từng biến.
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()

            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            # Tất cả biến sử dụng chung một lớp Linear dự báo.
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """

        if self.individual:
            # Xử lý từng biến độc lập bằng Linear tương ứng.
            x_out = []

            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # [bs x (d_model * num_patch)]
                z = self.linears[i](z)               # [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)

            # Ghép kết quả của các biến thành một tensor.
            x = torch.stack(x_out, dim=1)  # [bs x nvars x forecast_len]

        else:
            # Gộp đặc trưng của các patch cho từng biến.
            x = self.flatten(x)  # [bs x nvars x (d_model * num_patch)]

            # Áp dụng Dropout và dự báo bằng Linear dùng chung.
            x = self.dropout(x)
            x = self.linear(x)  # [bs x nvars x forecast_len]

        # Đổi trục để đưa chiều thời gian dự báo lên trước chiều biến.
        return x.transpose(2, 1)  # [bs x forecast_len x nvars]


# Head tiền huấn luyện: tái tạo dữ liệu của từng patch từ biểu diễn đã mã hóa.
class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x num_patch x nvars x patch_len]
        """

        # Đưa chiều num_patch lên trước chiều d_model.
        x = x.transpose(2, 3)  # [bs x nvars x num_patch x d_model]

        # Chiếu biểu diễn d_model về độ dài ban đầu của patch.
        x = self.linear(self.dropout(x))  # [bs x nvars x num_patch x patch_len]

        # Sắp xếp lại trục theo định dạng đầu ra yêu cầu.
        x = x.permute(0, 2, 1, 3)  # [bs x num_patch x nvars x patch_len]

        return x


# Backbone của PatchTST: biến đổi từng patch thành vector đặc trưng,
# bổ sung thông tin vị trí và đưa qua Transformer Encoder.
class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len,
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu",
                 store_attn=False, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding

        # Input encoding: chiếu mỗi patch từ patch_len chiều sang d_model chiều.
        if not shared_embedding:
            # Mỗi biến có một lớp Linear riêng để mã hóa patch.
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars):
                self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            # Tất cả biến dùng chung một lớp Linear để mã hóa patch.
            self.W_P = nn.Linear(patch_len, d_model)

        # Positional encoding: bổ sung thông tin thứ tự vị trí của các patch.
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Dropout trên biểu diễn sau khi cộng positional encoding.
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder: học quan hệ giữa các patch trong từng biến.
        self.encoder = TSTEncoder(
            d_model, n_heads, d_ff=d_ff, norm=norm,
            attn_dropout=attn_dropout, dropout=dropout,
            pre_norm=pre_norm, activation=act,
            res_attention=res_attention, n_layers=n_layers,
            store_attn=store_attn
        )

    def forward(self, x) -> Tensor:
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """

        bs, num_patch, n_vars, patch_len = x.shape

        # Input encoding: chuyển từng patch thành vector đặc trưng d_model chiều.
        if not self.shared_embedding:
            # Nếu không chia sẻ trọng số, mã hóa từng biến bằng Linear riêng.
            x_out = []
            for i in range(n_vars):
                z = self.W_P[i](x[:, :, i, :])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            # Nếu chia sẻ trọng số, sử dụng cùng một Linear cho mọi biến.
            x = self.W_P(x)  # [bs x num_patch x nvars x d_model]

        # Đưa chiều biến lên trước chiều patch.
        x = x.transpose(1, 2)  # [bs x nvars x num_patch x d_model]

        # Gộp batch và số biến để Transformer xử lý từng biến độc lập.
        u = torch.reshape(x, (bs * n_vars, num_patch, self.d_model))

        # Cộng positional encoding và áp dụng Dropout.
        u = self.dropout(u + self.W_pos)  # [bs*nvars x num_patch x d_model]

        # Đưa chuỗi patch qua Transformer Encoder để học quan hệ thời gian.
        z = self.encoder(u)  # [bs*nvars x num_patch x d_model]

        # Khôi phục chiều batch và số biến ban đầu.
        z = torch.reshape(z, (-1, n_vars, num_patch, self.d_model))

        # Đổi trục để đưa d_model đứng trước num_patch.
        z = z.permute(0, 1, 3, 2)  # [bs x nvars x d_model x num_patch]

        return z


# Transformer Encoder gồm nhiều Encoder Layer xếp chồng.
# Có thể truyền attention scores giữa các layer nếu bật res_attention.
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0.,
                 activation='gelu', res_attention=False, n_layers=1,
                 pre_norm=False, store_attn=False):
        super().__init__()

        # Khởi tạo danh sách các Encoder Layer.
        self.layers = nn.ModuleList([
            TSTEncoderLayer(
                d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                attn_dropout=attn_dropout, dropout=dropout,
                activation=activation, res_attention=res_attention,
                pre_norm=pre_norm, store_attn=store_attn
            )
            for i in range(n_layers)
        ])

        self.res_attention = res_attention

    def forward(self, src: Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """

        output = src
        scores = None

        if self.res_attention:
            # Truyền attention scores từ layer trước sang layer tiếp theo.
            for mod in self.layers:
                output, scores = mod(output, prev=scores)
            return output
        else:
            # Các layer xử lý tuần tự mà không truyền attention scores.
            for mod in self.layers:
                output = mod(output)
            return output


# Một Encoder Layer gồm Multi-Head Attention và Feed-Forward Network,
# kết hợp Residual Connection, Dropout và Normalization.
class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True,
                 activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()

        # Đảm bảo d_model có thể chia đều cho số attention heads.
        assert not d_model % n_heads, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        # Số chiều của mỗi head trong Multi-Head Attention.
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head Attention: học mối quan hệ giữa các patch.
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(
            d_model, n_heads, d_k, d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention
        )

        # Add & Norm sau nhánh Attention.
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Feed-Forward Network: xử lý đặc trưng riêng tại từng vị trí patch.
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias)
        )

        # Add & Norm sau nhánh Feed-Forward.
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None):
        """
        src: tensor [bs x q_len x d_model]
        """

        # Nhánh Multi-Head Attention.
        if self.pre_norm:
            src = self.norm_attn(src)

        # Tính Attention; có thể sử dụng scores từ layer trước nếu bật res_attention.
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)

        # Lưu attention weights nếu được yêu cầu.
        if self.store_attn:
            self.attn = attn

        # Add & Norm: cộng kết quả Attention với đầu vào ban đầu.
        src = src + self.dropout_attn(src2)

        if not self.pre_norm:
            src = self.norm_attn(src)

        # Nhánh Feed-Forward Network.
        if self.pre_norm:
            src = self.norm_ffn(src)

        src2 = self.ff(src)

        # Add & Norm: cộng kết quả Feed-Forward với đầu vào nhánh.
        src = src + self.dropout_ffn(src2)

        if not self.pre_norm:
            src = self.norm_ffn(src)

        # Trả về biểu diễn mới và attention scores nếu sử dụng residual attention.
        if self.res_attention:
            return src, scores
        else:
            return src