# Ghi chú (VI): Mã nguồn trong pipeline Transformer chứa các thành phần kiến trúc và phép biến đổi
#    tensor.
# Thành phần chính trong file: ConvLayer, EncoderLayer, Encoder, DecoderLayer, Decoder.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ConvLayer thực hiện tích chập, chuẩn hóa, kích hoạt và giảm độ dài chuỗi.
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # Chuyển tensor từ [B, L, D] sang [B, D, L] để thực hiện Conv1d.
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)

        # Max pooling giảm số lượng vị trí theo chiều thời gian.
        x = self.maxPool(x)

        # Đưa tensor trở lại dạng [B, L, D].
        x = x.transpose(1, 2)
        return x


# EncoderLayer gồm self-attention và mạng Feed Forward, kết hợp residual và LayerNorm.
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # Self-attention: Query, Key và Value đều lấy từ cùng tensor đầu vào.
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )

        # Kết nối residual và chuẩn hóa sau attention.
        x = x + self.dropout(new_x)

        # Feed Forward gồm hai lớp tích chập kernel_size=1 để biến đổi đặc trưng.
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # Kết hợp residual lần nữa và chuẩn hóa đầu ra của EncoderLayer.
        return self.norm2(x + y), attn


# Encoder xếp chồng nhiều EncoderLayer để trích xuất đặc trưng từ chuỗi đầu vào.
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []

        # Nếu có ConvLayer, thực hiện attention rồi giảm chiều chuỗi sau từng lớp.
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)

            # Thực hiện EncoderLayer cuối cùng mà không đi qua ConvLayer.
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            # Nếu không có ConvLayer, chỉ lần lượt thực hiện các lớp attention.
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        # Chuẩn hóa đầu ra cuối Encoder nếu được cấu hình.
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


# DecoderLayer kết hợp self-attention, cross-attention và Feed Forward.
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self-attention giúp Decoder xử lý mối quan hệ giữa các vị trí trong chính đầu vào của nó.
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        # Cross-attention giúp Decoder tiếp nhận thông tin đặc trưng từ đầu ra Encoder.
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        # Feed Forward biến đổi đặc trưng sau hai bước attention.
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # Residual và LayerNorm tạo đầu ra của DecoderLayer.
        return self.norm3(x + y)


# Decoder xếp chồng nhiều DecoderLayer và thực hiện chuẩn hóa, chiếu đầu ra nếu cần.
class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Lần lượt truyền dữ liệu qua các DecoderLayer.
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        # Chuẩn hóa đầu ra Decoder nếu được cấu hình.
        if self.norm is not None:
            x = self.norm(x)

        # Chiếu về số chiều đầu ra mong muốn nếu có lớp projection.
        if self.projection is not None:
            x = self.projection(x)
        return x