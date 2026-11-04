# Ghi chú (VI): Mã nguồn trong pipeline Autoformer chứa các thành phần kiến trúc và phép biến đổi
#    tensor.
# Thành phần chính trong file: my_Layernorm, moving_avg, series_decomp, EncoderLayer, Encoder,
#    DecoderLayer, ....

import torch
import torch.nn as nn
import torch.nn.functional as F


# LayerNorm được thiết kế riêng cho thành phần mùa vụ, loại bỏ độ lệch trung bình theo chiều thời gian.
class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        # Chuẩn hóa đặc trưng theo LayerNorm.
        x_hat = self.layernorm(x)

        # Tính và loại bỏ giá trị trung bình theo chiều thời gian.
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


# Tính trung bình trượt để làm nổi bật thành phần xu hướng của chuỗi.
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Bổ sung biên bằng cách lặp lại giá trị đầu và cuối chuỗi.
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # Tính trung bình trượt theo chiều thời gian.
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


# Phân rã chuỗi thành thành phần mùa vụ (residual) và xu hướng (trend).
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # Thành phần xu hướng được tính bằng trung bình trượt.
        moving_mean = self.moving_avg(x)

        # Thành phần mùa vụ là phần chênh lệch giữa chuỗi gốc và xu hướng.
        res = x - moving_mean
        return res, moving_mean


# Một lớp Encoder của Autoformer, kết hợp AutoCorrelation và phân rã chuỗi từng bước.
class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # Tìm mối tương quan theo thời gian và cộng phần thông tin thu được vào đầu vào.
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        # Phân rã lần thứ nhất để tách thành phần xu hướng khỏi đặc trưng sau attention.
        x, _ = self.decomp1(x)

        # Feed Forward biến đổi đặc trưng mùa vụ.
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # Phân rã lần thứ hai sau Feed Forward và trả về thành phần mùa vụ.
        res, _ = self.decomp2(x + y)
        return res, attn


# Encoder gồm nhiều EncoderLayer, có thể kết hợp thêm ConvLayer và lớp chuẩn hóa cuối.
class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []

        # Nếu có ConvLayer, lần lượt thực hiện attention và tích chập giảm chiều.
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)

            # Xử lý bằng EncoderLayer cuối cùng.
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            # Nếu không có ConvLayer, chỉ thực hiện tuần tự các lớp attention.
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        # Chuẩn hóa đầu ra Encoder nếu được cấu hình.
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


# DecoderLayer kết hợp self-attention, cross-attention và phân rã chuỗi nhiều lần.
class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)

        # Chiếu thành phần xu hướng về số biến đầu ra, sử dụng tích chập với circular padding.
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self-attention khai thác quan hệ giữa các vị trí trong đầu vào Decoder.
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])

        # Phân rã lần thứ nhất và lưu lại thành phần xu hướng.
        x, trend1 = self.decomp1(x)

        # Cross-attention tiếp nhận thông tin từ đầu ra Encoder.
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        # Phân rã lần thứ hai và lưu thành phần xu hướng tiếp theo.
        x, trend2 = self.decomp2(x)

        # Feed Forward biến đổi đặc trưng của Decoder.
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # Phân rã lần thứ ba sau Feed Forward.
        x, trend3 = self.decomp3(x + y)

        # Cộng các thành phần xu hướng thu được từ ba lần phân rã.
        residual_trend = trend1 + trend2 + trend3

        # Chiếu xu hướng về số biến đầu ra để tích lũy vào kết quả dự báo.
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


# Decoder xếp chồng nhiều DecoderLayer và đồng thời tích lũy thành phần xu hướng.
class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        # Truyền dữ liệu qua từng DecoderLayer và cộng dồn phần xu hướng.
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        # Chuẩn hóa và chiếu đặc trưng mùa vụ nếu được cấu hình.
        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend