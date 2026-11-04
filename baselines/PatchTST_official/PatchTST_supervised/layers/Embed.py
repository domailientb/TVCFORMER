# Ghi chú (VI): Nhúng giá trị chuỗi, vị trí và dấu thời gian thành biểu diễn đầu vào mô hình.
# Thành phần chính trong file: PositionalEmbedding, TokenEmbedding, FixedEmbedding,
#    TemporalEmbedding, TimeFeatureEmbedding, DataEmbedding, ....

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


# Tạo mã hóa vị trí Sinusoidal để biểu diễn thứ tự thời gian của các phần tử trong chuỗi.
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        # Tạo sẵn ma trận mã hóa vị trí, không cần học trong quá trình huấn luyện.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # Các chiều chẵn dùng sin, các chiều lẻ dùng cos.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Thêm chiều batch và đăng ký làm buffer để không cập nhật như tham số học.
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Lấy mã hóa vị trí có độ dài tương ứng với chuỗi đầu vào.
        return self.pe[:, :x.size(1)]


# Nhúng giá trị thực của chuỗi bằng tích chập một chiều.
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2

        # Conv1d trích xuất đặc trưng cục bộ trên chiều thời gian.
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)

        # Khởi tạo trọng số tích chập theo Kaiming.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # Chuyển [B, L, C] sang [B, C, L] để tích chập rồi đưa về [B, L, d_model].
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


# Tạo embedding cố định cho các giá trị thời gian rời rạc, không cập nhật trọng số.
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        # Tạo ma trận embedding bằng công thức Sinusoidal.
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        # Gán ma trận cố định cho lớp Embedding.
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        # Trả về embedding và ngắt gradient vì đây là embedding cố định.
        return self.emb(x).detach()


# Nhúng các thành phần thời gian rời rạc như tháng, ngày, thứ, giờ và phút.
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        # Số lượng giá trị có thể có của từng thành phần thời gian.
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        # Chọn embedding cố định hoặc embedding có thể học.
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        # Chuyển các chỉ số thời gian sang kiểu số nguyên để tra cứu embedding.
        x = x.long()

        # Lấy embedding tương ứng với từng thành phần thời gian.
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # Cộng các embedding thành một biểu diễn thời gian thống nhất.
        return hour_x + weekday_x + day_x + month_x + minute_x


# Nhúng các đặc trưng thời gian đã được biểu diễn dưới dạng số liên tục.
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # Xác định số chiều đặc trưng thời gian đầu vào theo tần suất dữ liệu.
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        # Chiếu các đặc trưng thời gian về không gian d_model.
        return self.embed(x)


# Kết hợp embedding giá trị, vị trí và thông tin thời gian thành đầu vào Transformer.
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # Chọn cách nhúng thời gian rời rạc hoặc đặc trưng thời gian liên tục.
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # Cộng embedding giá trị, thời gian và vị trí rồi áp dụng Dropout.
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


# DataEmbedding không sử dụng positional embedding, chỉ kết hợp giá trị và thông tin thời gian.
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # Chỉ cộng embedding giá trị và embedding thời gian.
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


# DataEmbedding chỉ sử dụng embedding giá trị, không cộng embedding vị trí và thời gian.
class DataEmbedding_wo_pos_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # Chỉ chuyển đổi giá trị chuỗi thành embedding.
        x = self.value_embedding(x)
        return self.dropout(x)


# DataEmbedding không sử dụng embedding thời gian, chỉ kết hợp giá trị và vị trí.
class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # Cộng embedding giá trị với embedding vị trí, bỏ qua thông tin thời gian.
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)