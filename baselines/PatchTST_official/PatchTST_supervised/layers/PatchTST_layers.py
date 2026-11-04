__all__ = ['Transpose', 'get_activation_fn', 'moving_avg', 'series_decomp', 'PositionalEncoding', 'SinCosPosEncoding', 'Coord2dPosEncoding', 'Coord1dPosEncoding', 'positional_encoding']
# Ghi chú (VI): Chứa các thành phần nền như mã hóa vị trí, chuẩn hóa và phân rã chuỗi dùng trong
#    PatchTST.
# Thành phần chính trong file: Transpose, get_activation_fn, moving_avg, series_decomp,
#    PositionalEncoding, Coord2dPosEncoding, ....

import torch
from torch import nn
import math

# Hoán đổi hai chiều của tensor, có thể chuyển tensor về vùng nhớ liên tục.
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


# Chọn hàm kích hoạt ReLU hoặc GELU theo cấu hình mô hình.
def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


# decomposition

# Tính trung bình trượt để tách thành phần xu hướng (trend) của chuỗi thời gian.
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Lặp lại giá trị đầu và cuối để bổ sung biên, giúp phép trung bình trượt giữ nguyên độ dài chuỗi.
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # AvgPool1d tính trung bình trên cửa sổ trượt theo chiều thời gian.
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


# Phân rã chuỗi thành thành phần mùa vụ (residual) và thành phần xu hướng (trend).
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # Trend là kết quả trung bình trượt; residual là phần chênh lệch so với trend.
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# pos_encoding

# Tạo mã hóa vị trí Sinusoidal để biểu diễn thứ tự của các patch trong chuỗi.
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # Chuẩn hóa giá trị mã hóa vị trí nếu được bật.
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


# Tên thay thế cho PositionalEncoding.
SinCosPosEncoding = PositionalEncoding


# Tạo mã hóa vị trí hai chiều dựa trên tọa độ của vị trí và chiều embedding.
def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)

        # Điều chỉnh số mũ để đưa giá trị trung bình của mã hóa về gần 0.
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1

    # Chuẩn hóa mã hóa vị trí hai chiều.
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


# Tạo mã hóa vị trí một chiều theo dạng tuyến tính hoặc hàm mũ.
def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)

    # Chuẩn hóa giá trị mã hóa vị trí.
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


# Lựa chọn phương pháp mã hóa vị trí dựa trên tham số pe và thiết lập có học hay không.
def positional_encoding(pe, learn_pe, q_len, d_model):
    if pe == None:
        # Khởi tạo ngẫu nhiên nhưng cố định, dùng để kiểm tra ảnh hưởng của positional encoding.
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")

    # Chuyển mã hóa vị trí thành tham số của mô hình; learn_pe quyết định có cập nhật khi huấn luyện không.
    return nn.Parameter(W_pos, requires_grad=learn_pe)