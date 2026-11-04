# Ghi chú (VI): Lớp RevIN chuẩn hóa theo từng chuỗi đầu vào và giải chuẩn hóa kết quả dự báo.
# Thành phần chính trong file: RevIN.
# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn


# Thực hiện chuẩn hóa và khôi phục dữ liệu theo thống kê riêng của từng chuỗi đầu vào.
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        # Khởi tạo hai tham số có thể học nếu bật affine.
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        # mode='norm': chuẩn hóa đầu vào; mode='denorm': khôi phục thang đo ban đầu.
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    # Khởi tạo trọng số và độ lệch riêng cho từng biến.
    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    # Tính mean, độ lệch chuẩn hoặc lấy giá trị cuối cùng làm mốc chuẩn hóa.
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        # detach để các thống kê này không tham gia cập nhật gradient.
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    # Trừ mean hoặc giá trị cuối, chia độ lệch chuẩn và áp dụng affine nếu được bật.
    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    # Đảo ngược các phép biến đổi để đưa dự báo về thang đo ban đầu.
    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x