# Ghi chú (VI): Callback chuẩn hóa đầu vào và khôi phục kết quả dự báo
# bằng RevIN trong quá trình mô hình thực hiện forward.
# Thành phần chính: RevInCB.

import torch
import torch.nn as nn
from .core import Callback
from src.models.layers.revin import RevIN


class RevInCB(Callback):
    def __init__(self, num_features: int, eps=1e-5, 
                        affine:bool=False, denorm:bool=True):
        """
        :param num_features: Số lượng biến đầu vào.
        :param eps: Giá trị nhỏ để đảm bảo tính ổn định khi chuẩn hóa.
        :param affine: Có sử dụng tham số scale và bias có thể học hay không.
        :param denorm: Có khôi phục dữ liệu dự báo về thang đo ban đầu hay không.

        Callback này chỉ hoạt động đúng với affine=False
        theo cách triển khai hiện tại.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.denorm = denorm

        # Khởi tạo lớp RevIN để chuẩn hóa và khôi phục dữ liệu.
        self.revin = RevIN(num_features, eps, affine)


    # Chuẩn hóa dữ liệu đầu vào trước khi mô hình thực hiện forward.
    def before_forward(self):
        self.revin_norm()

    # Khôi phục kết quả dự báo sau khi mô hình thực hiện forward.
    def after_forward(self): 
        if self.denorm:
            self.revin_denorm() 

    def revin_norm(self):
        # Chuẩn hóa đầu vào theo thống kê của từng mẫu dữ liệu.
        xb_revin = self.revin(self.xb, 'norm')  # [batch, seq_len, nvars]
        self.learner.xb = xb_revin

    def revin_denorm(self):
        # Đưa kết quả dự báo về thang đo ban đầu.
        pred = self.revin(self.pred, 'denorm')  # [batch, target_window, nvars]
        self.learner.pred = pred