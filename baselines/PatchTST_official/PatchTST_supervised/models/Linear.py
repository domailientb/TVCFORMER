# Ghi chú (VI): Mã nguồn trong pipeline PatchTST định nghĩa kiến trúc mô hình và luồng forward
#    tạo dự báo.
# Thành phần chính trong file: Model.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Mô hình Linear: sử dụng một lớp Linear duy nhất để ánh xạ chuỗi đầu vào thành chuỗi dự báo.
class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # Lấy độ dài chuỗi đầu vào và độ dài chuỗi cần dự báo từ cấu hình.
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Ánh xạ trực tiếp từ seq_len bước đầu vào sang pred_len bước dự báo.
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

        # Có thể sử dụng dòng dưới để khởi tạo trọng số phục vụ trực quan hóa.
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x có dạng [Batch, Input length, Channel].
        # Chuyển thành [Batch, Channel, Input length] để lớp Linear xử lý theo chiều thời gian.
        # Sau đó chuyển lại thành [Batch, Output length, Channel].
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x # [Batch, Output length, Channel]