# Ghi chú (VI): Mã nguồn trong pipeline PatchTST định nghĩa kiến trúc mô hình và luồng forward
#    tạo dự báo.
# Thành phần chính trong file: Model.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Mô hình NLinear: chuẩn hóa chuỗi bằng cách trừ giá trị cuối cùng trước khi dự báo.
class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # Lấy độ dài chuỗi đầu vào và độ dài chuỗi cần dự báo từ cấu hình.
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Ánh xạ chuỗi đầu vào từ seq_len sang pred_len bằng một lớp Linear.
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

        # Có thể sử dụng dòng dưới để khởi tạo trọng số phục vụ trực quan hóa.
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x có dạng [Batch, Input length, Channel].

        # Lấy giá trị cuối cùng của mỗi biến trong chuỗi đầu vào làm mốc chuẩn hóa.
        seq_last = x[:,-1:,:].detach()

        # Trừ giá trị cuối cùng khỏi toàn bộ chuỗi để đưa dữ liệu về dạng tương đối so với mốc cuối.
        x = x - seq_last

        # Chuyển trục để Linear xử lý theo chiều thời gian, sau đó đưa tensor về thứ tự ban đầu.
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)

        # Cộng lại giá trị cuối cùng để khôi phục mức dữ liệu ban đầu cho kết quả dự báo.
        x = x + seq_last
        return x # [Batch, Output length, Channel]