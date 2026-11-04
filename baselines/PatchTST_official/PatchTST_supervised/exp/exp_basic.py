# Ghi chú: Lớp cơ sở dùng chung cho các thí nghiệm, chịu trách nhiệm khởi tạo thiết bị và mô hình.
# Các lớp thí nghiệm cụ thể sẽ kế thừa Exp_Basic và triển khai những phương thức còn thiếu.
import os
import torch
import numpy as np


class Exp_Basic(object):
    # Khởi tạo thí nghiệm: lưu cấu hình, xác định thiết bị và xây dựng mô hình.
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    # Phương thức xây dựng mô hình, cần được lớp thí nghiệm kế thừa triển khai.
    def _build_model(self):
        raise NotImplementedError
        return None

    # Xác định thiết bị thực thi dựa trên cấu hình GPU hoặc CPU.
    def _acquire_device(self):
        if self.args.use_gpu:
            # Thiết lập GPU được phép sử dụng; hỗ trợ cả chế độ một GPU và nhiều GPU.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    # Lấy dữ liệu cho thí nghiệm; lớp kế thừa sẽ triển khai quy trình tải dữ liệu.
    def _get_data(self):
        pass

    # Đánh giá mô hình trên tập validation trong quá trình huấn luyện.
    def vali(self):
        pass

    # Triển khai quy trình huấn luyện mô hình.
    def train(self):
        pass

    # Triển khai quy trình kiểm tra và đánh giá mô hình.
    def test(self):
        pass