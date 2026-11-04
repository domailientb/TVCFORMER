# Callback hỗ trợ quản lý quá trình huấn luyện, dự đoán và kiểm thử mô hình: chuyển dữ liệu lên thiết bị tính toán, thu thập kết quả dự đoán và nhãn thực tế.

# Khai báo các thành phần được phép sử dụng từ module
__all__ = ['Callback', 'SetupLearnerCB', 'GetPredictionsCB', 'GetTestCB' ]


""" 
# Các thời điểm Callback được kích hoạt trong quá trình huấn luyện và dự đoán:
# before_fit: trước khi huấn luyện
# before_epoch / after_epoch: trước và sau mỗi epoch
# before_batch_train / after_batch_train: trước và sau mỗi batch huấn luyện
# before_batch_valid / after_batch_valid: trước và sau mỗi batch đánh giá
# before_predict / after_predict: trước và sau quá trình dự đoán
"""

from ..basics import *
import torch

DTYPE = torch.float32

# Lớp Callback cơ sở, cung cấp khả năng truy cập các thuộc tính của Learner.
class Callback(GetAttr): 
    _default='learner'


# Thiết lập thiết bị tính toán và chuyển mô hình, dữ liệu lên CPU/GPU.
class SetupLearnerCB(Callback): 
    def __init__(self):        
        self.device = default_device(use_cuda=True)

    # Chuyển batch dữ liệu lên thiết bị tính toán trước mỗi bước xử lý.
    def before_batch_train(self): self._to_device()
    def before_batch_valid(self): self._to_device()
    def before_batch_predict(self): self._to_device()
    def before_batch_test(self): self._to_device()

    # Tách batch thành dữ liệu đầu vào (xb) và nhãn (yb).
    def _to_device(self):
        batch = to_device(self.batch, self.device)        
        if self.n_inp > 1: xb, yb = batch
        else: xb, yb = batch, None        
        self.learner.batch = xb, yb

    # Đưa mô hình lên GPU nếu có, đồng thời lưu thiết bị sử dụng cho Learner.
    def before_fit(self): 
        self.learner.model.to(self.device)
        self.learner.device = self.device                        


# Thu thập và ghép kết quả dự đoán theo từng batch.
class GetPredictionsCB(Callback):
    def __init__(self):
        super().__init__()

    # Khởi tạo danh sách lưu kết quả dự đoán.
    def before_predict(self):
        self.preds = []        

    # Lưu kết quả dự đoán sau mỗi batch.
    def after_batch_predict(self):        
        self.preds.append(self.pred)

    # Ghép kết quả dự đoán của tất cả các batch thành một tensor.
    def after_predict(self):           
        self.preds = torch.concat(self.preds)


# Thu thập kết quả dự đoán và nhãn thực tế trong quá trình kiểm thử.
class GetTestCB(Callback):
    def __init__(self):
        super().__init__()

    # Khởi tạo danh sách lưu dự đoán và giá trị thực tế.
    def before_test(self):
        self.preds, self.targets = [], []        

    # Lưu dự đoán và nhãn thực tế sau mỗi batch kiểm thử.
    def after_batch_test(self):        
        self.preds.append(self.pred)
        self.targets.append(self.yb)

    # Ghép kết quả của tất cả các batch thành tensor để phục vụ đánh giá.
    def after_test(self):           
        self.preds = torch.concat(self.preds)
        self.targets = torch.concat(self.targets)