# Ghi chú (VI): Mã nguồn trong pipeline PatchTST định nghĩa kiến trúc mô hình và luồng forward
#    tạo dự báo.
# Thành phần chính trong file: Naive_repeat, Naive_thread, _arima, Arima, _sarima, SArima, ....
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pmdarima as pm
import threading
from sklearn.ensemble import GradientBoostingRegressor

# Mô hình Naive: lặp lại giá trị cuối cùng của chuỗi đầu vào cho toàn bộ khoảng dự báo.
class Naive_repeat(nn.Module):
    def __init__(self, configs):
        super(Naive_repeat, self).__init__()
        self.pred_len = configs.pred_len

    def forward(self, x):
        # Lấy kích thước batch, chiều dài chuỗi đầu vào và số biến.
        B,L,D = x.shape

        # Lấy giá trị cuối cùng của mỗi biến rồi lặp lại pred_len lần.
        x = x[:,-1,:].reshape(B,1,D).repeat(self.pred_len,axis=1)
        return x # [B, L, D]


# Tạo luồng riêng để thực hiện các tác vụ dự báo độc lập song song.
class Naive_thread(threading.Thread):
    def __init__(self,func,args=()):
        super(Naive_thread,self).__init__()
        self.func = func
        self.args = args

    # Thực thi hàm được truyền vào trong luồng.
    def run(self):
        self.results = self.func(*self.args)

    # Chờ luồng hoàn thành và trả về kết quả dự báo.
    def return_result(self):
        threading.Thread.join(self)
        return self.results


# Hàm thực hiện dự báo ARIMA cho một chuỗi thời gian đơn.
def _arima(seq,pred_len,bt,i):
    # Tự động lựa chọn cấu hình ARIMA phù hợp với chuỗi đầu vào.
    model = pm.auto_arima(seq)

    # Dự báo pred_len bước tiếp theo.
    forecasts = model.predict(pred_len) 
    return forecasts,bt,i


# Mô hình ARIMA: dự báo riêng từng biến trong từng mẫu dữ liệu.
class Arima(nn.Module):
    """
    Extremely slow, please sample < 0.1
    """
    def __init__(self, configs):
        super(Arima, self).__init__()
        self.pred_len = configs.pred_len

    def forward(self, x):
        # Khởi tạo mảng chứa kết quả dự báo với dạng [Batch, pred_len, Channel].
        result = np.zeros([x.shape[0],self.pred_len,x.shape[2]])
        threads = []

        # Duyệt từng mẫu trong batch và từng biến trong mẫu.
        for bt,seqs in tqdm(enumerate(x)):
            for i in range(seqs.shape[-1]):
                seq = seqs[:,i]

                # Tạo và khởi chạy một luồng ARIMA cho từng chuỗi đơn.
                one_seq = Naive_thread(func=_arima,args=(seq,self.pred_len,bt,i))
                threads.append(one_seq)
                threads[-1].start()

        # Thu thập kết quả từ các luồng và lưu đúng vị trí mẫu, biến.
        for every_thread in tqdm(threads):
            forcast,bt,i = every_thread.return_result()
            result[bt,:,i] = forcast

        return result # [B, L, D]


# Hàm thực hiện dự báo SARIMA cho một chuỗi thời gian đơn.
def _sarima(season,seq,pred_len,bt,i):
    # Tự động lựa chọn cấu hình SARIMA có xét đến tính mùa vụ.
    model = pm.auto_arima(seq, seasonal=True, m=season)
    forecasts = model.predict(pred_len) 
    return forecasts,bt,i


# Mô hình SARIMA: mở rộng ARIMA bằng cách xét thêm thành phần mùa vụ.
class SArima(nn.Module):
    """
    Extremely extremely slow, please sample < 0.01
    """
    def __init__(self, configs):
        super(SArima, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        # Thiết lập chu kỳ mùa vụ mặc định và điều chỉnh theo bộ dữ liệu.
        self.season = 24
        if 'Ettm' in configs.data_path:
            self.season = 12
        elif 'ILI' in configs.data_path:
            self.season = 1
        if self.season >= self.seq_len:
            self.season = 1

    def forward(self, x):
        # Khởi tạo mảng lưu kết quả dự báo cho từng mẫu và từng biến.
        result = np.zeros([x.shape[0],self.pred_len,x.shape[2]])
        threads = []

        # Tạo một luồng SARIMA riêng cho mỗi chuỗi đơn trong batch.
        for bt,seqs in tqdm(enumerate(x)):
            for i in range(seqs.shape[-1]):
                seq = seqs[:,i]
                one_seq = Naive_thread(func=_sarima,args=(self.season,seq,self.pred_len,bt,i))
                threads.append(one_seq)
                threads[-1].start()

        # Thu thập kết quả dự báo từ các luồng và ghép vào mảng kết quả.
        for every_thread in tqdm(threads):
            forcast,bt,i = every_thread.return_result()
            result[bt,:,i] = forcast
        return result # [B, L, D]


# Hàm huấn luyện Gradient Boosting Regressor trên một chuỗi và dự báo các bước tiếp theo.
def _gbrt(seq,seq_len,pred_len,bt,i):
    model = GradientBoostingRegressor()

    # Sử dụng chỉ số thời gian làm đầu vào và giá trị chuỗi làm mục tiêu huấn luyện.
    model.fit(np.arange(seq_len).reshape(-1,1),seq.reshape(-1,1))

    # Dự báo tại các chỉ số thời gian tiếp nối sau chuỗi đầu vào.
    forecasts = model.predict(np.arange(seq_len,seq_len+pred_len).reshape(-1,1))  
    return forecasts,bt,i


# Mô hình GBRT: áp dụng Gradient Boosting Regressor riêng cho từng biến trong từng mẫu.
class GBRT(nn.Module):
    def __init__(self, configs):
        super(GBRT, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forward(self, x):
        # Khởi tạo mảng kết quả có kích thước [Batch, pred_len, Channel].
        result = np.zeros([x.shape[0],self.pred_len,x.shape[2]])
        threads = []

        # Duyệt qua từng mẫu và từng biến để tạo mô hình dự báo riêng.
        for bt,seqs in tqdm(enumerate(x)):
            for i in range(seqs.shape[-1]):
                seq = seqs[:,i]

                # Tạo và khởi chạy luồng thực hiện dự báo GBRT.
                one_seq = Naive_thread(func=_gbrt,args=(seq,self.seq_len,self.pred_len,bt,i))
                threads.append(one_seq)
                threads[-1].start()

        # Tổng hợp dự báo từ tất cả các luồng.
        for every_thread in tqdm(threads):
            forcast,bt,i = every_thread.return_result()
            result[bt,:,i] = forcast
        return result # [B, L, D]