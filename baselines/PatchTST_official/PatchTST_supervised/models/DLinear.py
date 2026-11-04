# Ghi chú (VI): Mã nguồn trong pipeline PatchTST định nghĩa kiến trúc mô hình và luồng forward
#    tạo dự báo.
# Thành phần chính trong file: moving_avg, series_decomp, Model.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Tính trung bình trượt để làm nổi bật thành phần xu hướng của chuỗi thời gian.
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Lặp lại giá trị đầu và cuối chuỗi để bổ sung phần đệm, giữ độ dài chuỗi sau khi lấy trung bình trượt.
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


# Phân rã chuỗi thời gian thành thành phần xu hướng và thành phần mùa vụ/phần dư.
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # moving_mean là xu hướng; res là phần còn lại sau khi loại bỏ xu hướng.
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# Mô hình DLinear: phân rã chuỗi đầu vào rồi dự báo riêng hai thành phần.
class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Kích thước cửa sổ dùng để tính trung bình trượt khi phân rã chuỗi.
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        # Nếu individual=True, mỗi biến có một cặp Linear riêng để dự báo.
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Có thể sử dụng hai dòng dưới để khởi tạo trọng số phục vụ trực quan hóa.
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            # Nếu individual=False, các biến dùng chung hai lớp Linear.
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

            # Có thể sử dụng hai dòng dưới để khởi tạo trọng số phục vụ trực quan hóa.
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x có dạng [Batch, Input length, Channel].
        # Phân rã đầu vào thành thành phần mùa vụ/phần dư và thành phần xu hướng.
        seasonal_init, trend_init = self.decompsition(x)

        # Chuyển trục thành [Batch, Channel, Input length] để Linear xử lý theo chiều thời gian.
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        if self.individual:
            # Khởi tạo tensor chứa kết quả dự báo của từng biến.
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)

            # Dự báo riêng thành phần mùa vụ và xu hướng cho từng biến.
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            # Dự báo hai thành phần bằng các lớp Linear dùng chung giữa các biến.
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # Cộng hai kết quả dự báo để thu được dự báo cuối cùng.
        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]