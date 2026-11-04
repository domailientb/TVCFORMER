# Ghi chú (VI): Tính các chỉ số sai số dự báo như MSE, MAE, RMSE, MAPE và các biến thể liên
#    quan.
# Thành phần chính trong file: RSE, CORR, MAE, MSE, RMSE, MAPE, ....
import numpy as np


# Tính sai số tương đối giữa dự báo và giá trị thực tế.
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


# Tính hệ số tương quan giữa giá trị dự báo và giá trị thực tế.
def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


# Tính sai số tuyệt đối trung bình.
def MAE(pred, true):
    return np.mean(np.abs(pred - true))


# Tính sai số bình phương trung bình.
def MSE(pred, true):
    return np.mean((pred - true) ** 2)


# Tính căn bậc hai của sai số bình phương trung bình.
def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


# Tính sai số phần trăm tuyệt đối trung bình.
def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


# Tính sai số phần trăm bình phương trung bình.
def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


# Tổng hợp và trả về toàn bộ các chỉ số đánh giá dự báo.
def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr