# Ghi chú (VI): Tập hợp các tiện ích hỗ trợ quá trình thực nghiệm,
# bao gồm điều chỉnh learning rate, dừng sớm, chuẩn hóa dữ liệu,
# trực quan hóa kết quả và thống kê độ phức tạp mô hình.
# Thành phần chính: adjust_learning_rate, EarlyStopping, dotdict,
# StandardScaler, visual, test_params_flop.

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# Sử dụng backend không cần giao diện đồ họa, phù hợp khi chạy trên server.
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # Điều chỉnh learning rate theo chiến lược được lựa chọn trong args.lradj.
    # Mỗi chiến lược quy định cách giảm hoặc giữ nguyên learning rate theo epoch.
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        # Giảm một nửa learning rate sau mỗi epoch.
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        # Giảm learning rate tại các mốc epoch được thiết lập sẵn.
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        # Giữ learning rate trong 3 epoch đầu, sau đó giảm dần theo hệ số 0.9.
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        # Giữ learning rate không đổi trong suốt quá trình huấn luyện.
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        # Giữ learning rate trong 10 epoch đầu, sau đó giảm còn 10%.
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        # Giữ learning rate trong 15 epoch đầu, sau đó giảm còn 10%.
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        # Giữ learning rate trong 25 epoch đầu, sau đó giảm còn 10%.
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        # Giữ learning rate trong 5 epoch đầu, sau đó giảm còn 10%.
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}
    elif args.lradj == 'TST':
        # Lấy learning rate hiện tại trực tiếp từ scheduler.
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    # Nếu epoch hiện tại nằm trong danh sách điều chỉnh,
    # cập nhật learning rate cho tất cả nhóm tham số của optimizer.
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    # Theo dõi validation loss và dừng huấn luyện khi mô hình
    # không còn cải thiện sau một số epoch liên tiếp.
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience          # Số epoch tối đa được phép không cải thiện.
        self.verbose = verbose            # Có in thông báo khi lưu checkpoint hay không.
        self.counter = 0                  # Đếm số epoch liên tiếp không cải thiện.
        self.best_score = None            # Điểm validation tốt nhất đã đạt được.
        self.early_stop = False           # Trạng thái dừng sớm.
        self.val_loss_min = np.Inf        # Validation loss nhỏ nhất.
        self.delta = delta                # Mức cải thiện tối thiểu được chấp nhận.

    def __call__(self, val_loss, model, path):
        # Đổi dấu validation loss để có thể so sánh theo hướng điểm càng lớn càng tốt.
        score = -val_loss

        if self.best_score is None:
            # Lần đánh giá đầu tiên: lưu mô hình làm checkpoint tốt nhất.
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score + self.delta:
            # Không cải thiện đủ mức delta: tăng bộ đếm.
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            # Dừng huấn luyện khi số lần không cải thiện đạt patience.
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            # Có cải thiện: cập nhật điểm tốt nhất, lưu checkpoint và đặt lại bộ đếm.
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        # Lưu trọng số của mô hình tại thời điểm validation loss tốt nhất.
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """Cho phép truy cập phần tử dictionary bằng dấu chấm, ví dụ args.lr."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    # Chuẩn hóa dữ liệu theo trung bình và độ lệch chuẩn.
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        # Đưa dữ liệu về thang đo chuẩn hóa: (x - mean) / std.
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        # Khôi phục dữ liệu về thang đo ban đầu.
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Trực quan hóa giá trị thực tế và giá trị dự báo trên cùng biểu đồ.
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)

    # Nếu có dự báo, vẽ thêm đường biểu diễn kết quả của mô hình.
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)

    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def test_params_flop(model,x_shape):
    """
    Thống kê số tham số và độ phức tạp tính toán của mô hình.
    Đầu vào x_shape là hình dạng dữ liệu dùng để kiểm tra model.
    """
    model_params = 0

    # Đếm tổng số tham số của mô hình.
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))

    # Sử dụng ptflops để ước lượng MACs và số tham số.
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))