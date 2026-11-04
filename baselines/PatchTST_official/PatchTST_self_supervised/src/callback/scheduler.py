# Quản lý và điều chỉnh Learning Rate trong quá trình huấn luyện, hỗ trợ OneCycleLR và tìm tốc độ học phù hợp bằng LR Finder.

# Thành phần chính: OneCycleLR, LRFinderCB, LinearLR, ExponentialLR, valley.

__all__ = ['OneCycleLR', 'LRFinderCB', 'LinearLR', 'ExponentialLR']

from cmath import inf
from ..basics import *
from .core import Callback
from torch.optim import lr_scheduler 
from torch.optim.lr_scheduler import _LRScheduler


# Điều chỉnh Learning Rate theo chu kỳ: tăng dần đến LR tối đa rồi giảm dần.
class OneCycleLR(Callback):
    def __init__(self, lr_max=None,
                        total_steps=None,                        
                        steps_per_epoch=None,
                        pct_start=0.3,
                        anneal_strategy='cos',
                        cycle_momentum=True,
                        base_momentum=0.85,
                        max_momentum=0.95,
                        div_factor=25.,
                        final_div_factor=1e4,
                        three_phase=False,
                        last_epoch=-1,
                        verbose=False):

        super().__init__()        
        self.lr_max = lr_max if lr_max else self.lr  
        self.total_steps, self.steps_per_epoch = total_steps, steps_per_epoch         
        self.pct_start = pct_start
        self.anneal_strategy, self.cycle_momentum = anneal_strategy, cycle_momentum
        self.base_momentum, self.max_momentum = base_momentum, max_momentum
        self.div_factor, self.final_div_factor = div_factor, final_div_factor
        self.three_phase = three_phase
        self.last_epoch = last_epoch
        self.verbose = verbose


    # Khởi tạo bộ lập lịch LR trước khi bắt đầu huấn luyện.
    def before_fit(self):
        if not self.steps_per_epoch: self.steps_per_epoch = len(self.dls.train)
        self.lrs = []  # Lưu lịch sử Learning Rate

        self.scheduler = lr_scheduler.OneCycleLR(optimizer = self.opt, 
                                            max_lr = self.lr_max,
                                            total_steps = self.total_steps,
                                            epochs = self.n_epochs,
                                            steps_per_epoch=self.steps_per_epoch,
                                            pct_start=self.pct_start,
                                            anneal_strategy=self.anneal_strategy,
                                            cycle_momentum=self.cycle_momentum,
                                            base_momentum=self.base_momentum,
                                            max_momentum=self.max_momentum,
                                            div_factor=self.div_factor,
                                            final_div_factor=self.final_div_factor,
                                            three_phase=self.three_phase,
                                            last_epoch=self.last_epoch,
                                            verbose=self.verbose
                                            )

    # Cập nhật LR sau mỗi batch huấn luyện và lưu lại giá trị.
    def after_batch_train(self):
        if self.model.training: 
            self.scheduler.step()
            self.lrs.append(self.scheduler.get_last_lr()[0])                  

    # Lưu lịch sử Learning Rate sau khi huấn luyện.
    def after_fit(self):        
        self.learner.scheduled_lrs = self.lrs



# Tìm Learning Rate phù hợp bằng cách thử nhiều giá trị LR và theo dõi Loss.
class LRFinderCB(Callback):
    def __init__(self, start_lr=1e-7, end_lr=10, num_iter=100, step_mode='exp', beta=0.98, suggestion='valley'):
        self.start_lr, self.end_lr = start_lr, end_lr
        self.num_iter = num_iter
        self.step_mode = step_mode                
        if beta >= 1: raise ValueError("`num_iter` must be smaller than 1")
        else: self.beta = beta
        self.suggestion = suggestion

    # Khởi tạo quá trình dò LR và lưu trạng thái mô hình ban đầu.
    def before_fit(self):        
        self.losses, self.lrs = [], []
        self.best_loss, self.aver_loss = inf, 0 
        self.train_iter = 0

        # Lưu trạng thái mô hình để khôi phục sau khi dò LR.
        self.temp_path = self.save('current', 'temp/', with_opt=False)  

        # Đặt Learning Rate ban đầu.
        self.set_lr(self.start_lr)

        if not self.num_iter: self.num_iter = len(self.dls.train)

        # Chọn cách tăng LR: tuyến tính hoặc theo hàm mũ.
        if self.step_mode.lower() == "exp":
            self.scheduler = ExponentialLR(self.opt, self.end_lr, self.num_iter)
        elif self.step_mode.lower() == "linear":
            self.scheduler = LinearLR(self.opt, self.end_lr, self.num_iter)

    # Cập nhật LR, ghi nhận Loss và dừng khi quá trình dò không còn hiệu quả.
    def after_batch_train(self):        
        self.train_iter += 1
        self.scheduler.step()
        self.lrs.append(self.scheduler.get_last_lr()[0])             

        # Làm mượt Loss để giảm dao động giữa các batch.
        self.smoothing(self.beta)
        if self.smoothed_loss < self.best_loss: self.best_loss = self.smoothed_loss

        # Dừng nếu Loss tăng đột biến hoặc đã đủ số lần thử.
        if self.smoothed_loss > 4 * self.best_loss: 
            raise KeyboardInterrupt
        if self.train_iter > self.num_iter: 
            raise KeyboardInterrupt

    # Làm mượt Loss bằng trung bình động có trọng số.
    def smoothing(self, beta):        
        self.aver_loss = beta * self.aver_loss + (1-beta) * self.loss.detach().item()
        self.smoothed_loss = self.aver_loss / (1 - beta**self.train_iter)
        self.losses.append(self.smoothed_loss)

    # Khôi phục mô hình sau khi dò LR và đề xuất LR phù hợp.
    def after_fit(self):        
        self.learner.opt.zero_grad()
        if self.suggestion == 'valley':
            self.suggested_lr = valley(self.lrs, self.losses)

        self.load(self.temp_path)     

    # Gán Learning Rate mới cho các nhóm tham số của Optimizer.
    def set_lr(self, lrs):
        if not isinstance(lrs, list): lrs = [lrs] * len(self.opt.param_groups)
        if len(lrs) != len(self.opt.param_groups):
            raise ValueError(
                "Length of `lrs` is not equal to the number of parameter groups "
                + "in the given optimizer")
        for param_group, lr in zip(self.opt.param_groups, lrs):
            param_group["lr"] = lr

    # Vẽ biểu đồ thể hiện mối quan hệ giữa Learning Rate và Loss.
    def plot_lr_find(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(self.lrs, self.losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale('log')
        plt.grid()


# Tăng Learning Rate tuyến tính từ giá trị ban đầu đến end_lr.
class LinearLR(_LRScheduler):

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        if num_iter <= 1: raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):        
        r = (self.last_epoch+1) / (self.num_iter - 1)
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


# Tăng Learning Rate theo hàm mũ từ giá trị ban đầu đến end_lr.
class ExponentialLR(_LRScheduler):

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.last_epoch = last_epoch
        if num_iter <= 1: raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):  
        r = (self.last_epoch+1) / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


# Tìm vùng Loss giảm dài nhất trên biểu đồ LR Finder và chọn LR đề xuất trong vùng đó.
def valley(lrs:list, losses:list):
    n = len(losses)
    max_start, max_end = 0,0

    # Tìm đoạn có Loss giảm liên tiếp dài nhất.
    lds = [1]*n
    for i in range(1,n):
        for j in range(0,i):
            if (losses[i] < losses[j]) and (lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1
            if lds[max_end] < lds[i]:
                max_end = i
                max_start = max_end - lds[max_end]

    # Chọn Learning Rate nằm trong vùng giảm Loss để đề xuất.
    sections = (max_end - max_start) / 3
    idx = max_start + int(sections) + int(sections/2)

    return float(lrs[idx])