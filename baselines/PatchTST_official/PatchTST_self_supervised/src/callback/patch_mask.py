# Chuyển dữ liệu chuỗi thời gian thành các patch (đoạn con) và tạo mặt nạ che ngẫu nhiên các patch phục vụ tác vụ tái tạo dữ liệu trong tiền huấn luyện.

# Thành phần chính: PatchCB, PatchMaskCB, create_patch, Patch, random_masking, random_masking_3D.

import torch
from torch import nn

from .core import Callback


# Callback chuyển dữ liệu đầu vào thành các patch trước khi đưa vào mô hình.
class PatchCB(Callback):

    def __init__(self, patch_len, stride ):
        self.patch_len = patch_len
        self.stride = stride

    # Thực hiện chia chuỗi thành các patch trước bước lan truyền thuận.
    def before_forward(self): self.set_patch()

    def set_patch(self):
        # Chuyển dữ liệu từ [bs, seq_len, n_vars] thành [bs, num_patch, n_vars, patch_len].
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)
        self.learner.xb = xb_patch


# Callback tạo dữ liệu bị che để thực hiện tác vụ tái tạo patch trong tiền huấn luyện.
class PatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio,
                        mask_when_pred:bool=False):
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio        

    def before_fit(self):
        # Thay hàm loss mặc định bằng hàm loss chỉ tính trên các patch bị che.
        self.learner.loss_func = self._loss        
        device = self.learner.device       

    def before_forward(self): self.patch_masking()

    def patch_masking(self):
        # Chia chuỗi thành patch và giữ lại bản gốc làm mục tiêu tái tạo.
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)

        # Che ngẫu nhiên một tỷ lệ patch, đầu vào mô hình nhận dữ liệu đã che.
        xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)
        self.mask = self.mask.bool()

        self.learner.xb = xb_mask       # Dữ liệu đầu vào đã che.
        self.learner.yb = xb_patch      # Dữ liệu gốc làm mục tiêu tái tạo.

    # Tính MSE trên các patch bị che, không tính trên patch được giữ lại.
    def _loss(self, preds, target):        
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.mask).sum() / self.mask.sum()
        return loss


# Hàm chia chuỗi thời gian thành các patch có độ dài patch_len và khoảng cách stride.
def create_patch(xb, patch_len, stride):
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len  + stride*(num_patch-1)

    # Căn các patch về cuối chuỗi; nếu độ dài không chia hết thì bỏ phần dư ở đầu.
    s_begin = seq_len - tgt_len

    xb = xb[:, s_begin:, :]
    # Tạo các cửa sổ trượt trên chiều thời gian.
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)
    return xb, num_patch


# Lớp thực hiện chia chuỗi thành patch trực tiếp trong kiến trúc mạng.
class Patch(nn.Module):
    def __init__(self,seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
        tgt_len = patch_len  + stride*(self.num_patch-1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        # Cắt phần dư đầu chuỗi và chia thành các patch bằng cửa sổ trượt.
        x = x[:, self.s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return x


# Che ngẫu nhiên patch riêng biệt cho từng mẫu và từng biến chuỗi thời gian.
def random_masking(xb, mask_ratio):
    # xb: [bs, num_patch, n_vars, patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    # Xác định số patch được giữ lại theo tỷ lệ mask.
    len_keep = int(L * (1 - mask_ratio))

    # Sinh giá trị ngẫu nhiên để quyết định patch nào được giữ và patch nào bị che.
    noise = torch.rand(bs, L, nvars, device=xb.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Lấy các patch được giữ lại.
    ids_keep = ids_shuffle[:, :len_keep, :]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))

    # Thay các patch bị che bằng vector 0.
    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)

    # Khôi phục thứ tự patch ban đầu sau khi sắp xếp.
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D))

    # Tạo mặt nạ nhị phân: 0 là patch được giữ, 1 là patch bị che.
    mask = torch.ones([bs, L, nvars], device=x.device)
    mask[:, :len_keep, :] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, x_kept, mask, ids_restore


# Phiên bản masking cho dữ liệu 3 chiều [bs, num_patch, dim].
def random_masking_3D(xb, mask_ratio):
    bs, L, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, device=xb.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    x_removed = torch.zeros(bs, L-len_keep, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)

    # Đưa các patch về đúng vị trí ban đầu và tạo dữ liệu đã che.
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D))

    # Đánh dấu patch bị che bằng 1, patch được giữ bằng 0.
    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, x_kept, mask, ids_restore