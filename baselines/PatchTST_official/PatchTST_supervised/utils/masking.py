# Ghi chú (VI): Tạo mặt nạ attention nhằm giới hạn những vị trí được phép tham gia phép tính
#    attention.
# Thành phần chính trong file: TriangularCausalMask, ProbMask.
import torch


# Tạo mặt nạ tam giác trên để ngăn mô hình attention đến các bước thời gian ở tương lai.
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


# Tạo mặt nạ cho các vị trí được chọn trong cơ chế ProbSparse Attention.
class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask