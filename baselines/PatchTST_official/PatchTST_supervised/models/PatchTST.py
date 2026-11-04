# Ghi chú (VI): Định nghĩa PatchTST gồm encoder dùng chung theo kênh và các head dự báo, tiền
#    huấn luyện, hồi quy hoặc phân loại.
# Thành phần chính trong file: Model.
__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


# Khởi tạo mô hình PatchTST và thiết lập các thành phần kiến trúc từ cấu hình.
class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):

        super().__init__()

        # Lấy các tham số về số biến, độ dài chuỗi, số lớp encoder và kích thước mô hình.
        # Các tham số này quyết định cấu trúc và khả năng biểu diễn của PatchTST.
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        # Thiết lập cách chia chuỗi thành các patch.
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Lấy các tham số chuẩn hóa RevIN.
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        # Lấy cấu hình phân rã chuỗi thành xu hướng và phần dư.
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size


        # Khởi tạo kiến trúc mô hình.
        self.decomposition = decomposition
        if self.decomposition:
            # Nếu bật decomposition, tách chuỗi thành xu hướng và phần dư.
            # Hai thành phần được đưa qua hai backbone PatchTST riêng biệt.
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            # Nếu không bật decomposition, sử dụng một backbone PatchTST duy nhất.
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)


    # Luồng forward nhận chuỗi đầu vào và tạo ra chuỗi dự báo.
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            # Phân rã đầu vào thành phần dư và thành phần xu hướng.
            res_init, trend_init = self.decomp_module(x)

            # Chuyển tensor từ [Batch, Input length, Channel]
            # sang [Batch, Channel, Input length] để đưa vào backbone.
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]

            # Dự báo riêng phần dư và phần xu hướng bằng hai backbone.
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)

            # Cộng hai kết quả dự báo rồi chuyển về định dạng ban đầu.
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            # Nếu không phân rã, đưa trực tiếp chuỗi đầu vào qua backbone PatchTST.
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x