# -----------------------------------------------------------------------------
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_statistics_block(x, subtract_last=False, eps=1e-5):
    dim2reduce = -1
    if subtract_last:
        out_mean = x[:,:,:,-1].unsqueeze(-1)
    else:
        out_mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
    out_stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + eps).detach()
    return out_mean, out_stdev

def get_statistics_full(x_in,
                        padding_len,
                        padding_mode,
                        block_num,
                        subtract_last=False,
                        eps=1e-5,
                        ):
    z = x_in.permute(0,2,1)
    if padding_mode == 'start':
        z = F.pad(z, (padding_len, 0), mode='replicate')
    elif padding_mode == 'end':
        z = F.pad(z, (0, padding_len), mode='replicate')
    else:
        raise ValueError("Invalid padding mode")
    
    z_block_list = torch.chunk(z, block_num, dim=-1)
    z_block = torch.stack(z_block_list, dim=-2)
    z_mean, z_stdev = get_statistics_block(z_block, subtract_last=subtract_last, eps=eps)
    x_out = torch.cat((z_mean, z_stdev), dim=-1)
    
    return x_out
