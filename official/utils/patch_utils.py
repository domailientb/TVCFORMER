# -----------------------------------------------------------------------------
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License
# -----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
import math

def check_patch_divide(seq_len, time_node_num, patch_len, stride):
    k = math.ceil(((seq_len/time_node_num)-patch_len)/stride)
    if not k>=0:
        raise ValueError('patch_len too large')
    padding_len = int(time_node_num * (patch_len + (k * stride)) - seq_len)
    if padding_len >= int(patch_len + (k * stride)):
        raise ValueError('stride too large')
    patch_pre_node = k+1
    total_patch_num = int(patch_pre_node * time_node_num)
    
    return total_patch_num, patch_pre_node, padding_len

def cut_detail_patches(x_in, time_node_num, patch_len, stride):
    segment_in = torch.chunk(x_in, time_node_num, dim=-1)
    patch_list = []
    for segment_cell in segment_in:
        patch_list.append(segment_cell.unfold(dimension=-1, size=patch_len, step=stride))
    x_out = torch.cat(patch_list, dim=-2)
    
    return x_out
