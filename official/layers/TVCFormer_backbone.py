# -----------------------------------------------------------------------------
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License
# Description:
# Refer to PatchTST for code structure.
# TVCFormer pipeline.
# -----------------------------------------------------------------------------

__all__ = ['TVCFormer_backbone']

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.RevINs import RevIN
from layers.GraphGenerator import graph_generator
from layers.Decoders import Flatten_Head
from layers.TVC_Encoder import TSTiEncoder
from utils.normal_utils import get_statistics_full

from utils.patch_utils import check_patch_divide, cut_detail_patches
from utils.correlation_utils import get_pearson_correlation_vector, top_k_channel_pairs_choose

class TVCFormer_backbone(nn.Module):
    def __init__(self, c_in:int, device: str,  subgraph_keep_rate: float, 
                    graph_keep_self_cycles: bool, eps: float,
                    graph_conv_hidden_size:int, graph_conv_kernel_size:float,
                    time_node_num:int, gnn_r_degree:int, gnn_hidden_size:int,
                    context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                    n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                    d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                    padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                    pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                    pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                    verbose:bool=False, **kwargs):
        super(TVCFormer_backbone, self).__init__()
        
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        self.device = device
        self.subgraph_keep_rate = subgraph_keep_rate
        self.graph_keep_self_cycles = graph_keep_self_cycles
        
        self.statistics_num = 2
        self.subtract_last = subtract_last
        self.eps = eps
        self.c_in = c_in
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.time_node_num = time_node_num
        
        patch_num, patch_pre_node, x_padding_len = check_patch_divide(seq_len=context_window, time_node_num=time_node_num, patch_len=patch_len, stride=stride)
        
        self.time_node_len = patch_len + (patch_pre_node-1) * stride
        self.x_padding_len = x_padding_len
        
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, x_padding_len))
        elif padding_patch == 'start':
            self.padding_patch_layer = nn.ReplicationPad1d((x_padding_len, 0))
        else:
            raise ValueError('Invalid padding patch type')
            
        self.backbone = TSTiEncoder(c_in=c_in, patch_num=patch_num, patch_len=patch_len,
                                    time_node_num=time_node_num, patch_pre_node=patch_pre_node, gnn_r_degree=gnn_r_degree, gnn_hidden_size=gnn_hidden_size,
                                    max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        
        self.statistic_norm = nn.LayerNorm(normalized_shape=[c_in, time_node_num], elementwise_affine=affine)
        
        self.graph_generator = graph_generator(timenode_num=time_node_num, channel_num=c_in,
                                                embeded_dim=self.statistics_num,
                                                conv_hidden_size=graph_conv_hidden_size, conv_kernel_size=graph_conv_kernel_size)
        
        self.detail_patch_total_len = patch_pre_node * time_node_num * d_model
        
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        
        if self.pretrain_head: 
            self.detail_decoder = self.create_pretrain_head(self.detail_patch_total_len, c_in, fc_dropout)
        elif head_type == 'flatten': 
            self.detail_decoder = Flatten_Head(self.individual, self.c_in, self.detail_patch_total_len, target_window, head_dropout=head_dropout)
        
    def forward(self, z:Tensor):  # z:[batch_size, channel_num, seq_len]
        ch_relevent = get_pearson_correlation_vector(z) # batch_size × (channel_num * channel_num)
        ch_index_vector = top_k_channel_pairs_choose(c_vector=ch_relevent, ch_num=self.c_in, keep_rate=self.subgraph_keep_rate, 
                                                    keep_self_cycles=self.graph_keep_self_cycles, device=self.device)
        
        
        z_s = z.permute(0,2,1)  # z_s: [batch_size, seq_len, channel_num]
        z_embeded = get_statistics_full(x_in=z_s, padding_len=self.x_padding_len, padding_mode=self.padding_patch, 
                                        block_num=self.time_node_num, subtract_last=self.subtract_last, eps=self.eps)  # batch_size × channel_num × time_node_num × 2
        
        z_embeded = z_embeded.permute(0,3,1,2)  # batch_size × 2 × channel_num × time_node_num
        z_embeded = self.statistic_norm(z_embeded)
        z_embeded = z_embeded.permute(0,2,3,1)  # batch_size × channel_num × time_node_num × 2
        
        z = self.padding_patch_layer(z)  # padding
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        z_graph = z_embeded
        s_A_i, s_A_v, adj_full = self.graph_generator(z_graph, ch_index_vector)
        
        z = cut_detail_patches(z, self.time_node_num, self.patch_len, self.stride)  # batch_size × channel_num × total_patch_num x patch_len
        z_l, z_pred = self.backbone(z, s_A_i, s_A_v)
        
        z_pred = z_pred.permute(0,1,3,2)  # batch_size × channel_num x d_model × total_patch_num
        z_pred = self.detail_decoder(z_pred)
        
        if self.revin:
            z_pred = z_pred.permute(0,2,1)
            z_pred = self.revin_layer(z_pred, 'denorm')
            z_pred = z_pred.permute(0,2,1)
    
        return z_pred  # batch_size × channel_num x target_window
        
    def create_pretrain_head(self, head_nf, vars, dropout):
        
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )
