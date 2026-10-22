# -----------------------------------------------------------------------------
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License
# Description:
# Refer to PatchTST for code structure.
# Implementation of TVCFormer’s Temporal Encoder.
# -----------------------------------------------------------------------------

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_layers import positional_encoding, Transpose, get_activation_fn
from utils.segment_utils import generate_local_tokens, embedding_local_tokens, split_local_tokens_from_total_tokens
from layers.Graph_layers import dgl_graph_for_trans
from utils.graph_utils import genearte_graph_from_coo

class TSTiEncoder(nn.Module):
    def __init__(self, c_in, patch_num, patch_len,
                    time_node_num, patch_pre_node, gnn_r_degree, gnn_hidden_size,
                    max_seq_len=1024, n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                    d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                    key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                    pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.time_node_num = time_node_num
        
        q_len = patch_num + time_node_num
        
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len
        
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = TSTEncoder(q_len, d_model, n_heads, c_in=c_in,
                                    time_node_num=time_node_num, patch_pre_node=patch_pre_node, gnn_r_degree=gnn_r_degree, gnn_hidden_size=gnn_hidden_size,
                                    d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                    pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
    
    def forward(self, x, s_A_i, s_A_v) -> Tensor:
        
        n_vars = x.shape[1]
        
        x = self.W_P(x)
        
        s = generate_local_tokens(x_in=x, time_node_num=self.time_node_num)
        x = embedding_local_tokens(local_tokens=s, detail_tokens=x, time_node_num=self.time_node_num)
        
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        u = self.dropout(u + self.W_pos)
        u = torch.reshape(u, (-1,n_vars,u.shape[-2],u.shape[-1]))
        
        z = self.encoder(u, s_A_i, s_A_v)
        z_l, z_d = split_local_tokens_from_total_tokens(total_tokens=z, time_node_num=self.time_node_num)
        
        return z_l, z_d


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, c_in,
                    time_node_num, patch_pre_node, gnn_r_degree, gnn_hidden_size,
                    d_k=None, d_v=None, d_ff=None, 
                    norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                    res_attention=False, n_layers=2, pre_norm=False, store_attn=False):
        super().__init__()
        
        assert n_layers >= 1, "please give at least one layer"

        self.main_trans_layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                        attn_dropout=attn_dropout, dropout=dropout,
                                                        activation=activation, res_attention=res_attention,
                                                        pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        
        self.dgl_segment_graph_layers = nn.ModuleList([dgl_graph_for_trans(d_model=d_model, graph_r=gnn_r_degree, graph_hidden_size=gnn_hidden_size)  for k in range(n_layers)])
        
        self.c_in = c_in
        self.res_attention = res_attention
        self.time_node_num = time_node_num
    
    def forward(self, src:Tensor, s_A_i:Tensor, s_A_v:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        
        output = src
        scores = None
        
        batch_size = src.shape[0]
        total_node_num = batch_size * self.c_in * self.time_node_num
        batch_g, batch_weights = genearte_graph_from_coo(batch_index=s_A_i, batch_values=s_A_v, total_node_num=total_node_num)
        
        if self.res_attention:
            for main_trans_mods, dgl_segment_graph_mod in zip(self.main_trans_layers, self.dgl_segment_graph_layers):
                
                output = output.flatten(start_dim=0, end_dim=1)
                
                output, scores = main_trans_mods(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                
                output = output.reshape(-1, self.c_in, output.shape[-2], output.shape[-1])
                
                seq_local_tokens, seq_detail_tokens = split_local_tokens_from_total_tokens(total_tokens=output, time_node_num=self.time_node_num)
                seq_local_tokens = dgl_segment_graph_mod(seq_local_tokens, batch_g, batch_weights)
                
                output = embedding_local_tokens(local_tokens=seq_local_tokens, detail_tokens=seq_detail_tokens, time_node_num=self.time_node_num)
            
            return output
        else:
            raise ValueError("Residual attention is needed")
        
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                    norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))
        
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        
        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
        
        if self.pre_norm:
            src = self.norm_attn(src)
        
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)
        
        if self.pre_norm:
            src = self.norm_ffn(src)
        
        src2 = self.ff(src)
        
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        
        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)
        
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))
    
    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)
        
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        
        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        
        attn_scores = torch.matmul(q, k) * self.scale
        
        if prev is not None: attn_scores = attn_scores + prev
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights
