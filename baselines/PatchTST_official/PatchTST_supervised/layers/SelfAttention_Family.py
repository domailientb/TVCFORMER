# Ghi chú (VI): Các biến thể attention đầy đủ/thưa và lớp chiếu nhiều đầu cho Transformer.
# Thành phần chính trong file: FullAttention, ProbAttention, AttentionLayer.

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import os


# FullAttention tính attention trên toàn bộ cặp Query-Key.
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # queries: [B, L, H, E], values: [B, S, H, D].
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Tính độ tương đồng giữa Query và Key cho từng attention head.
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Áp dụng causal mask để ngăn vị trí hiện tại nhìn thấy các vị trí tương lai.
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Softmax chuyển điểm tương đồng thành trọng số attention.
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # Dùng trọng số attention tổng hợp thông tin từ Value.
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # Trả về ma trận attention nếu cấu hình yêu cầu lưu lại.
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


# ProbAttention là attention thưa, chỉ tập trung tính toán trên các Query quan trọng.
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    # Lấy mẫu Key để ước lượng mức độ quan trọng của các Query và chọn Top-k Query.
    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Mở rộng Key để lấy mẫu tương ứng với từng Query.
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # Đo độ thưa của từng Query dựa trên chênh lệch giữa giá trị lớn nhất và giá trị trung bình.
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Chỉ giữ lại các Query được đánh giá là quan trọng để tính attention chi tiết.
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    # Khởi tạo context ban đầu cho toàn bộ các vị trí Query.
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # Khi không dùng mask, khởi tạo context bằng trung bình của Value.
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # Với causal attention, dùng tổng tích lũy để không truy cập thông tin tương lai.
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    # Cập nhật context tại các vị trí Query đã được chọn.
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            # Tạo mask tương ứng với các Query được chọn.
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Tính trọng số attention cho các Query quan trọng.
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        # Cập nhật context bằng tổng có trọng số của các Value.
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)

        # Nếu cần, tạo ma trận attention đầy đủ để phục vụ việc theo dõi và phân tích.
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        # Đọc kích thước đầu vào trước khi chuyển đổi thứ tự các chiều.
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # Đưa chiều số head lên trước chiều độ dài chuỗi.
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # Số lượng Key lấy mẫu và số Query được chọn phụ thuộc logarit độ dài chuỗi.
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        # Giới hạn số lượng mẫu không vượt quá độ dài chuỗi.
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # Tính điểm attention cho các Query quan trọng đã được lựa chọn.
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Áp dụng hệ số scale để ổn định giá trị điểm attention.
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # Khởi tạo context cho toàn bộ vị trí Query.
        context = self._get_initial_context(values, L_Q)

        # Cập nhật context tại các vị trí Query được chọn.
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


# AttentionLayer chiếu đầu vào thành nhiều head rồi gọi cơ chế attention bên trong.
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        # Xác định số chiều Key và Value của mỗi head nếu chưa được cấu hình.
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Chiếu Query, Key, Value và tách thành H attention head.
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Thực hiện attention bằng FullAttention hoặc ProbAttention được truyền vào.
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        # Ghép các head lại và chiếu về số chiều d_model ban đầu.
        out = out.view(B, L, -1)

        return self.out_projection(out), attn