# Ghi chú (VI): Mã nguồn trong pipeline PatchTST định nghĩa kiến trúc mô hình và luồng forward
#    tạo dự báo.
# Thành phần chính trong file: Model.
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np


# Mô hình Transformer cơ bản sử dụng Full Attention với độ phức tạp O(L^2).
class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # Lấy độ dài dự báo và thiết lập có trả về ma trận attention hay không.
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Khởi tạo lớp Embedding cho encoder và decoder theo kiểu embedding được cấu hình.
        # Embedding kết hợp thông tin giá trị chuỗi và thông tin thời gian/vị trí tương ứng.
        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            # Sử dụng embedding không có thành phần positional embedding.
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            # Sử dụng embedding không có thành phần temporal embedding.
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            # Loại bỏ cả positional embedding và temporal embedding.
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        # Khởi tạo Encoder gồm nhiều EncoderLayer.
        # Mỗi lớp sử dụng FullAttention để học mối quan hệ giữa các bước thời gian.
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Khởi tạo Decoder gồm nhiều DecoderLayer.
        # Decoder sử dụng masked self-attention và cross-attention với đầu ra Encoder.
        # Lớp Linear cuối cùng ánh xạ biểu diễn về số biến đầu ra c_out.
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    # Luồng forward: đưa dữ liệu qua Embedding, Encoder và Decoder để tạo dự báo.
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # Mã hóa chuỗi đầu vào và thông tin thời gian thành biểu diễn đặc trưng.
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder học các mối quan hệ giữa các bước thời gian trong chuỗi đầu vào.
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Mã hóa dữ liệu đầu vào của Decoder.
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        # Decoder kết hợp thông tin từ chính đầu vào decoder và đầu ra encoder.
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # Chỉ lấy pred_len bước cuối cùng làm kết quả dự báo.
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]