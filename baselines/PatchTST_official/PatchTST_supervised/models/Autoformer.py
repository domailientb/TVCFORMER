# Ghi chú (VI): Mã nguồn trong pipeline PatchTST định nghĩa kiến trúc mô hình và luồng forward
#    tạo dự báo.
# Thành phần chính trong file: Model.
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos,DataEmbedding_wo_pos_temp,DataEmbedding_wo_temp
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


# Mô hình Autoformer sử dụng AutoCorrelation để khai thác quan hệ chuỗi theo chu kỳ,
# với độ phức tạp được thiết kế ở mức O(LlogL).
class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # Lấy độ dài chuỗi đầu vào, phần nhãn decoder và độ dài dự báo.
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Khởi tạo module phân rã chuỗi thành thành phần xu hướng và thành phần mùa vụ.
        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Khởi tạo embedding cho encoder và decoder.
        # Autoformer tận dụng thông tin tuần tự từ AutoCorrelation nên có thể loại bỏ positional embedding.
        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            # Sử dụng embedding không có temporal embedding.
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

        # Khởi tạo Encoder sử dụng AutoCorrelation thay cho Full Attention.
        # Các EncoderLayer đồng thời thực hiện phân rã chuỗi thông qua moving_avg.
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # Khởi tạo Decoder với AutoCorrelation.
        # Decoder kết hợp masked AutoCorrelation và cross-AutoCorrelation để khai thác
        # quan hệ trong chuỗi decoder và giữa decoder với encoder.
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    # Luồng forward: phân rã chuỗi, mã hóa thông tin và kết hợp hai thành phần dự báo.
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # Tính giá trị trung bình của chuỗi đầu vào và lặp lại cho pred_len bước dự báo.
        # Đây là giá trị khởi tạo cho thành phần xu hướng của decoder.
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)

        # Khởi tạo các giá trị 0 cho thành phần mùa vụ tại khoảng thời gian cần dự báo.
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)

        # Phân rã chuỗi đầu vào thành thành phần mùa vụ và thành phần xu hướng.
        seasonal_init, trend_init = self.decomp(x_enc)

        # Chuẩn bị đầu vào cho Decoder:
        # Thành phần xu hướng gồm label_len giá trị cuối và phần dự báo khởi tạo bằng mean.
        # Thành phần mùa vụ gồm label_len giá trị cuối và phần dự báo khởi tạo bằng 0.
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # Encoder mã hóa chuỗi đầu vào và học các quan hệ theo thời gian.
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder nhận embedding của thành phần mùa vụ và kết hợp thông tin từ Encoder.
        # Đồng thời cập nhật thành phần xu hướng trong quá trình giải mã.
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)

        # Cộng thành phần mùa vụ và thành phần xu hướng để tạo dự báo cuối cùng.
        # final
        dec_out = trend_part + seasonal_part

        # Chỉ lấy pred_len bước cuối cùng làm kết quả dự báo.
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]