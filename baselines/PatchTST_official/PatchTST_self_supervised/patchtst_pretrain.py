# Ghi chú (VI): Điểm vào tiền huấn luyện tự giám sát của PatchTST.
# Mô hình che ngẫu nhiên một phần patch đầu vào, sau đó học cách tái tạo
# các patch bị che. Trọng số tốt nhất được lưu lại để sử dụng khi fine-tuning.
# Thành phần chính trong file: get_model, find_lr, pretrain_func.

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *


import argparse

parser = argparse.ArgumentParser()

# Dataset and dataloader: cấu hình dữ liệu dùng cho tiền huấn luyện.
parser.add_argument('--dset_pretrain', type=str, default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')

# Patch: chia chuỗi đầu vào thành các patch có độ dài patch_len,
# các patch liên tiếp cách nhau một khoảng stride.
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')

# RevIN: chuẩn hóa từng chuỗi đầu vào; denorm=False vì tác vụ này
# tập trung tái tạo dữ liệu trong không gian đã chuẩn hóa.
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')

# Model args: cấu hình kiến trúc Transformer của PatchTST.
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')

# Pretrain mask: tỷ lệ patch đầu vào bị che ngẫu nhiên.
# Ví dụ mask_ratio=0.4 nghĩa là che khoảng 40% số patch.
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')

# Optimization args: số epoch và learning rate cho tiền huấn luyện.
parser.add_argument('--n_epochs_pretrain', type=int, default=10, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

# Định danh checkpoint và loại mô hình.
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


args = parser.parse_args()
print('args:', args)

# Đặt tên checkpoint theo cấu hình để phân biệt các lần tiền huấn luyện.
args.save_pretrained_model = 'patchtst_pretrained_cw'+str(args.context_points)+'_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-pretrain' + str(args.n_epochs_pretrain) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.pretrained_model_id)

# Thư mục lưu trọng số và lịch sử loss theo từng dataset.
args.save_path = 'saved_models/' + args.dset_pretrain + '/masked_patchtst/' + args.model_type + '/'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


# Thiết lập thiết bị tính toán: ưu tiên GPU nếu có, nếu không sử dụng CPU.
set_device()


def get_model(c_in, args):
    """
    c_in: số lượng biến đầu vào.
    """
    # Tính số patch được tạo từ chuỗi đầu vào.
    # Cần giữ cấu hình patch và số patch tương thích với giai đoạn fine-tuning.
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1
    print('number of patches:', num_patch)

    # Khởi tạo PatchTST với head_type='pretrain'.
    # Khác với head dự báo, pretrain head có nhiệm vụ tái tạo đầu vào đã bị che.
    model = PatchTST(c_in=c_in,
                target_dim=args.target_points,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act='relu',
                head_type='pretrain',
                res_attention=False
                )

    # Thống kê số tham số có thể được cập nhật trong quá trình học.
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr():
    # Tải dữ liệu dùng cho tiền huấn luyện.
    dls = get_dls(args)

    # Khởi tạo mô hình với số biến đầu vào lấy từ bộ dữ liệu.
    model = get_model(dls.vars, args)

    # Sử dụng MSE để đo sai lệch giữa patch được tái tạo và dữ liệu gốc.
    loss_func = torch.nn.MSELoss(reduction='mean')

    # RevIN chuẩn hóa dữ liệu nếu được bật.
    # denorm=False giữ dữ liệu ở không gian chuẩn hóa để thực hiện tái tạo.
    # PatchMaskCB chia dữ liệu thành patch và che ngẫu nhiên theo mask_ratio.
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio)]

    # Learner kết nối dữ liệu, mô hình, hàm loss và các callback.
    learn = Learner(dls, model,
                        loss_func,
                        lr=args.lr,
                        cbs=cbs,
                        )

    # Dò thử các learning rate và trả về giá trị được đề xuất.
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def pretrain_func(lr=args.lr):
    # Tải dữ liệu và tạo DataLoader cho quá trình tiền huấn luyện.
    dls = get_dls(args)

    # Khởi tạo PatchTST ở chế độ tái tạo patch bị che.
    model = get_model(dls.vars, args)

    # Hàm mất mát MSE dùng để tối ưu khả năng tái tạo dữ liệu.
    loss_func = torch.nn.MSELoss(reduction='mean')

    # Các callback trong quá trình tiền huấn luyện:
    # - RevInCB: chuẩn hóa dữ liệu, không thực hiện khôi phục thang đo.
    # - PatchMaskCB: che ngẫu nhiên các patch và xác định vùng cần tái tạo.
    # - SaveModelCB: lưu checkpoint có valid_loss tốt nhất.
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
         PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio),
         SaveModelCB(monitor='valid_loss', fname=args.save_pretrained_model,
                        path=args.save_path)
        ]

    # Learner quản lý quá trình học từ các patch bị che.
    learn = Learner(dls, model,
                        loss_func,
                        lr=lr,
                        cbs=cbs,
                        #metrics=[mse]
                        )

    # Tiền huấn luyện theo One Cycle Learning Rate trong số epoch đã cấu hình.
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)

    # Lấy loss trên tập train và validation sau từng epoch.
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']

    # Lưu lịch sử loss thành CSV để theo dõi quá trình hội tụ.
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_pretrained_model + '_losses.csv', float_format='%.6f', index=False)


if __name__ == '__main__':

    # Sử dụng dataset tiền huấn luyện làm dataset đầu vào cho get_dls.
    args.dset = args.dset_pretrain

    # Dò learning rate phù hợp trước khi bắt đầu tiền huấn luyện.
    suggested_lr = find_lr()

    # Thực hiện tiền huấn luyện tự giám sát và lưu checkpoint.
    pretrain_func(suggested_lr)

    print('pretraining completed')