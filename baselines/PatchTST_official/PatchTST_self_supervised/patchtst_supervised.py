# Ghi chú (VI): Điểm vào huấn luyện có giám sát của PatchTST.
#    File thực hiện toàn bộ quy trình: cấu hình tham số, tạo mô hình,
#    tìm learning rate phù hợp, huấn luyện và đánh giá kết quả dự báo.
# Thành phần chính trong file: get_model, find_lr, train_func, test_func.

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.patchTST import PatchTST
from src.learner import Learner
from src.callback.core import *
from src.callback.tracking import *
from src.callback.scheduler import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from datautils import get_dls

import argparse

# Khai báo các tham số cho dữ liệu, cấu trúc PatchTST và quá trình huấn luyện.
parser = argparse.ArgumentParser()

# Dataset and dataloader: cấu hình bộ dữ liệu và DataLoader
parser.add_argument('--dset', type=str, default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int, default=336, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--use_time_features', type=int, default=0, help='whether to use time features or not')

# Patch: chia chuỗi đầu vào thành các đoạn nhỏ có độ dài patch_len,
# mỗi patch tiếp theo cách patch trước một khoảng stride.
parser.add_argument('--patch_len', type=int, default=32, help='patch length')
parser.add_argument('--stride', type=int, default=16, help='stride between patch')

# RevIN: chuẩn hóa dữ liệu theo từng chuỗi đầu vào và khôi phục thang đo sau dự báo.
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')

# Model args: cấu hình kiến trúc Transformer trong PatchTST
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')

# Optimization args: cấu hình số epoch và learning rate
parser.add_argument('--n_epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

# Định danh mô hình và lựa chọn loại mô hình
parser.add_argument('--model_id', type=int, default=1, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')

# Chọn chế độ: 1 là huấn luyện, 0 là chỉ đánh giá mô hình đã lưu
parser.add_argument('--is_train', type=int, default=1, help='training the model')


args = parser.parse_args()
print('args:', args)

# Tạo tên checkpoint dựa trên cấu hình để phân biệt các lần thực nghiệm.
args.save_model_name = 'patchtst_supervised'+'_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride)+'_epochs'+str(args.n_epochs) + '_model' + str(args.model_id)

# Đường dẫn lưu trọng số mô hình theo dataset và loại mô hình.
args.save_path = 'saved_models/' + args.dset + '/patchtst_supervised/' + args.model_type + '/'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

def get_model(c_in, args):
    """
    c_in: số lượng biến đầu vào (số chuỗi thời gian).
    """
    # Tính số patch được tạo từ chuỗi đầu vào dựa trên context_points,
    # patch_len và stride.
    # Công thức: số patch = floor((độ dài đầu vào - patch_len) / stride) + 1.
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1
    print('number of patches:', num_patch)

    # Khởi tạo PatchTST với các tham số đã cấu hình.
    # Mỗi biến được chia thành các patch và xử lý bằng Transformer
    # với shared_embedding=True để dùng chung trọng số embedding giữa các biến.
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
                head_type='prediction',
                res_attention=False
                )
    return model

def find_lr():
    # Lấy DataLoader gồm dữ liệu train, validation và test.
    dls = get_dls(args)

    # Khởi tạo mô hình với số biến đầu vào lấy từ bộ dữ liệu.
    model = get_model(dls.vars, args)

    # Sử dụng MSE làm hàm mất mát để đo sai lệch giữa giá trị dự báo
    # và giá trị thực tế.
    loss_func = torch.nn.MSELoss(reduction='mean')

    # RevIN chuẩn hóa từng chuỗi nếu được bật.
    # PatchCB chuyển dữ liệu đầu vào thành các patch trước khi đưa vào mô hình.
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

    # Learner kết nối DataLoader, mô hình, hàm loss và các callback.
    learn = Learner(dls, model, loss_func, cbs=cbs)

    # Thử nghiệm các learning rate khác nhau và đề xuất giá trị phù hợp.
    return learn.lr_finder()


def train_func(lr=args.lr):
    # Tải dữ liệu và thông tin về số biến, chiều dài đầu vào,
    # chiều dài chuỗi dự báo.
    dls = get_dls(args)
    print('in out', dls.vars, dls.c, dls.len)

    # Khởi tạo mô hình PatchTST.
    model = get_model(dls.vars, args)

    # Hàm mất mát MSE: tối ưu sai số bình phương trung bình.
    loss_func = torch.nn.MSELoss(reduction='mean')

    # Các callback được thực hiện trong quá trình huấn luyện:
    # - RevInCB: chuẩn hóa và khôi phục dữ liệu nếu sử dụng RevIN.
    # - PatchCB: chia chuỗi đầu vào thành các patch.
    # - SaveModelCB: lưu checkpoint có valid_loss tốt nhất.
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_model_name,
                     path=args.save_path )
        ]

    # Learner quản lý quá trình huấn luyện và theo dõi chỉ số MSE.
    learn = Learner(dls, model,
                        loss_func,
                        lr=lr,
                        cbs=cbs,
                        metrics=[mse]
                        )

    # Huấn luyện theo One Cycle Learning Rate:
    # learning rate thay đổi trong quá trình huấn luyện để hỗ trợ tối ưu mô hình.
    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)


def test_func():
    # Xác định đường dẫn checkpoint đã được lưu trong quá trình huấn luyện.
    weight_path = args.save_path + args.save_model_name + '.pth'

    # Tải lại DataLoader và khởi tạo kiến trúc mô hình tương ứng.
    dls = get_dls(args)
    model = get_model(dls.vars, args)

    # Áp dụng lại đúng các bước tiền xử lý như khi huấn luyện:
    # RevIN (nếu bật) và chia chuỗi thành patch.
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

    # Tạo Learner để nạp trọng số và đánh giá trên tập test.
    learn = Learner(dls, model, cbs=cbs)

    # Trả về:
    # - out[0]: giá trị dự báo (prediction).
    # - out[1]: giá trị thực tế (target).
    # - out[2]: các chỉ số đánh giá MSE và MAE.
    out = learn.test(dls.test, weight_path=weight_path, scores=[mse,mae])
    return out

if __name__ == '__main__':

    if args.is_train:   # Chế độ huấn luyện
        # Bước 1: tìm learning rate phù hợp trước khi huấn luyện chính thức.
        suggested_lr = find_lr()
        print('suggested lr:', suggested_lr)

        # Bước 2: huấn luyện mô hình với learning rate được đề xuất.
        train_func(suggested_lr)

    else:   # Chế độ đánh giá
        # Yêu cầu checkpoint đã được tạo ở đúng đường dẫn cấu hình.
        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)

    print('----------- Complete! -----------')