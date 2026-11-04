# Ghi chú (VI): Điểm vào chạy thí nghiệm dự báo chuỗi thời gian.
# File tiếp nhận tham số dòng lệnh, thiết lập môi trường và điều phối
# quá trình huấn luyện, kiểm thử hoặc dự báo bằng Exp_Main.
# Thành phần chính: cấu hình tham số, cố định random seed, tạo mã thí nghiệm,
# gọi exp.train(), exp.test() và exp.predict().

import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # Cố định random seed để tăng khả năng tái lập kết quả giữa các lần chạy.
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # Cấu hình chung: lựa chọn chế độ chạy, mã thí nghiệm và mô hình.
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # Cấu hình dữ liệu: đường dẫn, loại bài toán và biến cần dự báo.
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Cấu hình bài toán dự báo:
    # seq_len: độ dài chuỗi đầu vào; label_len: độ dài phần nhãn đưa vào decoder;
    # pred_len: số bước thời gian cần dự báo.
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST: các tham số riêng cho quá trình chia patch và dự báo.
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers: tham số cấu trúc Transformer và các mô hình liên quan.
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # Optimization: cấu hình số worker, số lần chạy, epoch, batch size,
    # learning rate, early stopping và phương pháp điều chỉnh learning rate.
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU: lựa chọn thiết bị tính toán và cấu hình chạy đa GPU.
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # Đọc các tham số được truyền vào khi chạy chương trình.
    args = parser.parse_args()

    # Cố định seed cho Python, PyTorch và NumPy để kết quả thực nghiệm
    # có tính ổn định và dễ so sánh giữa các lần chạy.
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Chỉ sử dụng GPU khi người dùng bật tùy chọn và CUDA khả dụng.
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # Nếu chạy đa GPU, tách danh sách ID thiết bị và chọn GPU đầu tiên
    # làm thiết bị chính.
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # In toàn bộ cấu hình để kiểm tra và lưu lại thông tin thực nghiệm.
    print('Args in experiment:')
    print(args)

    # Chọn lớp thực nghiệm chính, chịu trách nhiệm quản lý train/test/predict.
    Exp = Exp_Main

    if args.is_training:
        # Lặp lại thí nghiệm itr lần để đánh giá tính ổn định của kết quả.
        for ii in range(args.itr):

            # Tạo mã định danh từ cấu hình mô hình, dữ liệu và tham số.
            # Mã này giúp phân biệt checkpoint và kết quả của từng lần chạy.
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,ii)

            # Khởi tạo đối tượng thực nghiệm với cấu hình đã thiết lập.
            exp = Exp(args)

            # Huấn luyện mô hình trên tập train và theo dõi quá trình học.
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            # Đánh giá mô hình đã huấn luyện trên tập test.
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            # Nếu bật do_predict, thực hiện dự báo trên dữ liệu tương lai chưa quan sát.
            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            # Giải phóng bộ nhớ GPU sau mỗi lần chạy.
            torch.cuda.empty_cache()

    else:
        # Chế độ chỉ kiểm thử: không huấn luyện lại mô hình.
        # Tạo mã thí nghiệm tương ứng để tìm đúng checkpoint đã lưu.
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                    args.model,
                                                                                                    args.data,
                                                                                                    args.features,
                                                                                                    args.seq_len,
                                                                                                    args.label_len,
                                                                                                    args.pred_len,
                                                                                                    args.d_model,
                                                                                                    args.n_heads,
                                                                                                    args.e_layers,
                                                                                                    args.d_layers,
                                                                                                    args.d_ff,
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil,
                                                                                                    args.des, ii)

        # Khởi tạo đối tượng thực nghiệm và đánh giá checkpoint trên tập test.
        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)

        # Giải phóng bộ nhớ GPU sau khi kiểm thử.
        torch.cuda.empty_cache()