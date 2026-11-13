# -----------------------------------------------------------------------------
# This file is adapted from PatchTST (https://github.com/PatchTST/PatchTST)
# Original project: “PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.”
# Original repository license: Apache-2.0
# Copyright (c) Nie et al.
#
# Modifications made by Yuan Feng et al., 2025:
# Simplify the code workflow.
# -----------------------------------------------------------------------------

from data_provider.data_loader import Dataset_ETTh, Dataset_ETTm, Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETTh,
    'ETTh2': Dataset_ETTh,
    'ETTm1': Dataset_ETTm,
    'ETTm2': Dataset_ETTm,
    'custom': Dataset_Custom,
    'exchange_rate': Dataset_Custom,
    'weather': Dataset_Custom,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        
    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        data_path=args.data_path,
        target=args.target,
        scale=args.scale
    )
    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    return data_set, data_loader
