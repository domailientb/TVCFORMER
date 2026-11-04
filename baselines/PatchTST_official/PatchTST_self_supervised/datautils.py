# Ghi chú (VI): Lựa chọn bộ dữ liệu, khởi tạo DataLoader và thiết lập các thông tin
#    cần thiết về số lượng biến, độ dài chuỗi đầu vào và đầu ra cho mô hình.
# Thành phần chính trong file: DSETS, get_dls.

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *


# Danh sách các bộ dữ liệu được hỗ trợ trong dự án.
DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange'
        ]


# Khởi tạo DataLoader tương ứng với dataset và các tham số được cung cấp.
def get_dls(params):

    # Kiểm tra dataset được yêu cầu có nằm trong danh sách hỗ trợ hay không.
    # Nếu chưa có thuộc danh sách, chương trình sẽ báo lỗi.
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"

    # Nếu chưa khai báo use_time_features thì mặc định không sử dụng đặc trưng thời gian.
    if not hasattr(params, 'use_time_features'):
        params.use_time_features = False

    # Dataset ETTm1: dữ liệu ETT có tần suất theo phút.
    if params.dset == 'ettm1':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'

        # Cấu hình độ dài chuỗi đầu vào, phần nhãn và khoảng thời gian dự báo.
        size = [params.context_points, 0, params.target_points]

        # Khởi tạo DataLoaders với lớp Dataset_ETT_minute.
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    # Dataset ETTm2: tương tự ETTm1 nhưng sử dụng dữ liệu ETTm2.
    elif params.dset == 'ettm2':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    # Dataset ETTh1: dữ liệu ETT có tần suất theo giờ.
    elif params.dset == 'etth1':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    # Dataset ETTh2: sử dụng cùng lớp xử lý dữ liệu theo giờ như ETTh1.
    elif params.dset == 'etth2':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    # Dataset Electricity: sử dụng Dataset_Custom để đọc dữ liệu CSV thông thường.
    elif params.dset == 'electricity':
        root_path = '/data/datasets/public/electricity/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'electricity.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    # Dataset Traffic: dữ liệu lưu lượng giao thông.
    elif params.dset == 'traffic':
        root_path = '/data/datasets/public/traffic/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'traffic.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    # Dataset Weather: dữ liệu thời tiết.
    elif params.dset == 'weather':
        root_path = '/data/datasets/public/weather/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    # Dataset Illness: dữ liệu bệnh tật quốc gia.
    elif params.dset == 'illness':
        root_path = '/data/datasets/public/illness/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'national_illness.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    # Dataset Exchange: dữ liệu tỷ giá hối đoái.
    elif params.dset == 'exchange':
        root_path = '/data/datasets/public/exchange_rate/'
        size = [params.context_points, 0, params.target_points]

        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'exchange_rate.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    # Lấy số lượng biến đầu vào và độ dài chuỗi đầu vào từ một mẫu dữ liệu.
    # dls.vars: số biến trong chuỗi đầu vào.
    # dls.len: độ dài chuỗi đầu vào theo context_points.
    # dls.c: số bước thời gian trong chuỗi đầu ra.
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]

    return dls


# Khối chạy thử: thiết lập tham số mẫu, khởi tạo DataLoader và kiểm tra kích thước batch.
if __name__ == "__main__":

    # Khai báo cấu hình thử nghiệm với dataset ETTh2.
    class Params:
        dset = 'etth2'
        context_points = 384
        target_points = 96
        batch_size = 64
        num_workers = 8
        with_ray = False
        features = 'M'

    # Khởi tạo DataLoader theo cấu hình đã khai báo.
    params = Params
    dls = get_dls(params)

    # Duyệt qua DataLoader validation để kiểm tra kích thước dữ liệu.
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)

    # Dừng chương trình tại đây để kiểm tra dữ liệu bằng debugger.
    breakpoint()