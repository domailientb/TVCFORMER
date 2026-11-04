# Ghi chú (VI): Chọn dataset và thiết lập DataLoader theo phân vùng train, validation, test hoặc
#    dự đoán.
# Thành phần chính trong file: data_provider.
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    # Dataset được chọn theo args.data; flag quyết định phân vùng và cách tạo batch.
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        # Không xáo trộn dữ liệu kiểm thử để giữ thứ tự thời gian.
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        # Dự đoán dùng một mẫu mỗi batch và không loại bỏ batch cuối.
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        # Train được xáo trộn; các phân vùng huấn luyện dùng batch đầy đủ.
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # size truyền lần lượt độ dài encoder, phần nhãn decoder và chân trời dự báo.
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    # Trả cả dataset (phục vụ scaler/thông tin dữ liệu) và DataLoader để lặp batch.
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
