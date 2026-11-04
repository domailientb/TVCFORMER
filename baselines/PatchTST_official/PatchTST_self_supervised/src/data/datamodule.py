# Ghi chú (VI): Xây dựng DataLoader để chia dữ liệu thành các batch
# phục vụ quá trình huấn luyện, đánh giá và kiểm thử mô hình.
# Thành phần chính: DataLoaders.

import warnings

import torch
from torch.utils.data import DataLoader


class DataLoaders:
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int=0,
        collate_fn=None,
        shuffle_train = True,
        shuffle_val = False
    ):
        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size

        # Loại bỏ split khỏi cấu hình chung vì split được truyền riêng
        # khi khởi tạo từng tập train, validation và test.
        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]

        self.dataset_kwargs = dataset_kwargs
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train, self.shuffle_val = shuffle_train, shuffle_val

        # Khởi tạo DataLoader riêng cho ba tập dữ liệu.
        self.train = self.train_dataloader()
        self.valid = self.val_dataloader()
        self.test = self.test_dataloader()        


    # Tạo DataLoader cho tập huấn luyện, mặc định có xáo trộn dữ liệu.
    def train_dataloader(self):
        return self._make_dloader("train", shuffle=self.shuffle_train)

    # Tạo DataLoader cho tập validation, mặc định không xáo trộn.
    def val_dataloader(self):        
        return self._make_dloader("val", shuffle=self.shuffle_val)

    # Tạo DataLoader cho tập test, không xáo trộn dữ liệu.
    def test_dataloader(self):
        return self._make_dloader("test", shuffle=False)

    def _make_dloader(self, split, shuffle=False):
        # Khởi tạo dataset tương ứng với từng giai đoạn.
        dataset = self.datasetCls(**self.dataset_kwargs, split=split)

        # Không tạo DataLoader nếu dataset rỗng.
        if len(dataset) == 0:
            return None

        # Chia dữ liệu thành các batch để đưa vào mô hình.
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    # Khai báo tham số dòng lệnh cho batch_size và số worker.
    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )

    def add_dl(self, test_data, batch_size=None, **kwargs):
        # Nếu dữ liệu đã là DataLoader thì sử dụng trực tiếp.
        from ray.train.torch import _WrappedDataLoader
        if isinstance(test_data, DataLoader) or isinstance(test_data, _WrappedDataLoader): 
            return test_data

        # Nếu chưa chỉ định batch_size thì sử dụng giá trị mặc định.
        if batch_size is None:
            batch_size = self.batch_size        

        # Nếu đầu vào chưa phải Dataset thì chuyển thành Dataset.
        if not isinstance(test_data, Dataset):
            test_data = self.train.dataset.new(test_data)        

        # Tạo DataLoader mới từ Dataset đã chuẩn bị.
        test_data = self.train.new(test_data, batch_size, **kwargs)
        return test_data