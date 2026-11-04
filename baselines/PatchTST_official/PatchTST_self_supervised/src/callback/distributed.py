# Hỗ trợ huấn luyện mô hình trên nhiều GPU bằng Distributed Data Parallel (DDP), phân chia dữ liệu cho từng GPU và đồng bộ quá trình tính toán.

# Thành phần chính: DistributedTrainer, DistributedDL.

from .core import Callback
import torch
from torch.utils.data import DistributedSampler, DataLoader, SequentialSampler
from torch.nn.parallel import DistributedDataParallel
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


# Quản lý quá trình huấn luyện phân tán trên nhiều GPU.
class DistributedTrainer(Callback):
    def __init__(self,
                 local_rank,
                 world_size,
                 sync_bn=True,  # Đồng bộ Batch Normalization giữa các GPU.
                 **kwargs
                 ):
        self.local_rank = local_rank  # GPU hiện tại của tiến trình.
        self.world_size = world_size  # Tổng số tiến trình/GPU tham gia.
        self.sync_bn = sync_bn
        self.kwargs = kwargs
        super().__init__()

    # Trước khi huấn luyện: chuyển mô hình sang DDP và phân phối DataLoader.
    def before_fit(self):
        self.learner.model = self.prepare_model(
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model) if self.sync_bn else self.model,
            ddp_kwargs=self.kwargs
        )
        self.old_train_dl = self.dls.train
        self.old_valid_dl = self.dls.valid

        self.learner.dls.train = self._wrap_dl(self.dls.train)
        self.learner.dls.valid = self._wrap_dl(self.dls.valid)

    # Bọc DataLoader thông thường thành DataLoader hỗ trợ phân tán.
    def _wrap_dl(self, dl):
        return dl if isinstance(dl, DistributedDL) else self.prepare_data_loader(dl)

    # Sau huấn luyện: khôi phục mô hình và DataLoader ban đầu.
    def after_fit(self): 
        self.learner.model = self.learner.model.module 
        self.learner.dls.train = self.old_train_dl
        self.learner.dls.valid = self.old_valid_dl

    # Thiết lập thiết bị và bọc mô hình bằng DistributedDataParallel.
    def prepare_model(self,
                      model: torch.nn.Module,
                      move_to_device: bool = True,
                      wrap_ddp: bool = True,
                      ddp_kwargs: Optional[Dict[str, Any]] = None) -> torch.nn.Module:

        ddp_kwargs = ddp_kwargs or {}

        rank = self.local_rank
        device = torch.device(f"cuda:{rank}")

        if torch.cuda.is_available():
            torch.cuda.set_device(device)

        if move_to_device:
            logger.info(f"Moving model to device: {device}")
            model = model.to(device)

        # Chỉ sử dụng DDP khi có nhiều hơn một GPU/tiến trình.
        if wrap_ddp and self.world_size > 1:
            logger.info("Wrapping provided model in DDP.")
            if torch.cuda.is_available():
                model = DistributedDataParallel(
                    model, device_ids=[rank], output_device=rank, **ddp_kwargs)
            else:
                model = DistributedDataParallel(model, **ddp_kwargs)

        return model

    # Phân chia dữ liệu và chuẩn bị DataLoader cho từng GPU.
    def prepare_data_loader(self,
                            data_loader: torch.utils.data.DataLoader,
                            add_dist_sampler: bool = True,
                            move_to_device: bool = True) -> torch.utils.data.DataLoader:

        # Tạo DistributedSampler để mỗi GPU xử lý một phần dữ liệu.
        def with_sampler(loader):
            # Xác định có cần xáo trộn dữ liệu hay không.
            shuffle = not isinstance(loader.sampler, SequentialSampler)

            data_loader_args = {
                "dataset": loader.dataset,
                "batch_size": loader.batch_size,
                "shuffle": False,
                "num_workers": loader.num_workers,
                "collate_fn": loader.collate_fn,
                "pin_memory": loader.pin_memory,
                "drop_last": loader.drop_last,
                "timeout": loader.timeout,
                "worker_init_fn": loader.worker_init_fn,
                "sampler": DistributedSampler(loader.dataset, shuffle=shuffle)
            }
            return DataLoader(**data_loader_args)

        data_loader = with_sampler(data_loader)

        # Chuyển dữ liệu đến GPU tương ứng với tiến trình.
        if move_to_device:
            if torch.cuda.is_available():
                rank = self.local_rank
                device = torch.device(f"cuda:{rank}")
            else:
                device = torch.device("cpu")
            data_loader = DistributedDL(data_loader, device)

        return data_loader


# DataLoader hỗ trợ chuyển từng batch dữ liệu đến thiết bị tính toán tương ứng.
class DistributedDL(DataLoader):
    def __init__(self, base_dataloader: DataLoader, device: torch.device):

        self.__dict__.update(getattr(base_dataloader, "__dict__", {}))
        self.dataloader = base_dataloader
        self.device = device

    # Chuyển các tensor trong batch lên GPU hoặc CPU.
    def _move_to_device(self, item):
        def try_move_device(i):
            try:
                i = i.to(self.device)
            except AttributeError:
                logger.debug(f"Item {i} cannot be moved to device "
                             f"{self.device}.")
            return i

        return tuple(try_move_device(i) for i in item)

    def __len__(self):
        return len(self.dataloader)

    # Lần lượt lấy batch và chuyển đến thiết bị tính toán.
    def __iter__(self):
        iterator = iter(self.dataloader)

        for item in iterator:
            yield self._move_to_device(item)