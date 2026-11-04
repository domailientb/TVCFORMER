# Ghi chú (VI): Cung cấp các hàm hỗ trợ lựa chọn thiết bị CPU/GPU và chuyển mô hình,
#    dữ liệu giữa các thiết bị. Ngoài ra, hỗ trợ chuyển Tensor sang NumPy để xử lý kết quả.
# Thành phần chính trong file: GetAttr, get_device, set_device, default_device,
#    get_available_cuda, to_device, to_numpy.

import torch

import collections
from collections import OrderedDict


# Hỗ trợ truy cập thuộc tính của đối tượng mặc định thông qua đối tượng hiện tại.
# Các thuộc tính được cho phép sẽ được chuyển tiếp sang đối tượng self.default.
class GetAttr:

    "Inherit from this to have all attr accesses in `self._xtra` passed down to `self.default`"
    _default = 'default'

    # Kiểm tra thuộc tính có được phép chuyển tiếp hay không.
    def _component_attr_filter(self, k):
        if k.startswith('__') or k in ('_xtra', self._default):
            return False
        xtra = getattr(self, '_xtra', None)
        return xtra is None or k in xtra

    # Lấy danh sách thuộc tính có thể truy cập từ đối tượng mặc định.
    def _dir(self):
        return [k for k in dir(getattr(self, self._default)) if self._component_attr_filter(k)]

    # Nếu không tìm thấy thuộc tính trong đối tượng hiện tại,
    # thử lấy thuộc tính đó từ đối tượng self.default.
    def __getattr__(self, k):
        if self._component_attr_filter(k):
            attr = getattr(self, self._default, None)
            if attr is not None:
                return getattr(attr, k)
        # raise AttributeError(k)

    # Hiển thị danh sách thuộc tính của đối tượng.
    def __dir__(self):
        return custom_dir(self, self._dir())

    # Khôi phục trạng thái thuộc tính khi giải tuần tự hóa đối tượng.
    # def __getstate__(self): return self.__dict__
    def __setstate__(self, data):
        self.__dict__.update(data)


# Lựa chọn thiết bị chạy mô hình: GPU nếu CUDA khả dụng, ngược lại dùng CPU.
# Nếu có nhiều GPU, ưu tiên GPU có mức sử dụng thấp hơn ngưỡng usage.
def get_device(use_cuda=True, device_id=None, usage=5):
    "Return or set default device; `use_cuda`: None - CUDA if available; True - error if not available; False - CPU"

    # Nếu không có GPU hỗ trợ CUDA thì chuyển sang CPU.
    if not torch.cuda.is_available():
        use_cuda = False
    else:
        # Nếu chưa chỉ định GPU, tìm GPU có mức sử dụng thấp và lấy GPU đầu tiên.
        if device_id is None:
            device_ids = get_available_cuda(usage=usage)
            device_id = device_ids[0]   # get the first available device

        # Thiết lập GPU được chọn làm thiết bị CUDA hiện tại.
        torch.cuda.set_device(device_id)

    # Trả về thiết bị GPU hiện tại hoặc CPU.
    return torch.device(torch.cuda.current_device()) if use_cuda else torch.device('cpu')


# Thiết lập GPU có mức sử dụng thấp hơn ngưỡng usage làm GPU hiện tại.
def set_device(usage=5):
    device_ids = get_available_cuda(usage=usage)
    torch.cuda.set_device(device_ids[0])   # get the first available device


# Trả về thiết bị mặc định: GPU hiện tại nếu CUDA khả dụng, nếu không thì CPU.
def default_device(use_cuda=True):
    "Return or set default device; `use_cuda`: None - CUDA if available; True - error if not available; False - CPU"

    # Kiểm tra khả năng sử dụng CUDA trước khi chọn thiết bị.
    if not torch.cuda.is_available():
        use_cuda = False

    return torch.device(torch.cuda.current_device()) if use_cuda else torch.device('cpu')


# Tìm các GPU có mức sử dụng thấp hơn ngưỡng usage.
def get_available_cuda(usage=10):
    # Nếu CUDA không khả dụng thì không có GPU để lựa chọn.
    if not torch.cuda.is_available():
        return

    # Duyệt qua các GPU và giữ lại những GPU có mức sử dụng thấp hơn ngưỡng.
    device_ids = []
    for device in range(torch.cuda.device_count()):
        if torch.cuda.utilization(device) < usage:
            device_ids.append(device)

    return device_ids


# Chuyển dữ liệu sang thiết bị được chỉ định một cách đệ quy.
# Hỗ trợ Tensor nằm trong dict, list hoặc tuple.
def to_device(b, device=None, non_blocking=False):
    """
    Recursively put `b` on `device`
    components of b are torch tensors
    """

    # Nếu chưa chỉ định thiết bị, sử dụng thiết bị mặc định.
    if device is None:
        device = default_device(use_cuda=True)

    # Nếu dữ liệu là dictionary, chuyển từng giá trị sang thiết bị.
    if isinstance(b, dict):
        return {key: to_device(val, device) for key, val in b.items()}

    # Nếu dữ liệu là list hoặc tuple, chuyển từng phần tử sang thiết bị.
    if isinstance(b, (list, tuple)):
        return type(b)(to_device(o, device) for o in b)

    # Nếu là Tensor, chuyển trực tiếp Tensor sang CPU hoặc GPU.
    return b.to(device, non_blocking=non_blocking)


# Chuyển Tensor sang NumPy để phục vụ xử lý, phân tích hoặc lưu kết quả.
def to_numpy(b):
    """
    Components of b are torch tensors
    """

    # Xử lý đệ quy nếu dữ liệu nằm trong dictionary.
    if isinstance(b, dict):
        return {key: to_numpy(val) for key, val in b.items()}

    # Xử lý đệ quy nếu dữ liệu nằm trong list hoặc tuple.
    if isinstance(b, (list, tuple)):
        return type(b)(to_numpy(o) for o in b)

    # Tách Tensor khỏi computational graph, chuyển về CPU rồi đổi sang NumPy.
    return b.detach().cpu().numpy()