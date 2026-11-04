# Ghi chú (VI): Chuyển dấu thời gian thành các đặc trưng lịch như giây, phút, giờ,
# ngày trong tuần, ngày trong tháng, tháng và tuần trong năm.
# Các đặc trưng được chuẩn hóa về khoảng [-0.5, 0.5] để thuận tiện cho mô hình xử lý.
# Thành phần chính trong file: TimeFeature, các lớp đặc trưng thời gian,
# time_features_from_frequency_str và time_features.

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


# Lớp cơ sở định nghĩa cách biểu diễn một đặc trưng thời gian.
# Các lớp kế thừa sẽ triển khai cách trích xuất đặc trưng cụ thể.
class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


# Trích xuất giây trong một phút và chuẩn hóa về khoảng [-0.5, 0.5].
class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


# Trích xuất phút trong một giờ và chuẩn hóa về khoảng [-0.5, 0.5].
class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


# Trích xuất giờ trong ngày và chuẩn hóa về khoảng [-0.5, 0.5].
class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


# Trích xuất thứ trong tuần (thứ Hai = 0, Chủ nhật = 6).
class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


# Trích xuất ngày trong tháng và chuẩn hóa về khoảng [-0.5, 0.5].
class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


# Trích xuất ngày thứ bao nhiêu trong năm.
class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


# Trích xuất tháng trong năm (tháng 1 = 1, tháng 12 = 12).
class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


# Trích xuất tuần thứ bao nhiêu trong năm theo lịch ISO.
class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


# Xác định những đặc trưng thời gian cần trích xuất dựa trên tần suất dữ liệu.
# Ví dụ: dữ liệu theo giờ sẽ lấy giờ, thứ, ngày trong tháng và ngày trong năm.
def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    # Ánh xạ từng loại tần suất với danh sách đặc trưng thời gian tương ứng.
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    # Chuyển chuỗi tần suất thành đối tượng offset để xác định loại dữ liệu.
    offset = to_offset(freq_str)

    # Tìm loại tần suất phù hợp và khởi tạo các lớp đặc trưng tương ứng.
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    # Báo lỗi nếu tần suất dữ liệu không thuộc các loại được hỗ trợ.
    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


# Áp dụng các hàm trích xuất đặc trưng lên toàn bộ danh sách thời điểm.
# Kết quả các đặc trưng được ghép thành ma trận, mỗi hàng tương ứng một đặc trưng.
def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])