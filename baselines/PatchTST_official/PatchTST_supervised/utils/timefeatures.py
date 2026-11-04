# Ghi chú (VI): Chuyển dấu thời gian thành các đặc trưng lịch như giây,
# phút, giờ, thứ, ngày và tháng. Các đặc trưng được chuẩn hóa về khoảng
# [-0.5, 0.5] để mô hình có thể sử dụng thông tin thời gian làm đầu vào.
# Thành phần chính: TimeFeature, các lớp đặc trưng thời gian,
# time_features_from_frequency_str và time_features.

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    # Lớp cơ sở định nghĩa cách chuyển DatetimeIndex thành đặc trưng số.
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # Phương thức được triển khai cụ thể ở từng lớp đặc trưng.
        pass

    def __repr__(self):
        # Trả về tên lớp khi hiển thị đối tượng.
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Mã hóa giây trong phút về khoảng [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Mã hóa phút trong giờ về khoảng [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Mã hóa giờ trong ngày về khoảng [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Mã hóa thứ trong tuần về khoảng [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Mã hóa ngày trong tháng về khoảng [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Mã hóa ngày trong năm về khoảng [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Mã hóa tháng trong năm về khoảng [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Mã hóa tuần trong năm về khoảng [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Chọn danh sách đặc trưng thời gian phù hợp với tần suất dữ liệu.
    Ví dụ: H là theo giờ, D là theo ngày, T là theo phút.
    """

    # Ánh xạ loại tần suất dữ liệu với những đặc trưng lịch cần trích xuất.
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

    # Trả về các đặc trưng tương ứng với tần suất đã cho.
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    # Báo lỗi nếu tần suất không thuộc các loại được hỗ trợ.
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


def time_features(dates, freq='h'):
    # Áp dụng từng hàm đặc trưng lên toàn bộ mốc thời gian,
    # sau đó ghép kết quả thành ma trận đặc trưng thời gian.
    # Kết quả có dạng (số đặc trưng thời gian, số mốc thời gian).
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])