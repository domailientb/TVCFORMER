import math
from typing import Dict, List, Tuple

import numpy as np


CALIBRATION_METHODS = ("global", "channel", "horizon", "horizon_channel")


def split_validation_indices(
    window_count: int,
    pred_len: int,
    monitor_fraction: float = 0.4,
    min_calibration_windows: int = 100,
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Split validation forecasts chronologically with disjoint target ranges."""
    if window_count <= 0 or pred_len <= 0:
        raise ValueError("window_count and pred_len must be positive")
    if not 0.0 < monitor_fraction < 1.0:
        raise ValueError("monitor_fraction must be between 0 and 1")
    if min_calibration_windows < 1:
        raise ValueError("min_calibration_windows must be positive")

    monitor_end = int(math.floor(window_count * monitor_fraction))
    calibration_start = monitor_end + pred_len
    calibration_count = window_count - calibration_start
    if monitor_end < 1 or calibration_count < min_calibration_windows:
        raise ValueError(
            "validation split leaves too few calibration windows: {}".format(
                calibration_count
            )
        )

    monitor_indices = list(range(monitor_end))
    calibration_indices = list(range(calibration_start, window_count))
    split_info = {
        "validation_windows": window_count,
        "monitor_windows": len(monitor_indices),
        "excluded_gap_windows": pred_len,
        "calibration_start_index": calibration_start,
        "calibration_windows": len(calibration_indices),
    }
    return monitor_indices, calibration_indices, split_info


def _validate_arrays(
    lower: np.ndarray, upper: np.ndarray, targets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    targets = np.asarray(targets)
    if lower.ndim != 3 or lower.shape != upper.shape or lower.shape != targets.shape:
        raise ValueError(
            "lower, upper, and targets must share [samples, horizon, channels] shape"
        )
    if lower.size == 0:
        raise ValueError("calibration requires at least one forecast value")
    chunk_size = max(1, min(lower.shape[0], 128))
    for start in range(0, lower.shape[0], chunk_size):
        stop = min(start + chunk_size, lower.shape[0])
        lower_chunk = lower[start:stop]
        upper_chunk = upper[start:stop]
        target_chunk = targets[start:stop]
        if not (
            np.isfinite(lower_chunk).all()
            and np.isfinite(upper_chunk).all()
            and np.isfinite(target_chunk).all()
        ):
            raise ValueError("calibration arrays must be finite")
        if np.any(lower_chunk > upper_chunk):
            raise ValueError("lower bounds must not exceed upper bounds")
    return lower, upper, targets


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> Tuple[float, int]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("a calibration group has no scores")
    rank = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    if rank > scores.size:
        raise ValueError(
            "calibration group is too small for the requested coverage"
        )
    return float(np.partition(scores, rank - 1)[rank - 1]), rank


def fit_cqr_calibration(
    lower: np.ndarray,
    upper: np.ndarray,
    targets: np.ndarray,
    alpha: float = 0.05,
    method: str = "horizon_channel",
    min_calibration_windows: int = 100,
) -> Dict[str, object]:
    """Fit signed CQR corrections using validation forecasts only.

    Each group's nonconformity score is max(lower - y, y - upper). The
    finite-sample corrected order statistic is computed within each group.
    """
    if method not in CALIBRATION_METHODS:
        raise ValueError("unsupported calibration method: {}".format(method))
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if min_calibration_windows < 1:
        raise ValueError("min_calibration_windows must be positive")

    lower, upper, targets = _validate_arrays(lower, upper, targets)
    sample_count, horizon, channels = targets.shape
    if sample_count < min_calibration_windows:
        raise ValueError(
            "only {} calibration windows; at least {} are required".format(
                sample_count, min_calibration_windows
            )
        )

    scores = np.maximum(lower - targets, targets - upper)
    if method == "global":
        groups = [scores.reshape(-1)]
        group_shape = ()
    elif method == "channel":
        groups = [scores[:, :, channel].reshape(-1) for channel in range(channels)]
        group_shape = (channels,)
    elif method == "horizon":
        groups = [scores[:, step, :].reshape(-1) for step in range(horizon)]
        group_shape = (horizon,)
    else:
        groups = [
            scores[:, step, channel]
            for step in range(horizon)
            for channel in range(channels)
        ]
        group_shape = (horizon, channels)

    corrections = []
    ranks = []
    score_counts = []
    for group in groups:
        correction, rank = _finite_sample_quantile(group, alpha)
        corrections.append(correction)
        ranks.append(rank)
        score_counts.append(int(group.size))

    adjustment = np.asarray(corrections, dtype=np.float64)
    if group_shape:
        adjustment = adjustment.reshape(group_shape)
        rank_values: object = np.asarray(ranks).reshape(group_shape).tolist()
        count_values: object = np.asarray(score_counts).reshape(group_shape).tolist()
        adjustment_value: object = adjustment.tolist()
    else:
        rank_values = ranks[0]
        count_values = score_counts[0]
        adjustment_value = float(adjustment[0])

    return {
        "method": method,
        "alpha": float(alpha),
        "nominal_coverage": float(1.0 - alpha),
        "score_definition": "max(lower - target, target - upper)",
        "finite_sample_correction": "ceil((n + 1) * (1 - alpha)) order statistic",
        "calibration_windows": int(sample_count),
        "score_count_per_group": count_values,
        "rank_per_group": rank_values,
        "adjustment": adjustment_value,
    }


def apply_cqr_calibration(
    lower: np.ndarray,
    upper: np.ndarray,
    median: np.ndarray,
    calibrator: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply validation-fitted CQR correction without changing the median."""
    lower, upper, _ = _validate_arrays(lower, upper, lower)
    median = np.asarray(median)
    if median.shape != lower.shape:
        raise ValueError("median must be finite and match interval shape")
    chunk_size = max(1, min(lower.shape[0], 128))
    for start in range(0, lower.shape[0], chunk_size):
        stop = min(start + chunk_size, lower.shape[0])
        lower_chunk = lower[start:stop]
        median_chunk = median[start:stop]
        upper_chunk = upper[start:stop]
        if not np.isfinite(median_chunk).all():
            raise ValueError("median must be finite and match interval shape")
        if np.any(lower_chunk > median_chunk) or np.any(median_chunk > upper_chunk):
            raise ValueError("input quantiles must satisfy lower <= median <= upper")

    method = calibrator.get("method")
    adjustment = np.asarray(calibrator.get("adjustment"), dtype=lower.dtype)
    if method == "global" and adjustment.ndim == 0:
        adjustment = adjustment.reshape(1, 1, 1)
    elif method == "channel" and adjustment.shape == (lower.shape[2],):
        adjustment = adjustment.reshape(1, 1, -1)
    elif method == "horizon" and adjustment.shape == (lower.shape[1],):
        adjustment = adjustment.reshape(1, -1, 1)
    elif method == "horizon_channel" and adjustment.shape == lower.shape[1:]:
        adjustment = adjustment.reshape(1, lower.shape[1], lower.shape[2])
    else:
        raise ValueError("calibration adjustment shape does not match method")

    output_dtype = np.result_type(lower.dtype, upper.dtype, median.dtype)
    calibrated_lower = np.empty(lower.shape, dtype=output_dtype)
    calibrated_upper = np.empty(upper.shape, dtype=output_dtype)
    for start in range(0, lower.shape[0], chunk_size):
        stop = min(start + chunk_size, lower.shape[0])
        lower_out = calibrated_lower[start:stop]
        upper_out = calibrated_upper[start:stop]
        np.subtract(
            lower[start:stop],
            adjustment,
            out=lower_out,
            casting="unsafe",
        )
        np.minimum(lower_out, median[start:stop], out=lower_out)
        np.add(
            upper[start:stop],
            adjustment,
            out=upper_out,
            casting="unsafe",
        )
        np.maximum(upper_out, median[start:stop], out=upper_out)
    return calibrated_lower, calibrated_upper
