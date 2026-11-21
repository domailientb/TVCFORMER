from typing import Dict, Optional

import numpy as np


def inverse_scale(values: np.ndarray, scaler: object) -> np.ndarray:
    """Apply a fitted sklearn-style scaler to [..., channel] arrays."""
    values = np.asarray(values)
    if values.ndim < 2:
        raise ValueError("values must include a channel dimension")
    original_shape = values.shape
    flattened = values.reshape(-1, original_shape[-1])
    inverse = scaler.inverse_transform(flattened)
    return np.asarray(inverse).reshape(original_shape)


def compute_forecast_metrics(
    targets: np.ndarray,
    median: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    nominal_coverage: float = 0.95,
) -> Dict[str, object]:
    """Compute point and, when provided, marginal interval metrics."""
    targets = np.asarray(targets)
    median = np.asarray(median)
    if targets.shape != median.shape or targets.ndim != 3:
        raise ValueError("targets and median must share [sample, horizon, channel] shape")
    if targets.size == 0:
        raise ValueError("metrics require at least one forecast value")
    if not np.isfinite(targets).all() or not np.isfinite(median).all():
        raise ValueError("targets and median forecasts must be finite")

    sample_count, horizon, channels = targets.shape
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    covered_by_channel = np.zeros(channels, dtype=np.float64)
    width_by_channel = np.zeros(channels, dtype=np.float64)
    covered_by_horizon = np.zeros(horizon, dtype=np.float64)
    width_by_horizon = np.zeros(horizon, dtype=np.float64)
    covered_by_horizon_channel = np.zeros((horizon, channels), dtype=np.float64)
    width_by_horizon_channel = np.zeros((horizon, channels), dtype=np.float64)
    covered_total = 0.0
    width_total = 0.0

    # Bound temporary arrays for long horizons and large test sets.
    chunk_size = max(1, min(sample_count, 128))
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        target_chunk = targets[start:stop].astype(np.float64, copy=False)
        median_chunk = median[start:stop].astype(np.float64, copy=False)
        error = median_chunk - target_chunk
        squared_error_sum += float(np.sum(error * error, dtype=np.float64))
        absolute_error_sum += float(np.sum(np.abs(error), dtype=np.float64))

    result: Dict[str, object] = {
        "mse": squared_error_sum / targets.size,
        "mae": absolute_error_sum / targets.size,
    }
    if lower is None and upper is None:
        return result
    if lower is None or upper is None:
        raise ValueError("lower and upper bounds must be provided together")
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must be between 0 and 1")

    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if lower.shape != targets.shape or upper.shape != targets.shape:
        raise ValueError("interval bounds must match target shape")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("interval bounds must be finite")
    if np.any(lower > median) or np.any(median > upper):
        raise ValueError("forecasts must satisfy lower <= median <= upper")

    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        target_chunk = targets[start:stop].astype(np.float64, copy=False)
        lower_chunk = lower[start:stop].astype(np.float64, copy=False)
        upper_chunk = upper[start:stop].astype(np.float64, copy=False)
        covered = (target_chunk >= lower_chunk) & (target_chunk <= upper_chunk)
        widths = upper_chunk - lower_chunk
        covered_total += float(np.count_nonzero(covered))
        width_total += float(np.sum(widths, dtype=np.float64))
        covered_by_channel += np.sum(covered, axis=(0, 1), dtype=np.float64)
        width_by_channel += np.sum(widths, axis=(0, 1), dtype=np.float64)
        covered_by_horizon += np.sum(covered, axis=(0, 2), dtype=np.float64)
        width_by_horizon += np.sum(widths, axis=(0, 2), dtype=np.float64)
        covered_by_horizon_channel += np.sum(covered, axis=0, dtype=np.float64)
        width_by_horizon_channel += np.sum(widths, axis=0, dtype=np.float64)

    coverage = covered_by_channel / (sample_count * horizon)
    ranges = np.ptp(targets.reshape(-1, channels), axis=0)
    normalized_width = np.divide(
        width_by_channel / (sample_count * horizon),
        ranges,
        out=np.zeros_like(ranges, dtype=np.float64),
        where=ranges > 0,
    )
    horizon_coverage = covered_by_horizon / (sample_count * channels)
    horizon_width = width_by_horizon / (sample_count * channels)
    mean_coverage = covered_total / targets.size

    result["interval"] = {
        "nominal_coverage": float(nominal_coverage),
        "picp": mean_coverage,
        "mpiw": width_total / targets.size,
        "pinaw": float(np.mean(normalized_width)),
        "ace": float(abs(mean_coverage - nominal_coverage)),
        "coverage_by_channel": coverage.tolist(),
        "pinaw_by_channel": normalized_width.tolist(),
        "coverage_by_horizon_step": horizon_coverage.tolist(),
        "mean_width_by_horizon_step": horizon_width.tolist(),
        "coverage_by_horizon_channel": (
            covered_by_horizon_channel / sample_count
        ).tolist(),
        "mean_width_by_horizon_channel": (
            width_by_horizon_channel / sample_count
        ).tolist(),
    }
    return result
