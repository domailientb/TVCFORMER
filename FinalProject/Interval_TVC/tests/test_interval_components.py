import numpy as np
import pytest
import torch

from FinalProject.Interval_TVC.losses.quantile_loss import quantile_loss
from FinalProject.Interval_TVC.metrics.interval_metrics import (
    compute_forecast_metrics,
    inverse_scale,
)
from FinalProject.Interval_TVC.models.interval_tvcformer import ordered_quantiles
from FinalProject.Interval_TVC.calibration.conformal import (
    apply_cqr_calibration,
    fit_cqr_calibration,
    split_validation_indices,
)


def test_ordered_quantiles_are_monotonic_and_differentiable():
    raw = torch.randn(2, 4, 3, requires_grad=True)
    quantiles = ordered_quantiles(raw)

    assert quantiles.shape == (2, 4, 3)
    assert torch.all(quantiles[..., 0] <= quantiles[..., 1])
    assert torch.all(quantiles[..., 1] <= quantiles[..., 2])
    quantiles.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_quantile_loss_matches_pinball_definition_and_backpropagates():
    prediction = torch.tensor([[[[0.0, 1.0, 2.0]]]], requires_grad=True)
    target = torch.tensor([[[1.5]]])

    loss = quantile_loss(prediction, target)
    expected = torch.tensor((0.025 * 1.5 + 0.5 * 0.5 + 0.025 * 0.5) / 3)

    assert torch.isclose(loss, expected)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_quantile_loss_rejects_misaligned_shapes():
    with pytest.raises(ValueError, match="dimensions do not align"):
        quantile_loss(torch.zeros(2, 3, 4, 3), torch.zeros(2, 3, 5))


def test_interval_metrics_report_expected_coverage_and_width():
    targets = np.array([[[0.0], [1.0]], [[2.0], [3.0]]])
    median = targets.copy()
    lower = targets - 1.0
    upper = targets + 1.0

    result = compute_forecast_metrics(
        targets, median, lower, upper, nominal_coverage=0.95
    )

    assert result["mse"] == 0.0
    assert result["mae"] == 0.0
    interval = result["interval"]
    assert interval["picp"] == 1.0
    assert interval["mpiw"] == 2.0
    assert interval["pinaw"] == pytest.approx(2.0 / 3.0)
    assert interval["ace"] == pytest.approx(0.05)


def test_interval_metrics_reject_crossing_quantiles():
    values = np.zeros((1, 2, 1))
    with pytest.raises(ValueError, match="lower <= median <= upper"):
        compute_forecast_metrics(values, values, lower=values + 1, upper=values + 2)


def test_inverse_scale_restores_original_values():
    class AffineScaler:
        def inverse_transform(self, values):
            return values * np.array([2.0, 3.0]) + np.array([10.0, -4.0])

    scaled = np.array([[[0.0, 1.0], [2.0, -1.0]]])
    original = inverse_scale(scaled, AffineScaler())

    np.testing.assert_array_equal(
        original, np.array([[[10.0, -1.0], [14.0, -7.0]]])
    )


def test_inverse_scale_is_applied_per_channel_to_each_quantile():
    class AffineScaler:
        def inverse_transform(self, values):
            return values * np.array([2.0, 3.0]) + np.array([10.0, -4.0])

    quantiles = np.zeros((2, 4, 2, 3))
    quantiles[..., 0] = -1.0
    quantiles[..., 1] = 0.0
    quantiles[..., 2] = 1.0
    restored = np.stack(
        [inverse_scale(quantiles[..., q], AffineScaler()) for q in range(3)],
        axis=-1,
    )

    np.testing.assert_array_equal(restored[0, 0, 0], [8.0, 10.0, 12.0])
    np.testing.assert_array_equal(restored[0, 0, 1], [-7.0, -4.0, -1.0])


def test_horizon_channel_cqr_uses_validation_scores_and_preserves_order():
    targets = np.zeros((25, 2, 2))
    lower = np.full_like(targets, -2.0)
    median = np.zeros_like(targets)
    upper = np.full_like(targets, 2.0)
    targets[:, 0, 0] = np.arange(25, dtype=np.float64)

    calibrator = fit_cqr_calibration(
        lower,
        upper,
        targets,
        alpha=0.05,
        method="horizon_channel",
        min_calibration_windows=20,
    )
    calibrated_lower, calibrated_upper = apply_cqr_calibration(
        lower, upper, median, calibrator
    )

    assert np.asarray(calibrator["adjustment"]).shape == (2, 2)
    assert np.all(calibrated_lower <= median)
    assert np.all(median <= calibrated_upper)


def test_validation_split_has_disjoint_monitor_and_calibration_targets():
    monitor, calibration, info = split_validation_indices(
        window_count=2161,
        pred_len=720,
        monitor_fraction=0.4,
        min_calibration_windows=100,
    )

    assert monitor[-1] + 720 < calibration[0]
    assert info["monitor_windows"] == len(monitor)
    assert info["calibration_windows"] == len(calibration)
    assert info["excluded_gap_windows"] == 720


def test_validation_split_rejects_insufficient_calibration_data():
    with pytest.raises(ValueError, match="too few calibration windows"):
        split_validation_indices(
            window_count=1200,
            pred_len=720,
            monitor_fraction=0.4,
            min_calibration_windows=100,
        )


def test_calibration_rejects_non_finite_scores():
    values = np.zeros((20, 2, 1))
    upper = values.copy()
    lower = values.copy()
    targets = values.copy()
    targets[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="must be finite"):
        fit_cqr_calibration(
            lower,
            upper,
            targets,
            method="global",
            min_calibration_windows=10,
        )
