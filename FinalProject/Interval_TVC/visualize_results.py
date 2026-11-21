import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "calibrated_v1"
CHANNELS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


def read_result(experiment: str, horizon: int, seed: int) -> Dict[str, object]:
    path = (
        RESULTS_ROOT
        / experiment
        / "ETTh1_h{}_s{}".format(horizon, seed)
        / "metrics.json"
    )
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ETTh1 interval calibration results")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    args = parser.parse_args()

    baseline_results: List[Dict[str, object]] = []
    interval_results: List[Dict[str, object]] = []
    calibrated_results: List[Dict[str, object]] = []
    for horizon in args.horizons:
        baseline_results.append(read_result("E0", horizon, args.seed))
        interval_results.append(read_result("E1", horizon, args.seed))
        calibrated_results.append(read_result("E1-C", horizon, args.seed))

    output_dir = RESULTS_ROOT / "plots" / "seed{}".format(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, label in zip(axes, ("mse", "mae"), ("MSE", "MAE")):
        axis.plot(
            args.horizons,
            [
                result["metrics_original_units"][metric]
                for result in baseline_results
            ],
            marker="o",
            label="E0 TVCFormer",
        )
        axis.plot(
            args.horizons,
            [
                result["metrics_original_units"][metric]
                for result in interval_results
            ],
            marker="o",
            label="E1 interval median",
        )
        axis.plot(
            args.horizons,
            [
                result["metrics_original_units"][metric]
                for result in calibrated_results
            ],
            marker="x",
            linestyle="--",
            label="E1-C calibrated median",
        )
        axis.set(xlabel="Prediction horizon", ylabel=label)
        axis.legend()
    figure.tight_layout()
    figure.savefig(str(output_dir / "point_metrics_by_horizon.png"), dpi=150)
    plt.close(figure)

    raw_intervals = [
        result["metrics_original_units"]["interval"] for result in interval_results
    ]
    calibrated_intervals = [
        result["metrics_original_units"]["interval"]
        for result in calibrated_results
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        args.horizons,
        [metrics["picp"] for metrics in raw_intervals],
        marker="o",
        label="E1 uncalibrated",
    )
    axis.plot(
        args.horizons,
        [metrics["picp"] for metrics in calibrated_intervals],
        marker="o",
        label="E1-C calibrated",
    )
    axis.axhline(0.95, linestyle="--", color="black", label="Nominal 95%")
    axis.set(xlabel="Prediction horizon", ylabel="PICP", ylim=(0.0, 1.0))
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(output_dir / "picp_before_after.png"), dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        args.horizons,
        [metrics["mpiw"] for metrics in raw_intervals],
        marker="o",
        label="E1 uncalibrated",
    )
    axis.plot(
        args.horizons,
        [metrics["mpiw"] for metrics in calibrated_intervals],
        marker="o",
        label="E1-C calibrated",
    )
    axis.set(xlabel="Prediction horizon", ylabel="MPIW (original units)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(output_dir / "mpiw_before_after.png"), dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(args.horizons), 1, figsize=(11, 3 * len(args.horizons)), squeeze=False
    )
    x = np.arange(len(CHANNELS))
    width = 0.36
    for axis, horizon, raw, calibrated in zip(
        axes[:, 0], args.horizons, raw_intervals, calibrated_intervals
    ):
        axis.bar(
            x - width / 2,
            raw["coverage_by_channel"],
            width,
            label="E1 uncalibrated",
        )
        axis.bar(
            x + width / 2,
            calibrated["coverage_by_channel"],
            width,
            label="E1-C calibrated",
        )
        axis.axhline(0.95, linestyle="--", color="black")
        axis.set(
            title="Horizon {}".format(horizon),
            ylabel="PICP",
            ylim=(0.0, 1.0),
            xticks=x,
            xticklabels=CHANNELS,
        )
        axis.legend()
    figure.tight_layout()
    figure.savefig(str(output_dir / "coverage_by_channel.png"), dpi=150)
    plt.close(figure)

    if 720 in args.horizons:
        index = args.horizons.index(720)
        raw = raw_intervals[index]
        calibrated = calibrated_intervals[index]
        steps = np.arange(1, 721)
        figure, axis = plt.subplots(figsize=(12, 4))
        axis.plot(steps, raw["coverage_by_horizon_step"], label="E1 uncalibrated")
        axis.plot(
            steps,
            calibrated["coverage_by_horizon_step"],
            label="E1-C calibrated",
        )
        axis.axhline(0.95, linestyle="--", color="black", label="Nominal 95%")
        axis.set(xlabel="Forecast step", ylabel="PICP", ylim=(0.0, 1.0))
        axis.legend()
        figure.tight_layout()
        figure.savefig(str(output_dir / "coverage_by_step_h720.png"), dpi=150)
        plt.close(figure)

        forecast_path = (
            RESULTS_ROOT
            / "E1"
            / "ETTh1_h720_s{}".format(args.seed)
            / "test_forecasts.npz"
        )
        with np.load(forecast_path) as forecasts:
            targets = forecasts["targets"]
            median = forecasts["median"]
            lower = forecasts["lower"]
            upper = forecasts["upper"]
            calibrated_lower = forecasts["calibrated_lower"]
            calibrated_upper = forecasts["calibrated_upper"]
        selected = sorted(set((0, len(targets) // 2, len(targets) - 1)))
        figure, axes = plt.subplots(len(selected), 1, figsize=(14, 10))
        axes = np.atleast_1d(axes)
        forecast_steps = np.arange(1, 721)
        for axis, sample_index in zip(axes, selected):
            axis.fill_between(
                forecast_steps,
                lower[sample_index, :, -1],
                upper[sample_index, :, -1],
                color="tab:blue",
                alpha=0.20,
                label="E1 interval",
            )
            axis.fill_between(
                forecast_steps,
                calibrated_lower[sample_index, :, -1],
                calibrated_upper[sample_index, :, -1],
                color="tab:orange",
                alpha=0.20,
                label="E1-C interval",
            )
            axis.plot(
                forecast_steps,
                targets[sample_index, :, -1],
                color="black",
                linewidth=1.0,
                label="OT actual",
            )
            axis.plot(
                forecast_steps,
                median[sample_index, :, -1],
                color="tab:red",
                linewidth=1.0,
                label="Median",
            )
            axis.set_title("OT, test window {}".format(sample_index))
            axis.set_xlabel("Forecast step")
            axis.legend(loc="best", ncol=2)
        figure.tight_layout()
        figure.savefig(str(output_dir / "ot_h720_intervals_before_after.png"), dpi=150)
        plt.close(figure)

    print("Saved plots to {}".format(output_dir))


if __name__ == "__main__":
    main()
