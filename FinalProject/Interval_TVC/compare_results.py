import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "calibrated_v1"
CHANNELS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
METHODS = ("global", "channel", "horizon", "horizon_channel")


def read_result(experiment: str, horizon: int, seed: int) -> Dict[str, object]:
    path = (
        RESULTS_ROOT
        / experiment
        / "ETTh1_h{}_s{}".format(horizon, seed)
        / "metrics.json"
    )
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_csv(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ETTh1 E0, E1, and validation-calibrated E1-C"
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    args = parser.parse_args()

    comparison_rows = []
    method_rows = []
    channel_rows = []
    horizon_rows = []
    markdown_rows = []
    for horizon in args.horizons:
        baseline = read_result("E0", horizon, args.seed)["metrics_original_units"]
        interval = read_result("E1", horizon, args.seed)
        calibrated = read_result("E1-C", horizon, args.seed)
        interval_metrics = interval["metrics_original_units"]
        calibrated_metrics = calibrated["metrics_original_units"]

        row = {
            "pred_len": horizon,
            "E0_mse": baseline["mse"],
            "E0_mae": baseline["mae"],
            "E1_mse": interval_metrics["mse"],
            "E1_mae": interval_metrics["mae"],
            "E1_picp": interval_metrics["interval"]["picp"],
            "E1_mpiw": interval_metrics["interval"]["mpiw"],
            "E1_pinaw": interval_metrics["interval"]["pinaw"],
            "E1_ace": interval_metrics["interval"]["ace"],
            "E1C_mse": calibrated_metrics["mse"],
            "E1C_mae": calibrated_metrics["mae"],
            "E1C_picp": calibrated_metrics["interval"]["picp"],
            "E1C_mpiw": calibrated_metrics["interval"]["mpiw"],
            "E1C_pinaw": calibrated_metrics["interval"]["pinaw"],
            "E1C_ace": calibrated_metrics["interval"]["ace"],
        }
        comparison_rows.append(row)
        method_metrics = {
            "uncalibrated": interval_metrics,
            **{
                method: interval["calibration_methods"][method][
                    "metrics_original_units"
                ]
                for method in METHODS
            },
        }
        for method, metrics in method_metrics.items():
            interval_metrics_for_method = metrics["interval"]
            method_rows.append(
                {
                    "pred_len": horizon,
                    "method": method,
                    "mse": metrics["mse"],
                    "mae": metrics["mae"],
                    "picp": interval_metrics_for_method["picp"],
                    "mpiw": interval_metrics_for_method["mpiw"],
                    "pinaw": interval_metrics_for_method["pinaw"],
                    "ace": interval_metrics_for_method["ace"],
                }
            )
            for channel_index, channel in enumerate(CHANNELS):
                channel_rows.append(
                    {
                        "pred_len": horizon,
                        "method": method,
                        "channel": channel,
                        "picp": interval_metrics_for_method[
                            "coverage_by_channel"
                        ][channel_index],
                        "pinaw": interval_metrics_for_method[
                            "pinaw_by_channel"
                        ][channel_index],
                    }
                )
            for step, (coverage, width) in enumerate(
                zip(
                    interval_metrics_for_method["coverage_by_horizon_step"],
                    interval_metrics_for_method["mean_width_by_horizon_step"],
                ),
                start=1,
            ):
                horizon_rows.append(
                    {
                        "pred_len": horizon,
                        "method": method,
                        "forecast_step": step,
                        "picp": coverage,
                        "mean_width": width,
                    }
                )
        markdown_rows.append(
            "| {pred_len} | {E0_mse:.3f} | {E0_mae:.3f} | "
            "{E1_mse:.3f} | {E1_mae:.3f} | {E1_picp:.3f} | "
            "{E1_mpiw:.3f} | {E1C_picp:.3f} | {E1C_mpiw:.3f} |".format(**row)
        )

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(
        RESULTS_ROOT / "comparison_seed{}.csv".format(args.seed),
        comparison_rows[0].keys(),
        comparison_rows,
    )
    write_csv(
        RESULTS_ROOT / "calibration_methods_seed{}.csv".format(args.seed),
        method_rows[0].keys(),
        method_rows,
    )
    write_csv(
        RESULTS_ROOT / "coverage_by_channel_seed{}.csv".format(args.seed),
        channel_rows[0].keys(),
        channel_rows,
    )
    write_csv(
        RESULTS_ROOT / "coverage_by_horizon_seed{}.csv".format(args.seed),
        horizon_rows[0].keys(),
        horizon_rows,
    )

    markdown = [
        "# ETTh1 Interval TVCFormer — seed {}".format(args.seed),
        "",
        "E1-C uses the predeclared horizon-by-channel CQR calibration fitted on "
        "the held-out validation calibration segment. Test labels are used only "
        "for final metrics.",
        "",
        "| Horizon | E0 MSE | E0 MAE | E1 MSE | E1 MAE | E1 PICP | "
        "E1 MPIW | E1-C PICP | E1-C MPIW |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *markdown_rows,
        "",
        "PICP is marginal empirical coverage over all test windows, forecast "
        "steps, and channels. MPIW is in original dataset units. The "
        "validation windows used for checkpoint selection and calibration are "
        "separated chronologically, with a prediction-horizon gap between "
        "their target ranges. Overlapping time-series data remain dependent, "
        "so nominal conformal guarantees do not strictly apply.",
        "",
    ]
    report_path = RESULTS_ROOT / "report_seed{}.md".format(args.seed)
    report_path.write_text("\n".join(markdown), encoding="utf-8")
    for row in comparison_rows:
        print(row)
    print("Saved comparison and coverage tables under {}".format(RESULTS_ROOT))


if __name__ == "__main__":
    main()
