import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "full_3seeds_v1"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
SEEDS = (2021, 2023, 2025)
PRED_LENS = (96, 192, 336, 720)
EXPERIMENTS = ("E0", "E1", "E1-C")


def write_csv(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def read_metrics(dataset: str, experiment: str, pred_len: int, seed: int) -> Dict[str, object]:
    path = (
        RESULTS_ROOT
        / experiment
        / "{}_h{}_s{}".format(dataset, pred_len, seed)
        / "metrics.json"
    )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_complete(dataset: str, experiment: str, pred_len: int, seed: int) -> bool:
    metrics_dir = RESULTS_ROOT / experiment / "{}_h{}_s{}".format(dataset, pred_len, seed)
    metrics = metrics_dir / "metrics.json"
    forecasts = metrics_dir / "test_forecasts.npz"
    if not metrics.is_file() or not forecasts.is_file():
        return False
    if experiment == "E1":
        if not (metrics_dir / "calibration.json").is_file():
            return False
    try:
        read_metrics(dataset, experiment, pred_len, seed)
    except (OSError, json.JSONDecodeError, KeyError):
        return False
    return True


def main() -> None:
    if not RESULTS_ROOT.is_dir():
        raise FileNotFoundError("full experiment result root does not exist")

    all_runs = []
    coverage_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for pred_len in PRED_LENS:
                for experiment in EXPERIMENTS:
                    if not is_complete(dataset, experiment, pred_len, seed):
                        continue
                    report = read_metrics(dataset, experiment, pred_len, seed)
                    metrics = report["metrics_original_units"]
                    row = {
                        "dataset": dataset,
                        "seed": seed,
                        "pred_len": pred_len,
                        "model": experiment,
                        "status": "completed",
                        "mse": metrics["mse"],
                        "mae": metrics["mae"],
                        "picp": "",
                        "mpiw": "",
                        "pinaw": "",
                        "ace": "",
                    }
                    if "interval" in metrics:
                        interval = metrics["interval"]
                        row.update(
                            picp=interval["picp"],
                            mpiw=interval["mpiw"],
                            pinaw=interval["pinaw"],
                            ace=interval["ace"],
                        )
                        channel_coverage = interval["coverage_by_channel"]
                        pinaw_by_channel = interval["pinaw_by_channel"]
                        for channel_index, (coverage, pinaw) in enumerate(
                            zip(channel_coverage, pinaw_by_channel)
                        ):
                            coverage_rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "pred_len": pred_len,
                                    "model": experiment,
                                    "channel": "OT" if channel_index == len(channel_coverage) - 1 else "channel{}".format(channel_index),
                                    "picp": coverage,
                                    "pinaw": pinaw,
                                }
                            )
                    all_runs.append(row)

    if not all_runs:
        raise RuntimeError("no completed full-experiment runs found")
    write_csv(
        RESULTS_ROOT / "all_runs.csv",
        all_runs[0].keys(),
        all_runs,
    )
    write_csv(
        RESULTS_ROOT / "coverage_by_channel.csv",
        coverage_rows[0].keys(),
        coverage_rows,
    )

    summary_rows = []
    point_fields = ("mse", "mae")
    interval_fields = ("picp", "mpiw", "pinaw", "ace")
    for dataset in DATASETS:
        for pred_len in PRED_LENS:
            for experiment in EXPERIMENTS:
                matching = [
                    row
                    for row in all_runs
                    if row["dataset"] == dataset
                    and row["pred_len"] == pred_len
                    and row["model"] == experiment
                ]
                if not matching:
                    continue
                summary = {
                    "dataset": dataset,
                    "pred_len": pred_len,
                    "model": experiment,
                    "num_seeds": len(matching),
                }
                for field in point_fields:
                    values = np.asarray([row[field] for row in matching], dtype=np.float64)
                    summary[field + "_mean"] = float(np.mean(values))
                    summary[field + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                if experiment != "E0":
                    for field in interval_fields:
                        values = np.asarray([row[field] for row in matching], dtype=np.float64)
                        summary[field + "_mean"] = float(np.mean(values))
                        summary[field + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                summary_rows.append(summary)
    write_csv(
        RESULTS_ROOT / "summary_mean_std.csv",
        (
            "dataset",
            "pred_len",
            "model",
            "num_seeds",
            "mse_mean",
            "mse_std",
            "mae_mean",
            "mae_std",
            "picp_mean",
            "picp_std",
            "mpiw_mean",
            "mpiw_std",
            "pinaw_mean",
            "pinaw_std",
            "ace_mean",
            "ace_std",
        ),
        summary_rows,
    )
    print("Saved {}".format(RESULTS_ROOT / "all_runs.csv"))
    print("Saved {}".format(RESULTS_ROOT / "summary_mean_std.csv"))
    print("Saved {}".format(RESULTS_ROOT / "coverage_by_channel.csv"))


if __name__ == "__main__":
    main()
