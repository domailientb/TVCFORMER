"""Resumable, single-process Clustering-TVCFormer suite for the five datasets."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "official"))

from FinalProject.Clustering_TVC import run_phase5 as phase5

DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
SEEDS = (2021, 2023, 2025)
HORIZONS = (96, 192, 336, 720)
RAW_CSV = PROJECT / "results" / "clustering_suite_runs_v2.csv"
CSV_FIELDS = phase5.CSV_FIELDS + ["validation_mse", "validation_mae"]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def read_rows(path):
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def valid_completed_row(row, log_path):
    """Only skip a run when the result row and its durable log evidence agree."""
    if row.get("status") != "completed" or not log_path.is_file():
        return False
    required_numeric = ("validation_mse", "validation_mae", "test_mse", "test_mae", "best_epoch", "training_time",
                        "test_samples", "total_time")
    try:
        values = [float(row[name]) for name in required_numeric]
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in values):
        return False
    if values[4] < 1 or values[6] < 1:
        return False
    marker = "HORIZON_COMPLETED pred_len={} seed={} test_samples={}".format(
        row.get("pred_len"), row.get("seed"), row.get("test_samples"))
    try:
        return marker in log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def append_raw(row):
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = RAW_CSV.is_file() and RAW_CSV.stat().st_size > 0
    if exists:
        with RAW_CSV.open("r", newline="", encoding="utf-8") as stream:
            if next(csv.reader(stream), []) != CSV_FIELDS:
                raise RuntimeError("existing suite CSV schema mismatch; refusing to append")
    with RAW_CSV.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def config_for(dataset):
    config_path = PROJECT / "configs" / (dataset + ".yaml")
    config = phase5._load_config(config_path)
    config["data"] = "weather" if dataset == "Weather" else dataset
    config["root_path"] = str((ROOT / config["root_path"]).resolve())
    config["config_yaml"] = str(config_path)
    if not Path(config["root_path"], config["data_path"]).is_file():
        raise FileNotFoundError(Path(config["root_path"], config["data_path"]))
    return config, config_path


def run_one(config, dataset, config_path, seed, horizon, log_path):
    run_id = "{}_p{}_clustering_s{}_{}".format(
        dataset, horizon, seed, datetime.now().strftime("%Y%m%dT%H%M%S%f"))
    row = {
        "dataset": dataset, "model": "clustering", "pred_len": horizon,
        "seed": seed, "status": "failed", "log_file": log_path.relative_to(ROOT).as_posix(),
        "config_file": config_path.relative_to(ROOT).as_posix(), "failure_reason": "",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        tee = Tee(sys.__stdout__, log)
        import contextlib
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print("\n---------- PREDICTION HORIZON: {} | RUN: {} ----------".format(horizon, run_id))
            print("Configuration source:", config_path)
            print("Configuration:", json.dumps(dict(config, pred_len=horizon,
                                                       random_seed=seed), sort_keys=True, default=str))
            try:
                result = phase5._train_and_evaluate(config, "clustering", horizon, seed, log)
                result["dataset"] = dataset
                result["model"] = "clustering"
                result["seed"] = seed
                result["status"] = "completed"
                result["log_file"] = row["log_file"]
                result["config_file"] = row["config_file"]
                append_raw(result)
                row.update(result)
                print("HORIZON_COMPLETED pred_len={} seed={} test_samples={}".format(
                    horizon, seed, result["test_samples"]))
                print("run_status=completed")
            except KeyboardInterrupt:
                row["status"] = "interrupted"
                row["failure_reason"] = "KeyboardInterrupt"
                print("HORIZON_INTERRUPTED pred_len={} seed={}".format(horizon, seed))
                append_raw(row)
                raise
            except Exception as error:
                row["failure_reason"] = "{}: {}".format(type(error).__name__, error)
                print("HORIZON_FAILED pred_len={} seed={} reason={}".format(
                    horizon, seed, row["failure_reason"]))
                traceback.print_exc()
                append_raw(row)
    return row


def write_summary(rows):
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    results_dir = PROJECT / "results"
    figures_dir = PROJECT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    completed = [row for row in rows if row.get("status") == "completed"]
    metrics = ("validation_mse", "validation_mae", "test_mse", "test_mae", "training_time", "validation_time",
               "eval_time", "total_time", "peak_gpu_memory")
    summary = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            group = [row for row in completed
                     if row.get("dataset") == dataset and int(row.get("pred_len", 0)) == horizon]
            record = {"dataset": dataset, "pred_len": horizon,
                      "completed_seeds": len({int(row["seed"]) for row in group}),
                      "expected_seeds": len(SEEDS),
                      "status": "complete" if len({int(row["seed"]) for row in group}) == len(SEEDS) else "partial"}
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in group
                                     if row.get(metric, "") not in (None, "")], dtype=float)
                record[metric + "_mean"] = float(values.mean()) if values.size else ""
                record[metric + "_std"] = float(values.std(ddof=1)) if values.size > 1 else (0.0 if values.size == 1 else "")
            summary.append(record)

    csv_path = results_dir / "clustering_suite_summary_{}.csv".format(timestamp)
    fields = list(summary[0]) if summary else ["dataset", "pred_len", "completed_seeds", "expected_seeds", "status"]
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    report_path = results_dir / "clustering_suite_summary_{}.md".format(timestamp)
    lines = ["# Clustering-TVCFormer experiment summary", "",
             "Source run table: `{}`.".format(RAW_CSV.relative_to(ROOT).as_posix()),
             "Only rows with `status=completed` are included in means; failed or missing runs are excluded.",
             "Standard deviation uses the sample definition (`ddof=1`); it is left blank with fewer than two values.", "",
             "| Dataset | Horizon | Seeds | Status | Validation MSE mean +/- std | Validation MAE mean +/- std | Test MSE mean +/- std | Test MAE mean +/- std | Total runtime mean +/- std (s) |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    def show(record, key):
        mean, std = record[key + "_mean"], record[key + "_std"]
        return "n/a" if mean == "" else "{:.6f} +/- {:.6f}".format(mean, std)
    for record in summary:
        lines.append("| {} | {} | {}/{} | {} | {} | {} | {} | {} | {} |".format(
            record["dataset"], record["pred_len"], record["completed_seeds"],
            record["expected_seeds"], record["status"], show(record, "validation_mse"),
            show(record, "validation_mae"), show(record, "test_mse"),
            show(record, "test_mae"), show(record, "total_time")))
    failures = [row for row in rows if row.get("status") != "completed"]
    lines += ["", "## Failed or incomplete rows", ""]
    if failures:
        lines += ["| Dataset | Seed | Horizon | Status | Reason | Log |", "|---|---:|---:|---|---|---|"]
        for row in failures:
            lines.append("| {} | {} | {} | {} | {} | `{}` |".format(
                row.get("dataset", ""), row.get("seed", ""), row.get("pred_len", ""),
                row.get("status", "failed"), str(row.get("failure_reason", "" )).replace("|", "\\|"),
                row.get("log_file", "")))
    else:
        lines.append("No failed rows recorded in this result table.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for metric, title, ylabel in (("test_mse", "Test MSE", "MSE"),
                                  ("test_mae", "Test MAE", "MAE"),
                                  ("total_time", "Total runtime", "Seconds")):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for dataset in DATASETS:
            group = [record for record in summary if record["dataset"] == dataset
                     and record[metric + "_mean"] != ""]
            if group:
                ax.errorbar([row["pred_len"] for row in group],
                            [row[metric + "_mean"] for row in group],
                            yerr=[row[metric + "_std"] for row in group],
                            marker="o", capsize=3, label=dataset)
        ax.set_title(title + " by dataset and horizon")
        ax.set_xlabel("Prediction horizon")
        ax.set_ylabel(ylabel)
        ax.set_xticks(HORIZONS)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "clustering_suite_{}_{}.png".format(metric, timestamp), dpi=160)
        plt.close(fig)
    return csv_path, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--pred_lens", nargs="+", type=int, default=list(HORIZONS))
    args = parser.parse_args()
    if set(args.pred_lens) - set(HORIZONS):
        parser.error("pred_lens must be chosen from 96, 192, 336, 720")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seed list contains duplicates")
    os_cwd = Path.cwd()
    import os
    os.chdir(ROOT)
    results_dir = PROJECT / "results"
    result_path = RAW_CSV
    config, config_path = config_for(args.dataset)
    rows = read_rows(result_path)
    prior_successes = {}
    for row in rows:
        if row.get("dataset") != args.dataset or row.get("model") != "clustering":
            continue
        log_path = ROOT / row.get("log_file", "")
        key = (int(row.get("seed", -1)), int(row.get("pred_len", -1)))
        if valid_completed_row(row, log_path):
            prior_successes[key] = row

    invocation_failures = 0
    for seed in args.seeds:
        log_path = PROJECT / "logs" / ("{}_seed{}.log".format(args.dataset, seed))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            tee = Tee(sys.__stdout__, log)
            import contextlib
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                print("\n========== DATASET: {} | SEED: {} | INVOCATION: {} ==========".format(
                    args.dataset, seed, datetime.now().isoformat(timespec="seconds")))
                print("Requested horizons:", args.pred_lens)
                print("Config:", config_path)
                for horizon in args.pred_lens:
                    key = (seed, horizon)
                    prior = prior_successes.get(key)
                    if prior and valid_completed_row(prior, ROOT / prior["log_file"]):
                        print("HORIZON_RESUME_SKIP pred_len={} seed={} validated_result_row=true".format(horizon, seed))
                        continue
                    row = run_one(config, args.dataset, config_path, seed, horizon, log_path)
                    rows.append(row)
                    if row.get("status") == "completed":
                        prior_successes[key] = row
                    else:
                        invocation_failures += 1
                completed = all((seed, h) in prior_successes for h in HORIZONS)
                if completed:
                    print("========== SEED {} COMPLETED | DATASET {} ==========".format(seed, args.dataset))
                else:
                    print("========== SEED {} INCOMPLETE | DATASET {} ==========".format(seed, args.dataset))

    csv_path, report_path = write_summary(read_rows(result_path))
    print("Raw result CSV:", result_path)
    print("Summary CSV:", csv_path)
    print("Summary report:", report_path)
    print("Invocation failed runs:", invocation_failures)
    os.chdir(os_cwd)
    return 1 if invocation_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
