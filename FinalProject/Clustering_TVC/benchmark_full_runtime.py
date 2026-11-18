"""Repeat a matched ETTh1 baseline vs optimized-Clustering runtime benchmark."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import yaml

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
RESULTS = PROJECT / "results"
RUNNER = PROJECT / "run_phase5.py"
CONFIG = PROJECT / "configs" / "phase5_etth1.yaml"
PRED_LEN = 96
SEED = 2021
MODEL_NAMES = ("baseline", "clustering")
CONFIG_KEYS = (
    "seq_len", "pred_len", "batch_size", "num_workers", "learning_rate",
    "train_epochs", "patience", "lradj", "pct_start", "weight_decay",
    "d_model", "n_heads", "e_layers", "d_layers", "d_ff", "patch_len",
    "stride", "subgraph_keep_rate", "n_clusters", "cross_edge_ratio",
    "candidate_multiplier", "use_amp", "scale", "target", "enc_in",
)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mean(rows, key):
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (TypeError, ValueError, KeyError):
            pass
    return statistics.mean(values) if values else None


def fmt(value, digits=3):
    return "—" if value is None else f"{value:,.{digits}f}"


def validate_config():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["dataset"] == "ETTh1"
    assert config["seq_len"] == 336 and config["batch_size"] == 128
    assert config["train_epochs"] == 100 and config["patience"] == 5
    assert config["learning_rate"] == 0.001 and config["lradj"] == "type3"
    assert config["decomposition"] is False
    return config


def run_one(repeat, model, run_csv_name, log_dir):
    command = [
        sys.executable, str(RUNNER),
        "--models", model,
        "--pred_lens", str(PRED_LEN),
        "--seed", str(SEED),
        "--config", str(CONFIG),
        "--result_csv", run_csv_name,
        "--log_dir", str(log_dir),
        "--skip_analysis",
    ]
    print(f"\n===== repeat={repeat} model={model} =====", flush=True)
    print("command:", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    run_csv = RESULTS / run_csv_name
    if not run_csv.is_file():
        return {
            "repeat": repeat, "dataset": "ETTh1", "seq_len": 336,
            "pred_len": PRED_LEN, "seed": SEED, "model": model,
            "status": "failed", "failure_reason": f"runner exit={completed.returncode}; no result CSV",
            "runner_exit_code": completed.returncode,
        }
    rows = read_csv(run_csv)
    if not rows:
        return {
            "repeat": repeat, "dataset": "ETTh1", "seq_len": 336,
            "pred_len": PRED_LEN, "seed": SEED, "model": model,
            "status": "failed", "failure_reason": f"runner exit={completed.returncode}; empty result CSV",
            "runner_exit_code": completed.returncode,
        }
    row = dict(rows[-1])
    row.update(repeat=repeat, seq_len=336, runner_exit_code=completed.returncode,
               raw_result_csv=run_csv_name)
    if completed.returncode != 0 and row.get("status") == "completed":
        row["status"] = "failed"
        row["failure_reason"] = f"runner exit={completed.returncode}"
    return row


def summarize(records, session_dir, session_id, config):
    raw_csv = session_dir / f"runtime_benchmark_{session_id}_runs.csv"
    fieldnames = list(dict.fromkeys(key for row in records for key in row))
    with raw_csv.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    completed = {model: [r for r in records if r.get("model") == model and r.get("status") == "completed"]
                 for model in MODEL_NAMES}
    summary_rows = []
    time_fields = (
        "train_loop_time", "validation_time", "eval_time", "training_time",
        "total_time", "precompute_time", "pearson_time", "clustering_time", "graph_time",
    )
    for model in MODEL_NAMES:
        group = completed[model]
        summary = {
            "dataset": "ETTh1", "seq_len": 336, "pred_len": PRED_LEN,
            "seed": SEED, "model": model, "completed_runs": len(group),
            "requested_runs": config["repeats"], "status": "completed" if group else "failed",
            "test_mse_mean": mean(group, "test_mse"),
            "test_mae_mean": mean(group, "test_mae"),
            "best_epoch_mean": mean(group, "best_epoch"),
        }
        for field in time_fields:
            summary[field + "_mean"] = mean(group, field)
            values = [float(r[field]) for r in group if r.get(field) not in (None, "")]
            summary[field + "_stdev"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
        summary_rows.append(summary)

    baseline = next((r for r in summary_rows if r["model"] == "baseline"), None)
    clustering = next((r for r in summary_rows if r["model"] == "clustering"), None)
    comparison = {"dataset": "ETTh1", "seq_len": 336, "pred_len": PRED_LEN, "seed": SEED}
    for field in time_fields:
        base_value = baseline.get(field + "_mean") if baseline else None
        cluster_value = clustering.get(field + "_mean") if clustering else None
        comparison[field + "_difference_pct_vs_baseline"] = (
            100.0 * (cluster_value - base_value) / base_value
            if base_value not in (None, 0) and cluster_value is not None else None
        )
    summary_csv = session_dir / f"runtime_benchmark_{session_id}_summary.csv"
    with summary_csv.open("x", newline="", encoding="utf-8") as stream:
        summary_fields = list(dict.fromkeys(key for row in summary_rows + [comparison] for key in row))
        writer = csv.DictWriter(stream, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
        writer.writerow(comparison)

    report = [
        "# ETTh1 whole-model runtime benchmark",
        "",
        "## Protocol",
        "",
        f"- Dataset ETTh1, `seq_len=336`, `pred_len={PRED_LEN}`, seed `{SEED}`; GPU/hardware is recorded in every run log.",
        f"- Config source: `{CONFIG.relative_to(ROOT).as_posix()}`. Batch size `{config['batch_size']}`, max epochs `{config['train_epochs']}`, patience `{config['patience']}`, learning rate `{config['learning_rate']}`, optimizer/scheduler and all remaining hyperparameters are shared.",
        "- Baseline is imported from the unmodified `official/` implementation. Clustering uses the optimized indexed-cache and vectorized NumPy selector. Runs are sequential; order alternates by repeat to reduce systematic warm-up/thermal bias.",
        "- Train-loop, validation-loop, and full-test timers use `time.perf_counter()` with CUDA synchronization at each boundary. Total time begins before model/data setup and ends after full test metrics; it includes clustering precompute for the clustering model.",
        "- Total time is measured once as an end-to-end interval. Train, validation, test, precompute, Pearson, clustering, and Graph Generator timers are nested/component diagnostics and are not added to total time.",
        "- MSE and MAE use the entire test set with the final partial batch included. Best epoch is selected from validation loss; early stopping uses the same patience.",
        "- Earlier phase-5 logs used a combined train+validation timer and second-resolution log wall-time reconstruction; those values are not mixed into these repeated-run means.",
        "",
        "## Individual runs",
        "",
        "| Repeat | Model | Status | Test MSE | Test MAE | Best epoch | Train (s) | Validation (s) | Test (s) | Total (s) | Precompute (s) | Log | Failure reason |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in records:
        report.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | `{}` | {} |".format(
            row.get("repeat", "—"), row.get("model", "—"), row.get("status", "failed"),
            fmt(float(row["test_mse"]) if row.get("test_mse") else None, 7),
            fmt(float(row["test_mae"]) if row.get("test_mae") else None, 7),
            row.get("best_epoch", "—") or "—", fmt(float(row["train_loop_time"]) if row.get("train_loop_time") else None),
            fmt(float(row["validation_time"]) if row.get("validation_time") else None),
            fmt(float(row["eval_time"]) if row.get("eval_time") else None),
            fmt(float(row["total_time"]) if row.get("total_time") else None),
            fmt(float(row["precompute_time"]) if row.get("precompute_time") else None),
            row.get("log_file", "—"), row.get("failure_reason", "") or "—"))

    report.extend(["", "## Mean results and runtime differences", ""])
    report.append("| Model | Completed runs | MSE mean | MAE mean | Best epoch mean | Train mean ± SD (s) | Validation mean ± SD (s) | Test mean ± SD (s) | Total mean ± SD (s) | Precompute mean (s) |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        report.append("| {} | {}/{} | {} | {} | {} | {} ± {} | {} ± {} | {} ± {} | {} ± {} | {} |".format(
            row["model"], row["completed_runs"], row["requested_runs"],
            fmt(row["test_mse_mean"], 7), fmt(row["test_mae_mean"], 7), fmt(row["best_epoch_mean"], 2),
            fmt(row.get("train_loop_time_mean")), fmt(row.get("train_loop_time_stdev")),
            fmt(row.get("validation_time_mean")), fmt(row.get("validation_time_stdev")),
            fmt(row.get("eval_time_mean")), fmt(row.get("eval_time_stdev")),
            fmt(row.get("total_time_mean")), fmt(row.get("total_time_stdev")),
            fmt(row.get("precompute_time_mean"))))
    report.extend(["", "| Time measure | Clustering minus baseline (%) |", "|---|---:|"])
    labels = {
        "train_loop_time": "Training loop", "validation_time": "Validation",
        "eval_time": "Full test evaluation", "training_time": "Train + validation combined",
        "total_time": "End-to-end total", "precompute_time": "Clustering precompute",
    }
    for field, label in labels.items():
        report.append(f"| {label} | {fmt(comparison[field + '_difference_pct_vs_baseline'], 2)} |")
    report.extend([
        "",
        "Negative percentages mean Clustering-TVCFormer took less time; positive percentages mean it took longer. A mean is shown only for completed runs. No missing/failed run is assigned a substitute value.",
        "",
        "## Artifacts and status",
        "",
        f"- Individual-run CSV: `{raw_csv.relative_to(ROOT).as_posix()}`",
        f"- Mean/percent-comparison CSV: `{summary_csv.relative_to(ROOT).as_posix()}`",
        f"- Per-run logs: `{(session_dir / 'logs').relative_to(ROOT).as_posix()}/`",
        f"- Runs requested: `{config['repeats'] * 2}`; completed: `{sum(r.get('status') == 'completed' for r in records)}`.",
        "",
    ])
    report_path = session_dir / f"runtime_benchmark_{session_id}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return raw_csv, summary_csv, report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be >= 1")
    base_config = validate_config()
    session_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    session_dir = RESULTS / f"full_runtime_benchmark_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=False)
    log_dir = session_dir / "logs"
    log_dir.mkdir()
    session_config = {
        "dataset": "ETTh1", "seq_len": 336, "pred_len": PRED_LEN, "seed": SEED,
        "repeats": args.repeats, "models": MODEL_NAMES,
        "source_config": CONFIG.relative_to(ROOT).as_posix(),
        "hardware_preflight": "pending", "status": "running",
    }
    (session_dir / "session.json").write_text(json.dumps(session_config, indent=2), encoding="utf-8")

    # Fail closed if another workload is using the single 4 GB GPU. The caller
    # can rerun once the device is idle; this prevents unfair timing contention.
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        gpu_rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    except Exception as error:
        session_config.update(status="not_started", hardware_preflight=f"failed: {error}")
        (session_dir / "session.json").write_text(json.dumps(session_config, indent=2), encoding="utf-8")
        raise RuntimeError(f"GPU preflight failed; benchmark was not started: {error}")
    session_config["hardware_preflight"] = gpu_rows
    (session_dir / "session.json").write_text(json.dumps(session_config, indent=2), encoding="utf-8")
    busy = []
    for line in gpu_rows:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4:
            try:
                used, utilization = int(fields[1]), int(fields[3])
                if used > 256 or utilization > 5:
                    busy.append(line)
            except ValueError:
                busy.append(line)
    if busy:
        session_config.update(status="not_started_gpu_busy", gpu_busy_rows=busy)
        (session_dir / "session.json").write_text(json.dumps(session_config, indent=2), encoding="utf-8")
        report = session_dir / f"runtime_benchmark_{session_id}.md"
        report.write_text(
            "# ETTh1 whole-model runtime benchmark\n\n"
            "Status: not started because GPU preflight found another workload.\n\n"
            f"GPU query: `{gpu_rows}`\n\n"
            "Run this script again after the GPU is idle. No model run or metric was fabricated.\n",
            encoding="utf-8",
        )
        print("GPU is busy; benchmark not started. Preflight report:", report, flush=True)
        return

    print("Benchmark preflight passed:", gpu_rows, flush=True)
    records = []
    for repeat in range(1, args.repeats + 1):
        order = MODEL_NAMES if repeat % 2 else tuple(reversed(MODEL_NAMES))
        for model in order:
            result_name = f"runtime_benchmark_{session_id}_r{repeat}_{model}.csv"
            row = run_one(repeat, model, result_name, log_dir)
            records.append(row)
            if row.get("status") != "completed":
                print("run failed:", row.get("failure_reason"), flush=True)
            # subprocess exit releases CUDA allocations before the next run.
    raw_csv, summary_csv, report_path = summarize(records, session_dir, session_id,
                                                   dict(base_config, repeats=args.repeats))
    session_config.update(status="completed", run_count=len(records),
                          raw_csv=str(raw_csv.relative_to(ROOT)),
                          summary_csv=str(summary_csv.relative_to(ROOT)),
                          report=str(report_path.relative_to(ROOT)))
    (session_dir / "session.json").write_text(json.dumps(session_config, indent=2), encoding="utf-8")
    print("raw CSV:", raw_csv)
    print("summary CSV:", summary_csv)
    print("report:", report_path)


if __name__ == "__main__":
    main()
