"""Compare full Clustering-TVCFormer runs with sequential vs batched precompute."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys

import yaml

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
RESULTS = PROJECT / "results"
RUNNER = PROJECT / "run_phase5.py"
SOURCE_CONFIG = PROJECT / "configs" / "phase5_etth1.yaml"
VARIANTS = ("sequential", "batched")
METRICS = ("test_mse", "test_mae", "best_epoch", "training_time", "train_loop_time",
           "validation_time", "eval_time", "total_time", "precompute_time",
           "clustering_time", "pearson_time", "graph_time", "peak_gpu_memory")


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run_one(session_id, directory, repeat, variant, config):
    run_config = dict(config, precompute_implementation=variant)
    config_path = directory / "configs" / f"ETTh1_p96_{variant}_r{repeat}.yaml"
    config_path.write_text(yaml.safe_dump(run_config, sort_keys=True), encoding="utf-8")
    result_name = f"precompute_e2e_{session_id}_r{repeat}_{variant}.csv"
    log_dir = directory / "logs"
    command = [sys.executable, str(RUNNER), "--models", "clustering", "--pred_lens", "96",
               "--seed", "2021", "--config", str(config_path), "--result_csv", result_name,
               "--log_dir", str(log_dir), "--skip_analysis"]
    print(f"\n===== repeat={repeat} precompute={variant} =====", flush=True)
    print("command:", subprocess.list2cmdline(command), flush=True)
    process = subprocess.run(command, cwd=ROOT, check=False)
    result_path = RESULTS / result_name
    if not result_path.is_file() or not read_rows(result_path):
        return {"repeat": repeat, "precompute_implementation": variant, "status": "failed",
                "failure_reason": f"runner exit={process.returncode}; missing/empty CSV"}
    row = read_rows(result_path)[-1]
    row.update(repeat=repeat, precompute_implementation=variant,
               runner_exit_code=process.returncode, raw_result_csv=result_name,
               benchmark_config=str(config_path.relative_to(ROOT)))
    if process.returncode and row.get("status") == "completed":
        row.update(status="failed", failure_reason=f"runner exit={process.returncode}")
    return row


def write_report(records, directory, session_id, repeats, config, gpu_query):
    raw_path = directory / f"precompute_e2e_{session_id}_runs.csv"
    fieldnames = list(dict.fromkeys(key for record in records for key in record))
    with raw_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    summaries = []
    for variant in VARIANTS:
        group = [r for r in records if r.get("precompute_implementation") == variant
                 and r.get("status") == "completed"]
        row = {"precompute_implementation": variant, "completed_runs": len(group),
               "requested_runs": repeats, "status": "completed" if group else "failed"}
        for key in METRICS:
            values = [float(r[key]) for r in group if r.get(key) not in (None, "")]
            row[key + "_mean"] = statistics.mean(values) if values else None
            row[key + "_stdev"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
        summaries.append(row)
    old, new = summaries
    comparison = {}
    for key in METRICS:
        old_value, new_value = old[key + "_mean"], new[key + "_mean"]
        comparison[key + "_difference_pct"] = (
            100.0 * (new_value - old_value) / old_value
            if old_value not in (None, 0) and new_value is not None else None)
    summary_path = directory / f"precompute_e2e_{session_id}_summary.csv"
    with summary_path.open("x", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(k for row in summaries + [comparison] for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries + [comparison])

    lines = [
        "# Full-model ETTh1 benchmark: sequential vs batched precompute", "",
        f"- ETTh1, seq_len=336, pred_len=96, seed=2021, batch size {config['batch_size']}, GTX 1650.",
        "- The only variant is indexed precompute grouping (`sequential` vs `batched`). Both use batched edge assembly and the same model, data splits, scaler, optimizer, schedule, epochs and early stopping.",
        "- CUDA synchronization is used at runner timer boundaries. End-to-end time includes setup, precompute, training, validation and full test; component timers are not added to it.",
        f"- GPU preflight: `{gpu_query}`", "",
        "| Precompute | Runs | Test MSE | Test MAE | Best epoch | Train+validation (s) | Precompute (s) | Test (s) | Total (s) | Peak GPU (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def fmt(value, digits=4):
            return "n/a" if value is None else f"{value:.{digits}f}"
        lines.append("| {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["precompute_implementation"], row["completed_runs"], row["requested_runs"],
            fmt(row["test_mse_mean"], 7), fmt(row["test_mae_mean"], 7), fmt(row["best_epoch_mean"], 2),
            fmt(row["training_time_mean"]), fmt(row["precompute_time_mean"]),
            fmt(row["eval_time_mean"]), fmt(row["total_time_mean"]), fmt(row["peak_gpu_memory_mean"], 2)))
    lines.extend(["", "| Timer | Batched minus sequential (%) |", "|---|---:|"])
    for key in METRICS:
        if comparison[key + "_difference_pct"] is not None:
            lines.append(f"| {key} | {comparison[key + '_difference_pct']:.2f} |")
    lines.extend(["", "## Individual runs", ""])
    for row in records:
        lines.append("- repeat {} `{}`: {}, total={} s, precompute={} s, MSE={}, MAE={}, log=`{}`{}".format(
            row.get("repeat"), row.get("precompute_implementation"), row.get("status"),
            row.get("total_time", "n/a"), row.get("precompute_time", "n/a"),
            row.get("test_mse", "n/a"), row.get("test_mae", "n/a"), row.get("log_file", "n/a"),
            ("; failure=" + str(row["failure_reason"])) if row.get("failure_reason") else ""))
    lines.extend(["", f"Raw CSV: `{raw_path.relative_to(ROOT).as_posix()}`",
                  f"Summary CSV: `{summary_path.relative_to(ROOT).as_posix()}`",
                  f"Logs: `{(directory / 'logs').relative_to(ROOT).as_posix()}/`", ""])
    report_path = directory / f"precompute_e2e_{session_id}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return raw_path, summary_path, report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    if config.get("dataset") != "ETTh1" or config.get("seq_len") != 336:
        raise ValueError("source config does not match ETTh1 benchmark settings")
    session_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    directory = RESULTS / f"precompute_e2e_benchmark_{session_id}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "configs").mkdir()
    (directory / "logs").mkdir()
    metadata = {"status": "running", "session_id": session_id, "repeats": args.repeats,
                "dataset": "ETTh1", "seq_len": 336, "pred_len": 96, "seed": 2021}
    meta_path = directory / "session.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    gpu_rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    busy = []
    for line in gpu_rows:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4 and (int(parts[1]) > 256 or int(parts[3]) > 5):
            busy.append(line)
    if busy:
        metadata.update(status="not_started_gpu_busy", gpu_preflight=gpu_rows)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        report_path = directory / f"precompute_e2e_{session_id}.md"
        report_path.write_text(
            "# Full-model ETTh1 precompute benchmark\n\n"
            "Not started: GPU preflight found another workload. No model run or metric was fabricated.\n\n"
            f"GPU preflight: `{gpu_rows}`\n",
            encoding="utf-8")
        print("GPU busy; no training started:", gpu_rows)
        print("preflight report:", report_path)
        return

    records = []
    for repeat in range(1, args.repeats + 1):
        order = VARIANTS if repeat % 2 else tuple(reversed(VARIANTS))
        for variant in order:
            records.append(run_one(session_id, directory, repeat, variant, config))
    raw, summary, report = write_report(records, directory, session_id, args.repeats,
                                        config, gpu_rows)
    metadata.update(status="completed", raw_csv=str(raw.relative_to(ROOT)),
                    summary_csv=str(summary.relative_to(ROOT)), report=str(report.relative_to(ROOT)),
                    run_count=len(records), gpu_preflight=gpu_rows)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("raw CSV:", raw)
    print("summary CSV:", summary)
    print("report:", report)


if __name__ == "__main__":
    main()
