"""Matched full-run benchmark of sample-wise versus batched COO assembly."""

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
VARIANTS = (("samplewise_reference", False), ("batched", True))
MEASURES = (
    "train_loop_time", "validation_time", "eval_time", "training_time",
    "total_time", "precompute_time", "clustering_time", "pearson_time", "graph_time",
)


def load_rows(path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric_mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return statistics.mean(values) if values else None


def number(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def run_one(session_id, session_dir, repeat, variant, batch_assembly, source_config):
    config = dict(source_config)
    config["batch_edge_assembly"] = batch_assembly
    config_path = session_dir / "configs" / f"ETTh1_p96_{variant}_r{repeat}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    result_name = f"edge_assembly_{session_id}_r{repeat}_{variant}.csv"
    log_dir = session_dir / "logs"
    command = [
        sys.executable, str(RUNNER), "--models", "clustering", "--pred_lens", "96",
        "--seed", "2021", "--config", str(config_path), "--result_csv", result_name,
        "--log_dir", str(log_dir), "--skip_analysis",
    ]
    print(f"\n===== repeat={repeat} variant={variant} =====", flush=True)
    print("command:", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    result_path = RESULTS / result_name
    if not result_path.is_file() or not load_rows(result_path):
        return {"repeat": repeat, "variant": variant, "status": "failed",
                "failure_reason": f"runner exit={completed.returncode}; missing/empty result CSV"}
    row = load_rows(result_path)[-1]
    row.update(repeat=repeat, variant=variant, runner_exit_code=completed.returncode,
               raw_result_csv=result_name, variant_config=str(config_path.relative_to(ROOT)))
    if completed.returncode and row.get("status") == "completed":
        row.update(status="failed", failure_reason=f"runner exit={completed.returncode}")
    return row


def summarize(records, session_dir, session_id, repeats, preoptimization_reference):
    raw_path = session_dir / f"edge_assembly_{session_id}_runs.csv"
    fields = list(dict.fromkeys(key for row in records for key in row))
    with raw_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    summaries = []
    groups = {}
    for variant, _ in VARIANTS:
        group = [r for r in records if r.get("variant") == variant and r.get("status") == "completed"]
        groups[variant] = group
        row = {"dataset": "ETTh1", "seq_len": 336, "pred_len": 96, "seed": 2021,
               "variant": variant, "requested_runs": repeats, "completed_runs": len(group),
               "status": "completed" if group else "failed"}
        for metric in ("test_mse", "test_mae", "best_epoch", "peak_gpu_memory") + MEASURES:
            vals = [float(r[metric]) for r in group if r.get(metric) not in (None, "")]
            row[metric + "_mean"] = statistics.mean(vals) if vals else None
            row[metric + "_stdev"] = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None)
        for metric in ("pearson_pairs", "cache_hits", "cache_misses", "kmeans_calls"):
            vals = [int(float(r[metric])) for r in group if r.get(metric) not in (None, "")]
            row[metric + "_mean"] = statistics.mean(vals) if vals else None
        summaries.append(row)

    reference, optimized = summaries
    comparison = {"dataset": "ETTh1", "seq_len": 336, "pred_len": 96, "seed": 2021,
                  "comparison": "batched_minus_samplewise_pct"}
    for metric in MEASURES:
        old, new = reference.get(metric + "_mean"), optimized.get(metric + "_mean")
        comparison[metric + "_difference_pct"] = (
            100 * (new - old) / old if old not in (None, 0) and new is not None else None)
    old_total, new_total = reference.get("total_time_mean"), optimized.get("total_time_mean")
    comparison["total_time_reduction_pct"] = (
        100 * (old_total - new_total) / old_total
        if old_total not in (None, 0) and new_total is not None else None)
    summary_path = session_dir / f"edge_assembly_{session_id}_summary.csv"
    with summary_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=list(dict.fromkeys(k for r in summaries + [comparison] for k in r)),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries + [comparison])

    report = [
        "# Clustering-TVCFormer edge assembly optimization benchmark", "",
        "## Protocol", "",
        "- ETTh1, seq_len=336, pred_len=96, seed=2021, GTX 1650; three sequential repeats per variant.",
        "- The sole variant change is graph COO assembly: retained sample-wise reference path versus batched CPU COO assembly and one tensor transfer. The source training config is otherwise copied unchanged.",
        "- Training, validation and test timers synchronize CUDA. End-to-end time is measured once and includes clustering precompute; component timers are not added to it.",
        "- Test metrics cover all 2,785 ETTh1 test windows. Early stopping, optimizer, schedule, batch size and epoch limit remain fixed.", "",
        "## Aggregate results", "",
        "| Variant | Runs | MSE | MAE | Best epoch | Train loop (s) | Validation (s) | Test (s) | Total (s) | Precompute (s) | Peak GPU (MiB) | Pearson pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report.append("| {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["variant"], row["completed_runs"], row["requested_runs"],
            number(row["test_mse_mean"], 7), number(row["test_mae_mean"], 7),
            number(row["best_epoch_mean"], 2), number(row["train_loop_time_mean"]),
            number(row["validation_time_mean"]), number(row["eval_time_mean"]),
            number(row["total_time_mean"]), number(row["precompute_time_mean"]),
            number(row["peak_gpu_memory_mean"], 2), number(row["pearson_pairs_mean"], 0)))
    report.extend(["", "## Runtime change", "",
                   "| Timer | Batched minus sample-wise (%) |", "|---|---:|"])
    for metric in MEASURES:
        report.append(f"| {metric} | {number(comparison[metric + '_difference_pct'], 2)} |")
    report.extend(["", "## Run outcomes and limitations", ""])
    for row in records:
        report.append("- Repeat {} `{}`: **{}**, MSE={}, MAE={}, best_epoch={}, total={}s, log=`{}`{}".format(
            row.get("repeat"), row.get("variant"), row.get("status"),
            row.get("test_mse", "n/a"), row.get("test_mae", "n/a"),
            row.get("best_epoch", "n/a"), row.get("total_time", "n/a"),
            row.get("log_file", "n/a"),
            ("; reason=" + str(row.get("failure_reason"))) if row.get("failure_reason") else ""))
    report.extend(["", "No missing run or metric is imputed. The single fixed seed and one dataset limit generalization. A runtime reduction is claimed only if the measured end-to-end total decreases; operator-only improvement is insufficient.", "",
                   f"Prior optimized selector reference: MSE={preoptimization_reference['test_mse_mean']}, MAE={preoptimization_reference['test_mae_mean']}, total={preoptimization_reference['total_time_mean']} s (three-run ETTh1 benchmark).",
                   f"This benchmark session: `{session_dir.relative_to(ROOT).as_posix()}`.",
                   f"Raw rows: `{raw_path.relative_to(ROOT).as_posix()}`; summary: `{summary_path.relative_to(ROOT).as_posix()}`; logs: `{(session_dir / 'logs').relative_to(ROOT).as_posix()}/`.", ""])
    report_path = session_dir / f"edge_assembly_{session_id}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return raw_path, summary_path, report_path, summaries, comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    if (config.get("dataset") != "ETTh1" or config.get("seq_len") != 336 or
            config.get("batch_size") != 128 or config.get("train_epochs") != 100 or
            config.get("patience") != 5):
        raise ValueError("phase5 configuration differs from the validated benchmark protocol")
    previous_path = RESULTS / "full_runtime_benchmark_20261118T070657043363" / "runtime_benchmark_20261118T070657043363_summary.csv"
    previous_rows = load_rows(previous_path)
    previous = next(row for row in previous_rows if row.get("model") == "clustering")
    session_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    session_dir = RESULTS / f"edge_assembly_benchmark_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "configs").mkdir()
    (session_dir / "logs").mkdir()
    metadata = {"status": "running", "session_id": session_id, "repeats": args.repeats,
                "dataset": "ETTh1", "seq_len": 336, "pred_len": 96, "seed": 2021,
                "source_config": str(SOURCE_CONFIG.relative_to(ROOT)),
                "variants": dict(VARIANTS)}
    meta_path = session_dir / "session.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    query = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    gpu_rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    metadata["gpu_preflight"] = gpu_rows
    busy = []
    for line in gpu_rows:
        parts = [value.strip() for value in line.split(",")]
        if len(parts) >= 4 and (int(parts[1]) > 256 or int(parts[3]) > 5):
            busy.append(line)
    if busy:
        metadata.update(status="not_started_gpu_busy", gpu_busy=busy)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("GPU busy; benchmark not run:", busy)
        return

    records = []
    for repeat in range(1, args.repeats + 1):
        order = VARIANTS if repeat % 2 else tuple(reversed(VARIANTS))
        for variant, batch_assembly in order:
            row = run_one(session_id, session_dir, repeat, variant, batch_assembly, config)
            records.append(row)
            if row.get("status") != "completed":
                print("FAILED:", row.get("variant"), row.get("failure_reason"), flush=True)
    raw, summary, report, aggregates, comparison = summarize(records, session_dir, session_id,
                                                               args.repeats, previous)
    metadata.update(status="completed", run_count=len(records), raw_csv=str(raw.relative_to(ROOT)),
                    summary_csv=str(summary.relative_to(ROOT)), report=str(report.relative_to(ROOT)))
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("raw CSV:", raw)
    print("summary CSV:", summary)
    print("report:", report)


if __name__ == "__main__":
    main()
