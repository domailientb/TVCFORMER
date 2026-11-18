"""Combine separately stored sequential/batched full-run sessions."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import statistics

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
METRICS = ("test_mse", "test_mae", "best_epoch", "training_time", "train_loop_time",
           "validation_time", "eval_time", "total_time", "precompute_time",
           "clustering_time", "pearson_time", "graph_time", "peak_gpu_memory")
VARIANTS = ("sequential", "batched")


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="Session directories containing precompute_e2e_*_runs.csv")
    args = parser.parse_args()
    records = []
    for session in args.sessions:
        directory = Path(session)
        if not directory.is_absolute():
            directory = ROOT / directory
        matches = list(directory.glob("precompute_e2e_*_runs.csv"))
        if len(matches) != 1:
            raise ValueError(f"expected one runs CSV in {directory}, found {len(matches)}")
        for row in read_csv(matches[0]):
            row["source_session"] = str(directory.relative_to(ROOT).as_posix())
            records.append(row)
    if any(row.get("status") != "completed" for row in records):
        raise RuntimeError("cannot summarize sessions with failed/incomplete runs")
    groups = {name: [r for r in records if r.get("precompute_implementation") == name]
              for name in VARIANTS}
    if any(len(group) != 3 for group in groups.values()):
        raise ValueError("expected exactly three completed runs for each implementation")

    session_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    out_dir = PROJECT / "results" / f"precompute_e2e_aggregate_{session_id}"
    out_dir.mkdir(parents=True, exist_ok=False)
    raw_path = out_dir / f"precompute_e2e_{session_id}_runs.csv"
    fields = list(dict.fromkeys(k for row in records for k in row))
    with raw_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    summaries = []
    for variant in VARIANTS:
        group = groups[variant]
        summary = {"dataset": "ETTh1", "seq_len": 336, "pred_len": 96, "seed": 2021,
                   "precompute_implementation": variant, "completed_runs": len(group)}
        for metric in METRICS:
            values = [float(r[metric]) for r in group]
            summary[metric + "_mean"] = statistics.mean(values)
            summary[metric + "_median"] = statistics.median(values)
            summary[metric + "_stdev"] = statistics.stdev(values)
        summaries.append(summary)
    comparison = {"comparison": "batched_minus_sequential_pct"}
    for metric in METRICS:
        old = summaries[0][metric + "_mean"]
        new = summaries[1][metric + "_mean"]
        comparison[metric + "_difference_pct"] = 100 * (new - old) / old if old else None
    summary_path = out_dir / f"precompute_e2e_{session_id}_summary.csv"
    with summary_path.open("x", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(k for row in summaries + [comparison] for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries + [comparison])

    lines = ["# ETTh1 full-run precompute comparison (three runs per variant)", "",
             "- All runs: seq_len=336, pred_len=96, seed=2021, batch size 128, GTX 1650; full test set (2,785 windows).",
             "- The variant changes only sequential vs batched history precompute. Each run measures end-to-end independently; component timers are not added to total.", "",
             "| Variant | Runs | Test MSE | Test MAE | Best epoch | Train+validation (s) | Precompute (s) | Test (s) | Total (s) | Peak memory (MiB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append("| {} | {} | {:.7f} ± {:.7f} | {:.7f} ± {:.7f} | {:.2f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} |".format(
            row["precompute_implementation"], row["completed_runs"],
            row["test_mse_mean"], row["test_mse_stdev"], row["test_mae_mean"], row["test_mae_stdev"],
            row["best_epoch_mean"], row["training_time_mean"], row["training_time_stdev"],
            row["precompute_time_mean"], row["precompute_time_stdev"],
            row["eval_time_mean"], row["eval_time_stdev"], row["total_time_mean"],
            row["total_time_stdev"], row["peak_gpu_memory_mean"]))
    lines.extend(["", "| Timer | Batched minus sequential (%) |", "|---|---:|"])
    for metric in METRICS:
        lines.append(f"| {metric} | {comparison[metric + '_difference_pct']:.2f} |")
    lines.extend(["", "## Runs", ""])
    for row in records:
        lines.append("- {} repeat {}: total={}s, precompute={}s, MSE={}, MAE={}, log=`{}`".format(
            row["precompute_implementation"], row["repeat"], row["total_time"],
            row["precompute_time"], row["test_mse"], row["test_mae"], row["log_file"]))
    lines.extend(["", f"Raw data: `{raw_path.relative_to(ROOT).as_posix()}`",
                  f"Summary: `{summary_path.relative_to(ROOT).as_posix()}`", ""])
    report_path = out_dir / f"precompute_e2e_{session_id}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)
    print(summary_path)


if __name__ == "__main__":
    main()
