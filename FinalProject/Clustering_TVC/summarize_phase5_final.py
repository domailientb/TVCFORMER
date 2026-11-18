"""Build the final three-way ETTh1 comparison from recorded runs only."""

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "FinalProject" / "Clustering_TVC"
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"
HISTORIC_CSV = RESULTS / "phase5_etth1_results.csv"
OPTIMIZED_CSV = RESULTS / "phase5_etth1_optimized_results.csv"
OUTPUT_CSV = RESULTS / "phase5_etth1_final_comparison.csv"
REPORT = RESULTS / "phase5_etth1_final_comparison.md"
HORIZONS = (96, 192, 336, 720)
SEED = 2021
CONFIG_KEYS = (
    "seq_len", "batch_size", "learning_rate", "train_epochs", "patience",
    "lradj", "pct_start", "weight_decay", "d_model", "n_heads", "e_layers",
    "d_layers", "d_ff", "patch_len", "stride", "subgraph_keep_rate",
    "n_clusters", "cross_edge_ratio", "candidate_multiplier",
)


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def find_completed(rows, model, horizon):
    matches = [row for row in rows if row.get("dataset") == "ETTh1"
               and row.get("model") == model
               and row.get("pred_len") == str(horizon)
               and row.get("seed") == str(SEED)
               and row.get("status") == "completed"]
    return matches[-1] if matches else None


def parse_log_wall(log_ref):
    path = ROOT / Path(log_ref.replace("\\", "/"))
    if not path.is_file():
        return ""
    start = end = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("start_time:"):
            start = datetime.fromisoformat(line.split(":", 1)[1].strip())
        elif line.startswith("end_time:"):
            end = datetime.fromisoformat(line.split(":", 1)[1].strip())
    return max(0.0, (end - start).total_seconds()) if start and end else ""


def number(row, key):
    value = row.get(key, "") if row else ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def make_record(group, source_row, horizon, status, baseline_pairs=None, optimized=False):
    if not source_row:
        return {
            "dataset": "ETTh1", "model_group": group, "pred_len": horizon,
            "seed": SEED, "status": status,
        }
    row = {
        "dataset": "ETTh1", "model_group": group, "pred_len": horizon,
        "seed": SEED, "status": source_row.get("status", status),
        "test_mse": source_row.get("test_mse", ""),
        "test_mae": source_row.get("test_mae", ""),
        "best_epoch": source_row.get("best_epoch", ""),
        "early_stopped": source_row.get("early_stopped", ""),
        "train_epochs_completed": source_row.get("train_epochs_completed", ""),
        "training_time_s": source_row.get("training_time", ""),
        "epoch_time_s": source_row.get("epoch_time", ""),
        "eval_time_s": source_row.get("eval_time", ""),
        "precompute_time_s": source_row.get("precompute_time", ""),
        "total_time_s": source_row.get("total_time", ""),
        "feature_time_s": source_row.get("feature_time", ""),
        "kmeans_time_s": source_row.get("kmeans_time", ""),
        "candidate_time_s": source_row.get("candidate_time", ""),
        "cache_lookup_time_s": source_row.get("cache_lookup_time", ""),
        "transfer_time_s": source_row.get("transfer_time", ""),
        "pearson_time_s": source_row.get("pearson_time", ""),
        "clustering_time_s": source_row.get("clustering_time", ""),
        "graph_time_s": source_row.get("graph_time", ""),
        "pearson_pairs": source_row.get("pearson_pairs", ""),
        "baseline_pearson_pairs": baseline_pairs or "",
        "pearson_reduction_pct": source_row.get("pearson_reduction_pct", ""),
        "kmeans_calls": source_row.get("kmeans_calls", ""),
        "cache_hits": source_row.get("cache_hits", ""),
        "cache_misses": source_row.get("cache_misses", ""),
        "peak_gpu_memory_mib": source_row.get("peak_gpu_memory", ""),
        "test_samples": source_row.get("test_samples", ""),
        "log_file": source_row.get("log_file", ""),
        "config_file": source_row.get("config_file", ""),
    }
    if not optimized:
        row["total_time_s"] = parse_log_wall(source_row.get("log_file", ""))
    return row


def fmt(value, digits=4):
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def write_chart(rows_by_key, field, title, ylabel, filename):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    groups = ("baseline", "pre_optimization", "optimized")
    labels = {
        "baseline": "TVCFormer baseline",
        "pre_optimization": "Clustering before optimization",
        "optimized": "Clustering optimized",
    }
    styles = {"baseline": "o-", "pre_optimization": "s--", "optimized": "^-"}
    for group in groups:
        xs, ys = [], []
        for horizon in HORIZONS:
            row = rows_by_key.get((group, horizon))
            val = number(row, field) if row else ""
            if val != "":
                xs.append(horizon)
                ys.append(val)
        if xs:
            ax.plot(xs, ys, styles[group], linewidth=2, markersize=6, label=labels[group])
    ax.set_title(title)
    ax.set_xlabel("Prediction length")
    ax.set_ylabel(ylabel)
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=160)
    plt.close(fig)


def main():
    historic_rows = read_rows(HISTORIC_CSV)
    optimized_rows = read_rows(OPTIMIZED_CSV)
    current_config = yaml.safe_load((PROJECT / "configs" / "phase5_etth1.yaml").read_text(encoding="utf-8"))

    records = []
    grouped = {}
    config_checks = []
    for horizon in HORIZONS:
        baseline = find_completed(historic_rows, "baseline", horizon)
        old_cluster = find_completed(historic_rows, "clustering", horizon)
        new_cluster = find_completed(optimized_rows, "clustering", horizon)
        if not baseline or not new_cluster:
            raise RuntimeError(f"missing required baseline or optimized result for pred_len={horizon}")
        baseline_pairs = int(float(baseline["pearson_pairs"]))
        records.append(make_record("baseline", baseline, horizon, "completed", baseline_pairs))
        records.append(make_record("pre_optimization", old_cluster, horizon,
                                   "completed" if old_cluster else "not_available",
                                   baseline_pairs))
        records.append(make_record("optimized", new_cluster, horizon, "completed",
                                   baseline_pairs, optimized=True))
        for group, row in (("baseline", baseline), ("pre_optimization", old_cluster),
                           ("optimized", new_cluster)):
            if row:
                config_path = ROOT / Path(row["config_file"].replace("\\", "/"))
                run_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                same = all(run_config.get(key) == current_config.get(key) for key in CONFIG_KEYS)
                config_checks.append((horizon, group, same))
                if not same:
                    differences = [key for key in CONFIG_KEYS
                                   if run_config.get(key) != current_config.get(key)]
                    raise RuntimeError(f"configuration mismatch {group} pred_len={horizon}: {differences}")

    fields = list(dict.fromkeys(key for row in records for key in row.keys()))
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    rows_by_key = {(row["model_group"], int(row["pred_len"])): row for row in records}
    write_chart(rows_by_key, "test_mse", "ETTh1 full-test MSE", "MSE", "phase5_etth1_final_mse.png")
    write_chart(rows_by_key, "test_mae", "ETTh1 full-test MAE", "MAE", "phase5_etth1_final_mae.png")
    write_chart(rows_by_key, "pearson_pairs", "Pearson pair evaluations per completed run",
                "Pearson pairs", "phase5_etth1_final_pearson_pairs.png")
    write_chart(rows_by_key, "total_time_s", "Run wall time (includes test; optimized includes precompute)",
                "Seconds", "phase5_etth1_final_total_time.png")
    write_chart(rows_by_key, "training_time_s", "Training and validation loop time",
                "Seconds", "phase5_etth1_final_training_time.png")
    write_chart(rows_by_key, "peak_gpu_memory_mib", "Peak allocated GPU memory",
                "MiB", "phase5_etth1_final_peak_gpu_memory.png")

    report = [
        "# ETTh1 final comparison: Clustering-TVCFormer after optimization",
        "",
        "## Protocol and provenance",
        "",
        "- Dataset: ETTh1; `seq_len=336`; prediction lengths 96, 192, 336, 720; seed 2021.",
        "- GPU: NVIDIA GeForce GTX 1650; PyTorch 1.13.1 + CUDA 11.6. Batch size 128; other settings are recorded in the linked run configs.",
        "- Baseline rows are reused from the completed original phase-5 CSV and their original logs; baseline was not rerun.",
        "- Pre-optimization rows use only completed historical clustering runs. No failed/interrupted row is used as a metric.",
        "- Optimized rows are the fresh indexed-cache runs and evaluate every test sample. Their CSV/logs are separate from prior results.",
        "- A config check compared the run YAMLs with the current phase-5 YAML for sequence length, batch size, optimizer/scheduler, model, and clustering hyperparameters. All recorded completed rows matched on those fields.",
        "- Training time is the runner's train+validation loop timer. Optimized precompute is outside that timer and is shown separately. Optimized total time is the high-resolution runner timer from model setup through full test evaluation. Historical baseline/pre-optimization total wall time is derived from second-resolution start/end timestamps in each run log; small rounding error is possible.",
        "- Pearson reduction is measured against the baseline's actual pair evaluations at the same prediction length. Optimized Pearson count is for unique indexed histories across train/validation/test, since cached rankings are computed once and reused across epochs.",
        "",
        "## Three-way results",
        "",
        "Training time is train+validation. Total time includes full test evaluation; for optimized runs it also includes one-time precompute. Pearson reduction is relative to baseline.",
        "",
        "| pred_len | model | status | test MSE | test MAE | best epoch | train+val (s) | mean epoch (s) | precompute (s) | total/wall (s) | Pearson pairs | reduction vs baseline | K-Means calls | cache hit/miss | peak GPU (MiB) | test samples |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        for group in ("baseline", "pre_optimization", "optimized"):
            row = rows_by_key.get((group, horizon), {})
            label = {"baseline": "TVCFormer baseline", "pre_optimization": "Clustering pre-optimization",
                     "optimized": "Clustering indexed optimized"}[group]
            if row.get("status") != "completed":
                report.append(f"| {horizon} | {label} | {row.get('status', 'not_available')} | — | — | — | — | — | — | — | — | — | — | — | — | — |")
                continue
            report.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                horizon, label, row["status"], fmt(row.get("test_mse"), 7), fmt(row.get("test_mae"), 7),
                row.get("best_epoch", "—"), fmt(row.get("training_time_s"), 2),
                fmt(row.get("epoch_time_s"), 2), fmt(row.get("precompute_time_s"), 2),
                fmt(row.get("total_time_s"), 2), fmt(row.get("pearson_pairs"), 0),
                fmt(row.get("pearson_reduction_pct"), 2), row.get("kmeans_calls", "—") or "—",
                "{}/{}".format(row.get("cache_hits", "—") or "—", row.get("cache_misses", "—") or "—"),
                fmt(row.get("peak_gpu_memory_mib"), 2), row.get("test_samples", "—")))

    report.extend(["", "## Optimized-run timing breakdown", ""])
    report.extend([
        "The selector clustering time includes indexed history preparation/selection activity as reported by the selector; Pearson and Graph Generator timers are reported separately. Precompute overlaps these component timers and must not be added to them when calculating total time.",
        "",
        "| pred_len | test evaluation (s) | precompute (s) | selector clustering (s) | feature extraction (s) | K-Means (s) | candidate generation (s) | Pearson (s) | Graph Generator (s) | cache lookup (s) | CPU/GPU transfer (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for horizon in HORIZONS:
        row = rows_by_key[("optimized", horizon)]
        report.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            horizon, fmt(row.get("eval_time_s"), 3), fmt(row.get("precompute_time_s"), 3),
            fmt(row.get("clustering_time_s"), 3), fmt(row.get("feature_time_s"), 3),
            fmt(row.get("kmeans_time_s"), 3), fmt(row.get("candidate_time_s"), 3),
            fmt(row.get("pearson_time_s"), 3), fmt(row.get("graph_time_s"), 3),
            fmt(row.get("cache_lookup_time_s"), 3), fmt(row.get("transfer_time_s"), 3)))
    report.append("")

    report.extend(["## Direct comparison and interpretation", ""])
    for horizon in HORIZONS:
        base = rows_by_key[("baseline", horizon)]
        new = rows_by_key[("optimized", horizon)]
        old = rows_by_key.get(("pre_optimization", horizon))
        wall_vs_base = (1 - float(new["total_time_s"]) / float(base["total_time_s"])) * 100
        report.append(f"### pred_len={horizon}")
        report.append("")
        report.append(f"- Optimized vs baseline run wall time: {wall_vs_base:+.1f}% (negative means slower).")
        report.append(f"- Optimized Pearson evaluations were reduced by {float(new['pearson_reduction_pct']):.2f}% versus baseline.")
        if old and old.get("status") == "completed":
            new_mse_delta = float(new["test_mse"]) - float(old["test_mse"])
            new_mae_delta = float(new["test_mae"]) - float(old["test_mae"])
            train_drop = (1 - float(new["training_time_s"]) / float(old["training_time_s"])) * 100
            wall_drop = (1 - float(new["total_time_s"]) / float(old["total_time_s"])) * 100
            pair_drop = (1 - float(new["pearson_pairs"]) / float(old["pearson_pairs"])) * 100
            report.append(f"- Versus the completed pre-optimization clustering run: test MSE changed by {new_mse_delta:+.8f}, MAE by {new_mae_delta:+.8f}; train+validation time changed by {train_drop:.1f}%, log/runner wall time by {wall_drop:.1f}%, and recorded Pearson pairs by {pair_drop:.1f}%.")
        else:
            report.append("- No completed pre-optimization clustering run exists for this horizon; the historical run was unavailable/failed, so no before/after claim is made.")
        report.append("")

    report.extend([
        "## Overall conclusion",
        "",
        "The indexed precompute materially lowers repeat work and Pearson evaluations, and for the three horizons with completed historical clustering runs the measured test metrics remain numerically very close to the pre-optimization results. This is one seed and one GPU, so it does not establish statistical equivalence or generalization across hardware.",
        "",
        "The optimization lowers clustering runtime versus the prior clustering implementation, but the indexed runs remain slower end-to-end than the baseline at every horizon because one-time K-Means/feature precomputation is substantial. The training-loop timer alone excludes that precompute and therefore must not be presented as total cost. On these measurements the method has not met a goal of being faster than baseline, although it substantially reduces the Pearson workload while preserving near-identical forecast metrics.",
        "",
        "For pred_len=720, there is no completed pre-optimization clustering result. The three-way comparison for that cell is unavailable; only baseline versus optimized can be assessed.",
        "",
        "## Artifacts",
        "",
        f"- Detailed long-form CSV: `{OUTPUT_CSV.relative_to(ROOT).as_posix()}`",
        f"- Optimized-only run CSV: `{OPTIMIZED_CSV.relative_to(ROOT).as_posix()}`",
        "- Per-run logs: `experiments/results/Clustering_TVC/`",
        "- Figures: `FinalProject/Clustering_TVC/figures/phase5_etth1_final_*.png`",
        "- All four optimized runs completed; no full experiments on ETTm1, ETTm2, or Electricity were started.",
        "",
    ])
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {REPORT}")
    print(f"config checks passed: {len(config_checks)}")
    print(f"figure count: {len(list(FIGURES.glob('phase5_etth1_final_*.png')))}")


if __name__ == "__main__":
    main()
