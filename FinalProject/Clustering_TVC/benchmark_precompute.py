"""Cold/warm indexed-cache benchmark of the complete ETTh1 precompute set."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(ROOT / "official"))
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from FinalProject.Clustering_TVC.clustering_utils import ClusteringGraphSelector
from FinalProject.Clustering_TVC.run_phase5 import _load_config


def get_datasets(config_path):
    config = _load_config(Path(config_path))
    config["root_path"] = str((ROOT / config["root_path"]).resolve())
    config.update(pred_len=96, random_seed=2021, use_gpu=False,
                  device="cpu", use_multi_gpu=False, model="TVCFormer")
    args = argparse.Namespace(**config)
    datasets = []
    for split in ("train", "val", "test"):
        dataset, _ = data_provider(args, split)
        datasets.append((split, dataset))
    return config, datasets


def bench_once(datasets, config, run_index, implementation):
    selector = ClusteringGraphSelector(
        n_clusters=config["n_clusters"], keep_rate=config["subgraph_keep_rate"],
        cross_edge_ratio=config["cross_edge_ratio"],
        candidate_multiplier=config["candidate_multiplier"],
        random_state=2021, kmeans_backend="numpy_vectorized",
        batch_edge_assembly=True)
    count = 0
    cold_start = time.perf_counter()
    for split, dataset in datasets:
        for start in range(0, len(dataset), 128):
            end = min(start + 128, len(dataset))
            histories = [dataset.data_x[index:index + dataset.seq_len]
                         for index in range(start, end)]
            if any(len(history) != dataset.seq_len for history in histories):
                raise RuntimeError(f"invalid history length in {split}[{start}:{end}]")
            if implementation == "batched":
                selector.prepare_indexed_windows_batch(
                    [(split, index) for index in range(start, end)],
                    np.stack(histories, axis=0))
            else:
                for offset, history in enumerate(histories):
                    selector.prepare_indexed_window((split, start + offset), history)
            count += len(histories)
    cold_seconds = time.perf_counter() - cold_start
    if count != selector.stats.cache_misses or count != selector.stats.kmeans_calls:
        raise AssertionError("cold precompute count, cache miss count and K-Means calls differ")
    cold = {
        "run": run_index, "window_count": count, "cold_seconds": cold_seconds,
        "feature_seconds": selector.stats.feature_seconds,
        "kmeans_seconds": selector.stats.kmeans_seconds,
        "candidate_seconds": selector.stats.candidate_seconds,
        "pearson_seconds": selector.stats.pearson_seconds,
        "edge_selection_seconds": selector.stats.edge_selection_seconds,
        "cache_lookup_seconds": selector.stats.cache_seconds,
        "other_seconds": cold_seconds - sum((
            selector.stats.feature_seconds, selector.stats.kmeans_seconds,
            selector.stats.candidate_seconds, selector.stats.pearson_seconds,
            selector.stats.edge_selection_seconds, selector.stats.cache_seconds)),
        "pearson_pairs": selector.stats.pearson_pairs,
        "kmeans_calls": selector.stats.kmeans_calls,
        "cache_misses": selector.stats.cache_misses,
    }
    warm_start = time.perf_counter()
    for split, dataset in datasets:
        for index in range(len(dataset)):
            selector.prepare_indexed_window((split, index),
                                            dataset.data_x[index:index + dataset.seq_len])
    warm_seconds = time.perf_counter() - warm_start
    if selector.stats.cache_hits != count or selector.stats.cache_misses != count:
        raise AssertionError("warm cache did not hit every precomputed window exactly once")
    warm = {"run": run_index, "window_count": count, "warm_seconds": warm_seconds,
            "warm_cache_hits": selector.stats.cache_hits,
            "warm_cache_misses": selector.stats.cache_misses,
            "warm_kmeans_calls": selector.stats.kmeans_calls}
    return cold, warm


def summarize(rows, warm_rows, directory, session_id, config, split_counts, implementation):
    cold_path = directory / f"precompute_{session_id}_cold.csv"
    warm_path = directory / f"precompute_{session_id}_warm.csv"
    for path, data in ((cold_path, rows), (warm_path, warm_rows)):
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            writer.writeheader()
            writer.writerows(data)

    component_keys = ("feature_seconds", "kmeans_seconds", "candidate_seconds",
                      "pearson_seconds", "edge_selection_seconds", "cache_lookup_seconds",
                      "other_seconds", "cold_seconds")
    summary = {}
    for key in component_keys:
        vals = [float(row[key]) for row in rows]
        summary[key + "_mean"] = statistics.mean(vals)
        summary[key + "_median"] = statistics.median(vals)
        summary[key + "_stdev"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    warm_values = [float(row["warm_seconds"]) for row in warm_rows]
    summary.update(warm_seconds_mean=statistics.mean(warm_values),
                   warm_seconds_median=statistics.median(warm_values),
                   window_count=rows[0]["window_count"],
                   pearson_pairs=rows[0]["pearson_pairs"],
                   kmeans_calls=rows[0]["kmeans_calls"])
    summary_path = directory / f"precompute_{session_id}_summary.csv"
    with summary_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    report = [
        "# ETTh1 precompute benchmark", "",
        "- Component-only benchmark; the data provider loads train/validation/test arrays before the timed section.",
        f"- Implementation: `{implementation}`; reference uses one window per call, batched uses 128 history windows per feature/K-Means call.",
        "- Cold means a fresh in-memory selector cache and all histories prepared once. Warm means every same `(split, index)` ID is looked up a second time in that selector.",
        "- Cold total is the wall timer around the full precompute loop. Feature, K-Means, candidate, Pearson, edge selection and cache timers are non-overlapping instrumented sections; other is their residual. Warm total is independently timed and is not added to cold.",
        "- Hardware: CPU NumPy precompute; model training is not part of this diagnostic.", "",
        f"Config: ETTh1, seq_len={config['seq_len']}, pred_len=96, seed=2021, K={config['n_clusters']}, keep_rate={config['subgraph_keep_rate']}, cross_edge_ratio={config['cross_edge_ratio']}, backend=numpy_vectorized.",
        f"Actual history count: {split_counts['train']} train + {split_counts['val']} validation + {split_counts['test']} test = {summary['window_count']} windows.",
        "",
        "| Component | Mean (s) | Median (s) | SD (s) |", "|---|---:|---:|---:|",
    ]
    labels = {"feature_seconds": "Feature extraction", "kmeans_seconds": "K-Means",
              "candidate_seconds": "Candidate pair generation/ranking",
              "pearson_seconds": "Candidate-only Pearson",
              "edge_selection_seconds": "Pearson top-k/edge selection",
              "cache_lookup_seconds": "Cold cache lookup",
              "other_seconds": "Other precompute loop work", "cold_seconds": "Cold total"}
    for key, label in labels.items():
        report.append(f"| {label} | {summary[key + '_mean']:.4f} | {summary[key + '_median']:.4f} | {summary[key + '_stdev']:.4f} |")
    report.extend(["", f"Warm-cache full-ID pass: mean {summary['warm_seconds_mean']:.4f}s, median {summary['warm_seconds_median']:.4f}s; hits={summary['window_count']}, misses={summary['window_count']} cumulative, K-Means calls remain {summary['kmeans_calls']}.",
                   f"Pearson pairs over the cold pass: {summary['pearson_pairs']}; each history is processed exactly once.",
                   "", f"Cold rows: `{cold_path.name}`; warm rows: `{warm_path.name}`; summary: `{summary_path.name}`.", ""])
    report_path = directory / f"precompute_{session_id}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return summary, report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT / "configs" / "phase5_etth1.yaml"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--implementation", choices=("reference", "batched"), default="reference")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    config, datasets = get_datasets(args.config)
    split_counts = {split: len(dataset) for split, dataset in datasets}
    expected = sum(split_counts.values())
    if expected != 13779:
        raise RuntimeError(f"current ETTh1 pipeline has {expected} history windows, expected 13779")

    session_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    directory = PROJECT / "results" / f"precompute_benchmark_{session_id}"
    directory.mkdir(parents=True, exist_ok=False)
    log_path = directory / f"precompute_{session_id}.log"
    (directory / "session.json").write_text(json.dumps({
        "status": "running", "dataset": "ETTh1", "seq_len": config["seq_len"],
        "pred_len": 96, "seed": 2021, "repeats": args.repeats,
        "implementation": args.implementation,
        "split_counts": split_counts, "window_count": expected,
        "config": str(Path(args.config).resolve()),
    }, indent=2), encoding="utf-8")
    cold_rows, warm_rows = [], []
    with log_path.open("x", encoding="utf-8") as log:
        for repeat in range(1, args.repeats + 1):
            print(f"precompute repeat {repeat}/{args.repeats}", flush=True)
            cold, warm = bench_once(datasets, config, repeat, args.implementation)
            cold_rows.append(cold)
            warm_rows.append(warm)
            line = f"run={repeat} cold={cold['cold_seconds']:.4f}s warm={warm['warm_seconds']:.4f}s KMeans={cold['kmeans_seconds']:.4f}s Pearson={cold['pearson_seconds']:.4f}s"
            print(line, flush=True)
            log.write(line + "\n")
    summary, report = summarize(cold_rows, warm_rows, directory, session_id,
                                config, split_counts, args.implementation)
    metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    metadata.update(status="completed", report=str(report.relative_to(ROOT)),
                    cold_csv=str((directory / f"precompute_{session_id}_cold.csv").relative_to(ROOT)),
                    warm_csv=str((directory / f"precompute_{session_id}_warm.csv").relative_to(ROOT)),
                    summary_csv=str((directory / f"precompute_{session_id}_summary.csv").relative_to(ROOT)))
    (directory / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("report:", report)
    print("summary:", directory / f"precompute_{session_id}_summary.csv")


if __name__ == "__main__":
    main()
