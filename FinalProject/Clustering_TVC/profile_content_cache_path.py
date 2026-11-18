"""Replay the selector's content-hash miss/hit workload on ETTh1 histories."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "official"))

from data_provider.data_factory import data_provider
from FinalProject.Clustering_TVC.clustering_utils import ClusteringGraphSelector
from FinalProject.Clustering_TVC.run_phase5 import _load_config


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_misses", type=int, default=13059)
    parser.add_argument("--cache_hits", type=int, default=72430)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", choices=("numpy", "numpy_vectorized"),
                        default="numpy_vectorized")
    args = parser.parse_args()
    config = _load_config(PROJECT / "configs" / "phase5_etth1.yaml")
    config["root_path"] = str((ROOT / config["root_path"]).resolve())
    config["pred_len"] = 336
    config["random_seed"] = 2021
    config["use_gpu"] = False
    data_args = argparse.Namespace(**config)
    windows = []
    split_sizes = {}
    for split in ("train", "val", "test"):
        dataset, _ = data_provider(data_args, split)
        split_sizes[split] = len(dataset)
        windows.extend(np.asarray(dataset.data_x[i:i + dataset.seq_len], dtype=np.float32)
                       for i in range(len(dataset)))
    if len(windows) != args.cache_misses:
        raise ValueError(f"requested misses={args.cache_misses}, loaded unique windows={len(windows)}")
    device = torch.device(args.device if args.device != "cuda:0" or torch.cuda.is_available() else "cpu")
    selector = ClusteringGraphSelector(
        n_clusters=config["n_clusters"], keep_rate=config["subgraph_keep_rate"],
        cross_edge_ratio=config["cross_edge_ratio"],
        candidate_multiplier=config["candidate_multiplier"], random_state=2021,
        kmeans_backend=args.backend,
    )

    def run_indices(indices):
        for start in range(0, len(indices), config["batch_size"]):
            chunk = indices[start:start + config["batch_size"]]
            # [B,T,C] histories are transposed to the selector's [B,C,T].
            cpu_values = torch.from_numpy(np.stack([windows[i].T for i in chunk]))
            values = cpu_values.to(device)
            selector.select(values)

    sync(device)
    wall_start = time.perf_counter()
    run_indices(list(range(len(windows))))
    # Deterministic repeat schedule with the exact requested hit count.
    run_indices([index % len(windows) for index in range(args.cache_hits)])
    sync(device)
    wall_seconds = time.perf_counter() - wall_start
    stats = selector.stats
    if stats.cache_misses != args.cache_misses or stats.cache_hits != args.cache_hits:
        raise RuntimeError(f"cache counters mismatch: hits={stats.cache_hits}, misses={stats.cache_misses}")
    if stats.kmeans_calls != stats.cache_misses:
        raise RuntimeError(f"K-Means calls {stats.kmeans_calls} != cache misses {stats.cache_misses}")
    if stats.pearson_pairs <= 0 or stats.full_pairs <= stats.pearson_pairs:
        raise RuntimeError("Pearson candidate counter or pruning check failed")

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    results = PROJECT / "results"
    results.mkdir(parents=True, exist_ok=True)
    csv_path = results / f"content_cache_profile_{timestamp}.csv"
    report_path = results / f"content_cache_profile_{timestamp}.md"
    stats_row = {
        "dataset": "ETTh1", "pred_len": 336, "seq_len": config["seq_len"],
        "device": str(device), "backend": selector.kmeans_backend,
        "unique_windows": len(windows), "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses, "kmeans_calls": stats.kmeans_calls,
        "wall_seconds": wall_seconds, "feature_seconds": stats.feature_seconds,
        "kmeans_seconds": stats.kmeans_seconds,
        "kmeans_init_seconds": stats.kmeans_init_seconds,
        "kmeans_fit_seconds": stats.kmeans_fit_seconds,
        "candidate_seconds": stats.candidate_seconds,
        "cache_lookup_and_hash_seconds": stats.cache_seconds,
        "transfer_seconds": stats.transfer_seconds,
        "pearson_seconds": stats.pearson_seconds,
        "cluster_seconds_excluding_pearson_transfer": stats.cluster_seconds,
        "pearson_pairs": stats.pearson_pairs,
        "full_pair_equivalent": stats.full_pairs,
    }
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats_row))
        writer.writeheader()
        writer.writerow(stats_row)
    report = [
        "# ETTh1 content-hash cache selector profile",
        "",
        f"- Histories: `{len(windows)}` from split lengths `{split_sizes}`; only `data_x[index:index+seq_len]` was read.",
        f"- Replay counts: `{stats.cache_misses}` cold misses and `{stats.cache_hits}` cache hits; K-Means calls were `{stats.kmeans_calls}`.",
        f"- Device for selector inputs: `{device}`; the model/Graph Generator was not run.",
        f"- Replay wall time: `{wall_seconds:.3f} s`.",
        "",
        "| Component | Seconds | Accounting note |",
        "|---|---:|---|",
        f"| Feature extraction | {stats.feature_seconds:.3f} | Cold misses only |",
        f"| K-Means total | {stats.kmeans_seconds:.3f} | Cold misses only |",
        f"| K-Means initialization | {stats.kmeans_init_seconds:.3f} | Included in K-Means total |",
        f"| Lloyd fit/update | {stats.kmeans_fit_seconds:.3f} | Included in K-Means total |",
        f"| Candidate construction | {stats.candidate_seconds:.3f} | Cold misses only |",
        f"| Cache hash/lookup | {stats.cache_seconds:.3f} | Cold and repeated windows |",
        f"| GPU-to-CPU selector transfer | {stats.transfer_seconds:.3f} | Includes CUDA synchronization |",
        f"| Pearson scoring | {stats.pearson_seconds:.3f} | Recomputed on every forward, including hits; {stats.pearson_pairs:,} actual pairs |",
        f"| Clustering selector timer excluding Pearson/transfer | {stats.cluster_seconds:.3f} | Hash/cache, miss-only features/K-Means/candidates and per-window selector Python work |",
        "",
        f"Full-pair equivalent: `{stats.full_pairs:,}`; candidate Pearson reduction: `{100.0 * (1.0 - stats.pearson_pairs / stats.full_pairs):.2f}%`.",
        "",
        "This deterministic hit schedule repeats windows round-robin to reproduce the requested hit count. It measures the selector path, not a model training run; real forward ordering and repeat frequencies can shift cache/hash and Pearson wall times.",
        "",
        f"Detailed CSV: `{csv_path.relative_to(ROOT).as_posix()}`",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("profile report:", report_path)
    print("profile csv:", csv_path)
    print("stats:", stats_row)


if __name__ == "__main__":
    main()
