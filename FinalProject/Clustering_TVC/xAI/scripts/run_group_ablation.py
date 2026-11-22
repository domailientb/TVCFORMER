"""Inference-only Group Ablation for a trained Clustering-TVC checkpoint.

This script never trains a model. Group membership is taken from the full,
unmasked history window for each test sample. The masked pass is then evaluated
by the same model and selector, which rebuilds its graph from the masked input.
"""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
ROOT = PROJECT.parent.parent
OFFICIAL = ROOT / "official"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from FinalProject.Clustering_TVC.clustering_utils import (
    ClusteringGraphSelector, extract_variable_features,
)
from FinalProject.Clustering_TVC.run_phase5 import _load_config, attach_selector, seed_all


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.source[index] + (index,)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _metrics(predictions, targets):
    difference = predictions.astype(np.float64) - targets.astype(np.float64)
    return float(np.mean(difference ** 2)), float(np.mean(np.abs(difference)))


def _write_csv(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_checkpoint(path, model, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    elif isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a model state_dict")
    # Accommodate DataParallel checkpoints without relaxing key/shape checks.
    if payload and all(key.startswith("module.") for key in payload):
        payload = {key[len("module."):]: value for key, value in payload.items()}
    model.load_state_dict(payload, strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / ".." / "configs" / "phase5_etth1.yaml"))
    parser.add_argument("--checkpoint", required=True, help="Trained Clustering-TVC state_dict")
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not config_path.is_file():
        raise FileNotFoundError("Configuration not found: {}".format(config_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Clustering-TVC checkpoint not found: {}. This script is inference-only; "
            "it will not train or substitute a baseline checkpoint.".format(checkpoint_path))

    os_config = _load_config(config_path)
    os_config["root_path"] = str((ROOT / os_config["root_path"]).resolve())
    os_config["pred_len"] = args.pred_len
    os_config["random_seed"] = args.seed
    os_config["use_gpu"] = bool(torch.cuda.is_available() and args.device != "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.chdir(ROOT)
    seed_all(args.seed)
    from argparse import Namespace
    model_args = Namespace(**os_config)
    experiment = Exp_Main(model_args)
    model = experiment.model
    if args.device == "cpu":
        model = model.cpu()
        experiment.device = torch.device("cpu")
    selector = ClusteringGraphSelector(
        n_clusters=model_args.n_clusters,
        keep_rate=model_args.subgraph_keep_rate,
        cross_edge_ratio=model_args.cross_edge_ratio,
        candidate_multiplier=model_args.candidate_multiplier,
        random_state=args.seed,
        batch_edge_assembly=getattr(model_args, "batch_edge_assembly", True),
    )
    attach_selector(model, selector)
    _load_checkpoint(checkpoint_path, model, experiment.device)
    model.eval()

    train_data, _ = data_provider(model_args, "train")
    test_data, _ = data_provider(model_args, "test")
    loader = DataLoader(IndexedDataset(test_data), batch_size=model_args.batch_size,
                        shuffle=False, num_workers=model_args.num_workers, drop_last=False)
    train_mean = torch.as_tensor(np.asarray(train_data.data_x).mean(axis=0),
                                 dtype=torch.float32, device=experiment.device)
    if train_mean.numel() != model_args.enc_in:
        raise ValueError("Train mean feature count does not match model enc_in")

    # Capture the exact directed edge tensor returned by the installed selector.
    edge_records = []
    original_select = selector.select
    def capture_select(*call_args, **call_kwargs):
        edges = original_select(*call_args, **call_kwargs)
        edge_records.append((edges.detach().cpu().numpy(),
                             [label.copy() for label in selector.last_labels]))
        return edges
    selector.select = capture_select

    output_root = HERE / "results"
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = "group_ablation_{}_p{}_s{}_{}".format(
        model_args.data, args.pred_len, args.seed, datetime.now().strftime("%Y%m%dT%H%M%S%f"))
    run_dir = output_root / run_id
    run_dir.mkdir()
    start_xai = time.perf_counter()
    baseline_predictions, targets = [], []
    cluster_windows = {}
    cluster_feature_sums = {}
    cluster_feature_counts = {}
    cluster_sensor_counts = {}
    graph_rows = []
    labels_by_sample = {}
    baseline_start = time.perf_counter()
    _sync(experiment.device)
    with torch.inference_mode():
        for batch_x, batch_y, _, _, sample_ids in loader:
            x = batch_x.float().to(experiment.device)
            y = batch_y[:, -args.pred_len:, :].float().to(experiment.device)
            # XAI inputs are not pre-indexed. Let the selector prepare/cache
            # graphs by window content, including distinct masked windows.
            selector.set_sample_ids(None)
            prediction = model(x)
            labels_batch = [label.copy() for label in selector.last_labels]
            baseline_predictions.append(prediction.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
            windows = x.detach().cpu().numpy().transpose(0, 2, 1)
            for local, (sample_id, labels, window) in enumerate(zip(sample_ids, labels_batch, windows)):
                labels_by_sample[int(sample_id)] = labels
                features = extract_variable_features(window)
                for group_id in np.unique(labels):
                    key = int(group_id)
                    members = np.flatnonzero(labels == group_id)
                    cluster_windows[key] = cluster_windows.get(key, 0) + 1
                    cluster_sensor_counts.setdefault(key, np.zeros(model_args.enc_in, dtype=np.int64))
                    cluster_sensor_counts[key][members] += 1
                    cluster_feature_sums.setdefault(key, np.zeros(features.shape[1], dtype=np.float64))
                    cluster_feature_counts[key] = cluster_feature_counts.get(key, 0) + len(members)
                    cluster_feature_sums[key] += features[members].sum(axis=0)
            # Edge accounting uses exactly the edges passed to the graph generator.
            captured = edge_records[-1]
            edge_array, edge_labels = captured
            for local, labels in enumerate(edge_labels):
                directed = edge_array[local]
                non_self = directed[directed[:, 1] != directed[:, 2]]
                intra = sum(labels[int(src)] == labels[int(dst)] for src, dst in non_self[:, 1:3])
                graph_rows.append({"sample_id": int(sample_ids[local]),
                                   "nodes": int(model_args.enc_in),
                                   "directed_edges": int(len(directed)),
                                   "undirected_nonself_edges": int(len(non_self) // 2),
                                   "intra_cluster_directed_edges": int(intra),
                                   "inter_cluster_directed_edges": int(len(non_self) - intra)})
    _sync(experiment.device)
    baseline_seconds = time.perf_counter() - baseline_start
    # Do not retain a second copy of every ablation graph in host memory.
    selector.select = original_select
    pred = np.concatenate(baseline_predictions)
    truth = np.concatenate(targets)
    baseline_mse, baseline_mae = _metrics(pred, truth)
    np.savez_compressed(run_dir / "baseline_predictions.npz", prediction=pred, target=truth)

    ablation_rows = []
    ablation_seconds_total = 0.0
    for group_id in sorted(cluster_windows):
        predictions = []
        started = time.perf_counter()
        _sync(experiment.device)
        with torch.inference_mode():
            for batch_x, _, _, _, sample_ids in loader:
                x = batch_x.float().to(experiment.device).clone()
                # Membership comes from the corresponding unmasked windows.
                for local, sample_id in enumerate(sample_ids):
                    labels = labels_by_sample[int(sample_id)]
                    mask = torch.as_tensor(labels == group_id, device=x.device)
                    x[local, :, mask] = train_mean[mask]
                selector.set_sample_ids(None)
                predictions.append(model(x).detach().cpu().numpy())
        _sync(experiment.device)
        duration = time.perf_counter() - started
        ablation_seconds_total += duration
        ablated = np.concatenate(predictions)
        mse, mae = _metrics(ablated, truth)
        members_seen = int(cluster_sensor_counts[group_id].sum())
        ablation_rows.append({
            "cluster": group_id,
            "sensors": ";".join("{}:{}".format(i, int(v)) for i, v in enumerate(cluster_sensor_counts[group_id]) if v),
            "number_of_sensors_max": int(np.count_nonzero(cluster_sensor_counts[group_id])),
            "sensor_window_memberships": members_seen,
            "windows_present": int(cluster_windows[group_id]),
            "baseline_mse": baseline_mse, "ablated_mse": mse, "delta_mse": mse - baseline_mse,
            "baseline_mae": baseline_mae, "ablated_mae": mae, "delta_mae": mae - baseline_mae,
            "inference_seconds": duration,
        })

    # Write detailed cluster descriptions in standardized per-window feature space.
    feature_names = ("mean", "std", "trend", "skewness", "q10", "q50", "q90", "autocorrelation_lag1")
    cluster_rows = []
    for group_id in sorted(cluster_feature_sums):
        means = cluster_feature_sums[group_id] / cluster_feature_counts[group_id]
        row = {"cluster_id_local": group_id, "window_appearances": cluster_windows[group_id],
               "sensor_window_memberships": cluster_feature_counts[group_id]}
        row.update({name: float(value) for name, value in zip(feature_names, means)})
        row["sensor_window_frequency"] = json.dumps(cluster_sensor_counts[group_id].tolist())
        cluster_rows.append(row)
    _write_csv(run_dir / "group_ablation_results.csv", ablation_rows,
               list(ablation_rows[0].keys()) if ablation_rows else ["cluster"])
    _write_csv(run_dir / "cluster_statistics.csv", cluster_rows,
               list(cluster_rows[0].keys()) if cluster_rows else ["cluster_id_local"])
    _write_csv(run_dir / "graph_statistics.csv", graph_rows,
               list(graph_rows[0].keys()) if graph_rows else ["sample_id"])
    runtime = [{"baseline_inference_seconds": baseline_seconds,
                "group_ablation_inference_seconds": ablation_seconds_total,
                "cluster_interpretation_seconds": "included_in_baseline_pass",
                "graph_analysis_seconds": "included_in_baseline_pass",
                "total_xai_seconds": time.perf_counter() - start_xai,
                "test_windows": len(test_data), "clusters_evaluated": len(ablation_rows),
                "checkpoint": str(checkpoint_path)}]
    _write_csv(run_dir / "xai_runtime.csv", runtime, list(runtime[0].keys()))
    _make_figures(run_dir, ablation_rows, cluster_rows, graph_rows)
    print("XAI outputs:", run_dir)
    print("baseline MSE={:.8f} MAE={:.8f}; windows={}".format(baseline_mse, baseline_mae, len(test_data)))


def _make_figures(run_dir, ablation_rows, cluster_rows, graph_rows):
    import matplotlib.pyplot as plt
    if ablation_rows:
        for metric, label, filename in (("delta_mse", "Delta MSE", "group_ablation_mse.png"),
                                        ("delta_mae", "Delta MAE", "group_ablation_mae.png")):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar([str(row["cluster"]) for row in ablation_rows], [row[metric] for row in ablation_rows])
            ax.set(xlabel="Local cluster ID", ylabel=label, title="Group Ablation: " + label)
            ax.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(run_dir / filename, dpi=160); plt.close(fig)
    if cluster_rows:
        names = ["mean", "std", "trend", "skewness", "q10", "q50", "q90", "autocorrelation_lag1"]
        matrix = np.asarray([[row[name] for name in names] for row in cluster_rows])
        fig, ax = plt.subplots(figsize=(10, max(3, len(cluster_rows) * .6)))
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set_yticks(range(len(cluster_rows)), [str(row["cluster_id_local"]) for row in cluster_rows])
        ax.set(xlabel="Feature (per-window standardized)", ylabel="Local cluster ID", title="Cluster feature summary")
        fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(run_dir / "cluster_feature_heatmap.png", dpi=160); plt.close(fig)
    if graph_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist([row["undirected_nonself_edges"] for row in graph_rows], bins="auto")
        ax.set(xlabel="Undirected non-self edges per test window", ylabel="Window count", title="TVC graph edge-count distribution")
        fig.tight_layout(); fig.savefig(run_dir / "tvc_graph_clusters.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
