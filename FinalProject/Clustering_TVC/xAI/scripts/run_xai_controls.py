"""Run matched feature/random ablations and graph overlap on the ETTh1 test set.

Inference only: loads an existing Clustering-TVC checkpoint and uses the same
test windows for the baseline forecast, every intervention, and graph analysis.
All interventions replace masked, scaled input channels with their training-set
mean, matching the existing group-ablation script.
"""
import argparse
import csv
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
XAI = HERE.parent
PROJECT = XAI.parent
ROOT = PROJECT.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "official"))

from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from FinalProject.Clustering_TVC.clustering_utils import (
    ClusteringGraphSelector, extract_variable_features,
)
from FinalProject.Clustering_TVC.run_phase5 import (
    _load_config, attach_selector, seed_all,
)


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.source[index] + (index,)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mse_mae(pred, target):
    d = pred.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(d * d)), float(np.mean(np.abs(d)))


def pearson_baseline_pairs(window, count):
    """Match official top_k_channel_pairs_choose: signed r, upper triangle."""
    c = window.shape[1]
    centered = window - window.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(centered, axis=0)
    pair_denom = denom[:, None] * denom[None, :]
    corr = (centered.T @ centered) / np.where(pair_denom > 0.0, pair_denom, 1.0)
    pairs = [(i, j) for i in range(c) for j in range(i + 1, c)]
    # The official selector sorts signed values descending, not |r|.
    pairs.sort(key=lambda pair: -float(corr[pair[0], pair[1]]))
    return set(pairs[:count])


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path, model, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    elif isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a model state_dict")
    if payload and all(key.startswith("module.") for key in payload):
        payload = {key[len("module."):]: value for key, value in payload.items()}
    model.load_state_dict(payload, strict=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(PROJECT / "configs" / "phase5_etth1.yaml"))
    p.add_argument("--checkpoint", default=str(PROJECT / "checkpoints" / "ETTh1_clustering_p96_s2021.pth"))
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--random_repeats", type=int, default=10)
    args = p.parse_args()
    config_path, checkpoint = Path(args.config).resolve(), Path(args.checkpoint).resolve()
    if not config_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Missing config or checkpoint: {}, {}".format(config_path, checkpoint))
    cfg = _load_config(config_path)
    cfg["root_path"] = str((ROOT / cfg["root_path"]).resolve())
    cfg["pred_len"], cfg["random_seed"] = args.pred_len, args.seed
    cfg["use_gpu"] = bool(torch.cuda.is_available() and args.device != "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    from argparse import Namespace
    import os
    os.chdir(ROOT)
    seed_all(args.seed)
    model_args = Namespace(**cfg)
    exp = Exp_Main(model_args)
    model = exp.model
    if args.device == "cpu":
        model = model.cpu()
        exp.device = torch.device("cpu")
    selector = ClusteringGraphSelector(
        n_clusters=model_args.n_clusters,
        keep_rate=model_args.subgraph_keep_rate,
        cross_edge_ratio=model_args.cross_edge_ratio,
        candidate_multiplier=model_args.candidate_multiplier,
        random_state=args.seed,
        batch_edge_assembly=getattr(model_args, "batch_edge_assembly", True),
    )
    attach_selector(model, selector)
    load_checkpoint(checkpoint, model, exp.device)
    model.eval()

    train_data, _ = data_provider(model_args, "train")
    test_data, _ = data_provider(model_args, "test")
    loader = DataLoader(IndexedDataset(test_data), batch_size=model_args.batch_size,
                        shuffle=False, num_workers=model_args.num_workers, drop_last=False)
    train_mean = torch.as_tensor(np.asarray(train_data.data_x).mean(axis=0),
                                 dtype=torch.float32, device=exp.device)

    # Capture the clustering graph actually consumed by Graph Generator.
    captured = []
    original_select = selector.select
    def capture(*a, **kw):
        edges = original_select(*a, **kw)
        captured.append((edges.detach().cpu().numpy(), [x.copy() for x in selector.last_labels]))
        return edges
    selector.select = capture
    predictions, targets = [], []
    labels_by_id, window_by_id, graph_by_id = {}, {}, {}
    with torch.inference_mode():
        for bx, by, _, _, ids in loader:
            x = bx.float().to(exp.device)
            y = by[:, -args.pred_len:, :].float().to(exp.device)
            selector.set_sample_ids(None)
            out = model(x)
            predictions.append(out.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
            edge_tensor, labels = captured[-1]
            windows = x.detach().cpu().numpy().transpose(0, 2, 1)
            for b, sid in enumerate(ids):
                sid = int(sid)
                labels_by_id[sid] = labels[b]
                window_by_id[sid] = windows[b]
                directed = edge_tensor[b]
                graph_by_id[sid] = {tuple(sorted((int(row[1]), int(row[2]))))
                                    for row in directed if int(row[1]) != int(row[2])}
    selector.select = original_select
    pred, truth = np.concatenate(predictions), np.concatenate(targets)
    base_mse_channels = ((pred.astype(np.float64) - truth.astype(np.float64)) ** 2).mean(axis=(0, 1))
    base_mse, base_mae = mse_mae(pred, truth)
    n, channels = pred.shape[0], pred.shape[-1]
    target_edges = int(model_args.subgraph_keep_rate * channels * (channels - 1) / 2)

    # Graph overlap and cluster-size patterns use only test history windows.
    jaccard_rows, size_counter = [], Counter()
    feature_names = ["mean", "std", "trend", "skewness", "q10", "q50", "q90", "autocorrelation_lag1"]
    aligned_centroids, aligned_counts = [], []
    for sid in range(n):
        window = window_by_id[sid]
        labels = labels_by_id[sid]
        sizes = tuple(sorted(np.bincount(labels, minlength=model_args.n_clusters).tolist(), reverse=True))
        size_counter[sizes] += 1
        baseline_edges = pearson_baseline_pairs(window, target_edges)
        cluster_edges = graph_by_id[sid]
        union = baseline_edges | cluster_edges
        jac = len(baseline_edges & cluster_edges) / len(union) if union else 1.0
        jaccard_rows.append({"sample_id": sid, "baseline_edges": len(baseline_edges),
                             "clustering_edges": len(cluster_edges), "intersection": len(baseline_edges & cluster_edges),
                             "union": len(union), "jaccard": jac})
        feats = extract_variable_features(window)
        # Cluster IDs are permutation-ambiguous; align by descending size then
        # smallest member index, rather than averaging local label IDs.
        groups = [np.flatnonzero(labels == g) for g in np.unique(labels)]
        groups.sort(key=lambda ix: (-len(ix), int(ix.min())))
        aligned_centroids.append([feats[ix].mean(axis=0) for ix in groups])
        aligned_counts.append([len(ix) for ix in groups])

    rng = np.random.default_rng(args.seed)
    masks = [("feature", str(i), (i,)) for i in range(channels)]
    random_masks = []
    for size in (1, 2, 4):
        for rep in range(args.random_repeats):
            chosen = tuple(sorted(int(i) for i in rng.choice(channels, size=size, replace=False)))
            random_masks.append(("random", "k{}_r{}_seed{}".format(size, rep + 1, args.seed), chosen))
    intervention_rows, channel_rows = [], []
    for kind, name, masked in masks + random_masks:
        out_parts = []
        masked_tensor = torch.as_tensor(masked, dtype=torch.long, device=exp.device)
        with torch.inference_mode():
            for bx, _, _, _, _ in loader:
                x = bx.float().to(exp.device).clone()
                x[:, :, masked_tensor] = train_mean[masked_tensor]
                selector.set_sample_ids(None)
                out_parts.append(model(x).detach().cpu().numpy())
        ablated = np.concatenate(out_parts)
        delta_by_channel = ((ablated.astype(np.float64) - truth.astype(np.float64)) ** 2).mean(axis=(0, 1)) - base_mse_channels
        rest = [c for c in range(channels) if c not in masked]
        amse, amae = mse_mae(ablated, truth)
        intervention_rows.append({
            "kind": kind, "name": name, "masked_indices": ";".join(map(str, masked)),
            "masked_count": len(masked), "baseline_mse": base_mse, "ablated_mse": amse,
            "delta_mse": amse - base_mse, "baseline_mae": base_mae, "ablated_mae": amae,
            "delta_mae": amae - base_mae,
            "mean_delta_mse_masked_channels": float(delta_by_channel[list(masked)].mean()),
            "mean_delta_mse_unmasked_channels": float(delta_by_channel[rest].mean()) if rest else float("nan"),
            "n_test_windows": n,
        })
        for c, delta in enumerate(delta_by_channel):
            channel_rows.append({"kind": kind, "name": name, "masked_indices": ";".join(map(str, masked)),
                                 "channel_index": c, "baseline_channel_mse": float(base_mse_channels[c]),
                                 "ablated_channel_mse": float(base_mse_channels[c] + delta),
                                 "delta_channel_mse": float(delta), "is_masked_channel": int(c in masked)})

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    outdir = XAI / "results" / "xai_controls_ETTh1_p{}_s{}_{}".format(args.pred_len, args.seed, stamp)
    outdir.mkdir(parents=True, exist_ok=False)
    write_csv(outdir / "jaccard_by_window.csv", jaccard_rows)
    write_csv(outdir / "ablation_controls.csv", intervention_rows)
    write_csv(outdir / "per_channel_mse_deltas.csv", channel_rows)
    size_rows = [{"cluster_sizes_desc": ";".join(map(str, size)), "windows": count,
                  "fraction": count / n} for size, count in sorted(size_counter.items(), key=lambda kv: (-kv[1], kv[0]))]
    write_csv(outdir / "cluster_size_distribution.csv", size_rows)
    centroids = np.asarray(aligned_centroids, dtype=np.float64)
    centroid_rows = []
    for rank in range(centroids.shape[1]):
        sizes_rank = np.asarray([row[rank] for row in aligned_counts])
        row = {"cluster_rank_by_size": rank + 1, "mean_cluster_size": float(sizes_rank.mean()),
               "singleton_windows": int(np.sum(sizes_rank == 1))}
        row.update({feature_names[j]: float(centroids[:, rank, j].mean()) for j in range(len(feature_names))})
        centroid_rows.append(row)
    write_csv(outdir / "cluster_feature_centroids_size_aligned.csv", centroid_rows)
    meta = {"dataset": "ETTh1", "pred_len": args.pred_len, "seed": args.seed,
            "test_windows": n, "baseline_mse": base_mse, "baseline_mae": base_mae,
            "channels": channels, "n_clusters": model_args.n_clusters,
            "target_undirected_edges_per_graph": target_edges,
            "baseline_edge_rule": "official top_k_channel_pairs_choose: signed Pearson descending, upper-triangle pairs",
            "ablation_value": "training-set mean in scaled input space",
            "checkpoint": str(checkpoint), "config": str(config_path),
            "jaccard_mean": float(np.mean([r["jaccard"] for r in jaccard_rows])),
            "jaccard_std": float(np.std([r["jaccard"] for r in jaccard_rows]))}
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("OUTPUT_DIR={}".format(outdir))
    print(json.dumps(meta, indent=2))
    print("CLUSTER_SIZE_DISTRIBUTION={}".format(size_rows))
    print("ALIGNED_CENTROIDS={}".format(centroid_rows))
    print("FEATURE_ABLATION={}".format([r for r in intervention_rows if r["kind"] == "feature"]))
    for size in (1, 2, 4):
        rows = [r for r in intervention_rows if r["kind"] == "random" and r["masked_count"] == size]
        print("RANDOM_SIZE_{} mean_delta_mse={:.8f} std_delta_mse={:.8f} mean_masked={:.8f} mean_unmasked={:.8f}".format(
            size, np.mean([r["delta_mse"] for r in rows]), np.std([r["delta_mse"] for r in rows]),
            np.mean([r["mean_delta_mse_masked_channels"] for r in rows]),
            np.mean([r["mean_delta_mse_unmasked_channels"] for r in rows])))


if __name__ == "__main__":
    main()
