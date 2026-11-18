"""Small, isolated Clustering-TVCFormer run using the original model stack."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR.parents[1]
OFFICIAL = ROOT / "official"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from FinalProject.Clustering_TVC.clustering_utils import ClusteringGraphSelector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", default=str(PROJECT_DIR / "configs" / "smoke_etth1.yaml"))
    parser.add_argument("--n_clusters", type=int)
    parser.add_argument("--cross_edge_ratio", type=float)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_train_batches", type=int, default=1)
    parser.add_argument("--max_eval_batches", type=int, default=1)
    args = parser.parse_args()
    with open(args.config_yaml, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key in ("n_clusters", "cross_edge_ratio", "random_seed", "batch_size"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config.update(max_train_batches=args.max_train_batches,
                  max_eval_batches=args.max_eval_batches,
                  config_yaml=args.config_yaml)
    return argparse.Namespace(**config)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import dgl
    dgl.random.seed(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def attach_selector(model, selector):
    root = model.module if isinstance(model, torch.nn.DataParallel) else model
    found = 0
    for name in ("model", "model_res", "model_trend"):
        backbone = getattr(root, name, None)
        if backbone is not None and hasattr(backbone, "graph_selector"):
            backbone.graph_selector = selector
            found += 1
    if found == 0:
        raise RuntimeError("the existing TVCFormer backbone has no graph_selector hook")


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run():
    args = parse_args()
    os.chdir(ROOT)
    if args.n_clusters < 1:
        raise ValueError("n_clusters must be positive; values above the channel count are clamped")
    if not 0.0 <= args.cross_edge_ratio <= 1.0:
        raise ValueError("cross_edge_ratio must be in [0, 1]")
    if args.max_train_batches < 1 or args.max_eval_batches < 1:
        raise ValueError("batch limits must be positive")
    if not os.path.isabs(args.root_path):
        args.root_path = str((ROOT / args.root_path).resolve())
    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    seed_all(args.random_seed)

    run_id = "{}_{}_{}_K{}_s{}_{}".format(
        args.data, args.pred_len, datetime.now().strftime("%Y%m%dT%H%M%S%f"),
        args.n_clusters, args.random_seed, os.getpid())
    results_dir = PROJECT_DIR / "results"
    figures_dir = PROJECT_DIR / "figures"
    log_dir = PROJECT_DIR / "logs" / "smoke"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (run_id + ".log")

    # Redirect all run output to a unique project log. Failed runs remain marked failed.
    import contextlib
    with log_path.open("x", encoding="utf-8") as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                print("run_id:", run_id)
                print("config:", json.dumps(vars(args), default=str, sort_keys=True))
                if not Path(args.root_path, args.data_path).is_file():
                    raise FileNotFoundError(Path(args.root_path, args.data_path))
                train_data, train_loader = data_provider(args, "train")
                _, test_loader = data_provider(args, "test")
                experiment = Exp_Main(args)
                selector = ClusteringGraphSelector(
                    n_clusters=args.n_clusters,
                    keep_rate=args.subgraph_keep_rate,
                    cross_edge_ratio=args.cross_edge_ratio,
                    candidate_multiplier=args.candidate_multiplier,
                    random_state=args.random_seed,
                )
                attach_selector(experiment.model, selector)
                graph_stats = {"calls": 0, "nodes": 0, "edges": 0,
                               "intra_edges": 0, "inter_edges": 0,
                               "construction_seconds": 0.0, "asymmetric_edges": 0,
                               "asymmetric_examples": []}
                root_model = experiment.model.module if isinstance(experiment.model, torch.nn.DataParallel) else experiment.model
                backbones = [getattr(root_model, n, None) for n in ("model", "model_res", "model_trend")]
                graph_generators = [b.graph_generator for b in backbones if b is not None and hasattr(b, "graph_generator")]
                if not graph_generators:
                    raise RuntimeError("no graph generator found")

                for generator in graph_generators:
                    timer = {"start": None}
                    def before(module, inputs, timer=timer):
                        sync(experiment.device)
                        timer["start"] = time.perf_counter()
                    def after(module, inputs, output, timer=timer):
                        sync(experiment.device)
                        graph_stats["construction_seconds"] += time.perf_counter() - timer["start"]
                        indices, weights = output[0], output[1]
                        if indices.ndim != 2 or indices.shape[1] != 2:
                            raise RuntimeError("GraphGenerator COO indices must have shape [edges, 2]")
                        if weights.ndim != 1 or weights.shape[0] != indices.shape[0]:
                            raise RuntimeError("GraphGenerator edge weights do not match COO indices")
                        graph_stats["calls"] += 1
                        graph_stats["edges"] += int(indices.shape[0])
                        batch = inputs[0].shape[0]
                        channels = args.enc_in
                        nodes_per_sample = channels * args.time_node_num
                        graph_stats["nodes"] += batch * nodes_per_sample
                        if indices.numel():
                            src, dst = indices[:, 0], indices[:, 1]
                            batch_idx = torch.div(src, nodes_per_sample, rounding_mode="floor")
                            src_ch = torch.remainder(src, nodes_per_sample) // args.time_node_num
                            dst_ch = torch.remainder(dst, nodes_per_sample) // args.time_node_num
                            # Recover the per-sample K-Means labels recorded by selector.
                            # Pair relation is counted at the graph COO level (time-node edges).
                            labels = selector.last_labels
                            same = torch.as_tensor(labels, device=src.device)[batch_idx, src_ch] == torch.as_tensor(labels, device=src.device)[batch_idx, dst_ch]
                            graph_stats["intra_edges"] += int(same.sum().item())
                            graph_stats["inter_edges"] += int((~same).sum().item())
                            edge_map = {}
                            for index, weight in zip(indices.detach().cpu().tolist(), weights.detach().cpu().tolist()):
                                edge_map[(index[0], index[1])] = weight
                            # Convolution/InstanceNorm can differ by a few float32 ulps
                            # across transposed positions while preserving symmetry.
                            tolerance = 1e-4
                            asymmetric = [
                                ((src_i, dst_i), weight, edge_map.get((dst_i, src_i)))
                                for (src_i, dst_i), weight in edge_map.items()
                                if (dst_i, src_i) not in edge_map or abs(weight - edge_map[(dst_i, src_i)]) > tolerance
                            ]
                            graph_stats["asymmetric_edges"] += len(asymmetric)
                            graph_stats["asymmetric_examples"].extend(asymmetric[:3])
                    generator.register_forward_pre_hook(before)
                    generator.register_forward_hook(after)

                optimizer = torch.optim.Adam(experiment.model.parameters(), lr=args.learning_rate)
                experiment.model.train()
                if experiment.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(experiment.device)
                sync(experiment.device)
                train_start = time.perf_counter()
                trained_batches = 0
                for batch_x, batch_y, _, _ in train_loader:
                    batch_x = batch_x.float().to(experiment.device)
                    batch_y = batch_y.float().to(experiment.device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = experiment.model(batch_x)
                    if tuple(prediction.shape) != (batch_x.shape[0], args.pred_len, args.enc_in):
                        raise RuntimeError("model output shape does not match [batch, pred_len, channels]")
                    target = batch_y[:, -args.pred_len:, :]
                    loss = torch.nn.functional.mse_loss(prediction, target)
                    contrastive = getattr(experiment.model, "contrastive_loss", None)
                    if contrastive is not None:
                        loss = loss + getattr(args, "sdei_lambda", 0.0) * contrastive
                    loss.backward()
                    if not any(parameter.grad is not None for parameter in experiment.model.parameters()
                               if parameter.requires_grad):
                        raise RuntimeError("backward produced no model parameter gradients")
                    optimizer.step()
                    trained_batches += 1
                    if trained_batches >= args.max_train_batches:
                        break
                sync(experiment.device)
                training_seconds = time.perf_counter() - train_start
                if trained_batches == 0:
                    raise RuntimeError("training loader produced no batch")

                experiment.model.eval()
                predictions, truths = [], []
                with torch.no_grad():
                    for batch_x, batch_y, _, _ in test_loader:
                        batch_x = batch_x.float().to(experiment.device)
                        batch_y = batch_y.float().to(experiment.device)
                        output = experiment.model(batch_x)
                        truth = batch_y[:, -args.pred_len:, :]
                        predictions.append(output.detach().cpu().numpy())
                        truths.append(truth.detach().cpu().numpy())
                        if len(predictions) >= args.max_eval_batches:
                            break
                if not predictions:
                    raise RuntimeError("test loader produced no batch")
                prediction = np.concatenate(predictions, axis=0)
                truth = np.concatenate(truths, axis=0)
                mse = float(np.mean((prediction - truth) ** 2))
                mae = float(np.mean(np.abs(prediction - truth)))
                peak_gpu_mb = (torch.cuda.max_memory_allocated(experiment.device) / (1024 ** 2)
                               if experiment.device.type == "cuda" else 0.0)

                stats = selector.stats
                if stats.pearson_pairs <= 0:
                    raise RuntimeError("no Pearson pairs were actually evaluated")
                if stats.pearson_pairs >= stats.full_pairs:
                    raise RuntimeError("candidate pruning did not reduce Pearson pair count")
                if graph_stats["calls"] == 0:
                    raise RuntimeError("graph generator hook observed no graph")
                if graph_stats["asymmetric_edges"]:
                    print("asymmetric examples:", graph_stats["asymmetric_examples"][:5])
                    raise RuntimeError("generated TVC graph is not symmetric")
                if graph_stats["edges"] == 0:
                    raise RuntimeError("graph generator returned no edges")

                config_snapshot = json.loads(json.dumps(vars(args), default=str))
                (results_dir / (run_id + "_config.yaml")).write_text(
                    yaml.safe_dump(config_snapshot, sort_keys=True), encoding="utf-8")
                append_csv(results_dir / "clustering_results.csv", {
                    "run_id": run_id, "dataset": args.data, "pred_len": args.pred_len,
                    "seed": args.random_seed, "n_clusters": args.n_clusters,
                    "cross_edge_ratio": args.cross_edge_ratio,
                    "eval_batches": len(predictions), "mse": mse, "mae": mae,
                    "status": "completed_smoke_run"})
                density = graph_stats["edges"] / max(1, graph_stats["nodes"] * (graph_stats["nodes"] - 1))
                append_csv(results_dir / "graph_statistics.csv", {
                    "run_id": run_id, "graph_calls": graph_stats["calls"],
                    "nodes_total": graph_stats["nodes"], "directed_edges_total": graph_stats["edges"],
                    "density": density, "intra_cluster_edges": graph_stats["intra_edges"],
                    "inter_cluster_edges": graph_stats["inter_edges"],
                    "asymmetric_edges": graph_stats["asymmetric_edges"]})
                append_csv(results_dir / "runtime_statistics.csv", {
                    "run_id": run_id, "selector_samples": stats.samples,
                    "pearson_pairs_computed": stats.pearson_pairs,
                    "full_pearson_pairs_equivalent": stats.full_pairs,
                    "pair_reduction_percent": 100.0 * (1.0 - stats.pearson_pairs / max(1, stats.full_pairs)),
                    "clustering_seconds": stats.cluster_seconds,
                    "pearson_seconds": stats.pearson_seconds,
                    "graph_construction_seconds": graph_stats["construction_seconds"],
                    "training_seconds": training_seconds, "peak_gpu_memory_mb": peak_gpu_mb})
                print("status: completed_smoke_run")
                print("pearson pairs computed/full equivalent:", stats.pearson_pairs, stats.full_pairs)
                print("graph nodes/edges/asymmetric:", graph_stats["nodes"], graph_stats["edges"], graph_stats["asymmetric_edges"])
                print("MSE:", mse, "MAE:", mae)
                print("train batches:", trained_batches, "eval batches:", len(predictions))
                log.flush()
            make_pair_figure(results_dir / "runtime_statistics.csv", figures_dir / "pearson_pair_reduction.png")
            print("Completed:", run_id)
            print("MSE={:.6f} MAE={:.6f}".format(mse, mae))
            print("Log:", log_path)
        except Exception:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                print("status: failed")
                traceback.print_exc()
                log.flush()
            raise


def make_pair_figure(csv_path, output_path):
    import csv
    import matplotlib.pyplot as plt
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    current = rows[-1]
    baseline_per_sample = float(current["full_pearson_pairs_equivalent"]) / max(1, int(current["selector_samples"]))
    selected_per_sample = float(current["pearson_pairs_computed"]) / max(1, int(current["selector_samples"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Full Pearson\n(pair count)", "Clustering\n(actual count)"],
           [baseline_per_sample, selected_per_sample], color=["#8a8f98", "#2f78b7"])
    ax.set_ylabel("Pearson pairs per sample")
    ax.set_title("ETTh1 smoke run: measured Pearson pair count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
