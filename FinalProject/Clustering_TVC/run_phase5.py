"""Sequential, full ETTh1 baseline-vs-clustering experiment runner."""

import argparse
import copy
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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR.parents[1]
OFFICIAL = ROOT / "official"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from utils.tools import adjust_learning_rate
from FinalProject.Clustering_TVC.clustering_utils import ClusteringGraphSelector


CSV_FIELDS = [
    "dataset", "model", "pred_len", "seed", "test_mse", "test_mae",
    "best_epoch", "training_time", "train_loop_time", "validation_time",
    "epoch_time", "eval_time", "clustering_time",
    "pearson_time", "graph_time", "pearson_pairs", "pearson_reduction_pct",
    "peak_gpu_memory", "status", "log_file",
    "pearson_scalar_evaluations", "full_pearson_pair_equivalent",
    "test_samples", "train_epochs_completed", "early_stopped", "graph_calls",
    "graph_nodes", "graph_edges", "config_file", "failure_reason",
    "baseline_pearson_pairs",
    "precompute_time", "feature_time", "kmeans_time", "candidate_time",
    "cache_lookup_time", "transfer_time", "kmeans_calls", "cache_hits",
    "cache_misses", "total_time",
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class IndexedDataset(Dataset):
    """Add a stable window ID without changing the underlying dataset samples."""
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index] + (index,)


def indexed_loader(original_loader, split):
    return DataLoader(
        IndexedDataset(original_loader.dataset, split),
        batch_sampler=original_loader.batch_sampler,
        num_workers=original_loader.num_workers,
        pin_memory=getattr(original_loader, "pin_memory", False),
    )


def precompute_graph_selections(selector, datasets, device, implementation="batched"):
    """Precompute from each dataset's encoder input windows only."""
    if implementation not in ("sequential", "batched"):
        raise ValueError("precompute_implementation must be sequential or batched")
    started = time.perf_counter()
    for split, dataset in datasets:
        if implementation == "sequential":
            for index in range(len(dataset)):
                history = dataset.data_x[index:index + dataset.seq_len]
                if len(history) != dataset.seq_len:
                    raise RuntimeError("dataset history window has unexpected length")
                selector.prepare_indexed_window((split, index), history)
            continue
        batch_size = 128
        for start in range(0, len(dataset), batch_size):
            indices = range(start, min(start + batch_size, len(dataset)))
            # Each tensor contains only the history slice from this sample;
            # target/future values are never presented to the selector.
            histories = []
            sample_ids = []
            for index in indices:
                history = dataset.data_x[index:index + dataset.seq_len]
                if len(history) != dataset.seq_len:
                    raise RuntimeError("dataset history window has unexpected length")
                histories.append(history)
                sample_ids.append((split, index))
            selector.prepare_indexed_windows_batch(sample_ids, np.stack(histories, axis=0))
    sync(device)
    return time.perf_counter() - started

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import dgl
    dgl.random.seed(seed)


def append_result(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", newline="", encoding="utf-8") as source:
            existing_fields = next(csv.reader(source), [])
        if existing_fields != CSV_FIELDS:
            raise ValueError("result CSV schema mismatch; use a new result filename")
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def attach_selector(model, selector):
    root_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    attached = []
    for name in ("model", "model_res", "model_trend"):
        backbone = getattr(root_model, name, None)
        if backbone is not None and hasattr(backbone, "graph_selector"):
            backbone.graph_selector = selector
            attached.append(backbone)
    if not attached:
        raise RuntimeError("TVCFormer backbone does not expose graph_selector")
    return attached


def _load_config(path):
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["data"] = config.pop("dataset", "ETTh1")
    config.setdefault("model", "TVCFormer")
    config.setdefault("random_seed", 2021)
    config.setdefault("use_gpu", True)
    config.setdefault("gpu", 0)
    config.setdefault("use_multi_gpu", False)
    config.setdefault("devices", "0")
    config.setdefault("model_id", "phase5_etth1")
    config.setdefault("is_training", 1)
    config.setdefault("individual", False)
    config.setdefault("sdei", 0)
    config.setdefault("sdei_lambda", 0.0)
    config.setdefault("sdei_shape", 1)
    config.setdefault("sdei_gate_init", -3.0)
    config.setdefault("sdei_gate_fixed", 0)
    config.setdefault("graph_gate", 0)
    config.setdefault("last_value_residual", 0)
    config.setdefault("a_ge", 0)
    config.setdefault("na_head", 0)
    config.setdefault("rank", 32)
    config.setdefault("gate_init", -1.0)
    config.setdefault("lr_gate_factor", 1.0)
    config.setdefault("config_yaml", str(path))
    return config


def _create_hooks(experiment, model_name, selector, args):
    stats = {
        "graph_seconds": 0.0, "graph_calls": 0, "graph_nodes": 0,
        "graph_edges": 0, "pearson_seconds": 0.0,
        "pearson_pairs": 0, "pearson_scalar_evaluations": 0,
        "full_pearson_pair_equivalent": 0,
    }
    device = experiment.device
    root_model = experiment.model.module if isinstance(experiment.model, torch.nn.DataParallel) else experiment.model
    graph_modules = []
    for name in ("model", "model_res", "model_trend"):
        backbone = getattr(root_model, name, None)
        if backbone is not None and hasattr(backbone, "graph_generator"):
            graph_modules.append(backbone.graph_generator)
    if not graph_modules:
        raise RuntimeError("no GraphGenerator was found")

    handles = []
    for graph_module in graph_modules:
        timer = {"start": None}

        def before(module, inputs, timer=timer):
            sync(device)
            timer["start"] = time.perf_counter()

        def after(module, inputs, output, timer=timer):
            sync(device)
            stats["graph_seconds"] += time.perf_counter() - timer["start"]
            indices, values = output[0], output[1]
            if indices.ndim != 2 or indices.shape[1] != 2 or values.ndim != 1 or len(values) != len(indices):
                raise RuntimeError("GraphGenerator returned invalid COO indices/weights")
            stats["graph_calls"] += 1
            stats["graph_nodes"] += inputs[0].shape[0] * args.enc_in * args.time_node_num
            stats["graph_edges"] += int(indices.shape[0])

        handles.append(graph_module.register_forward_pre_hook(before))
        handles.append(graph_module.register_forward_hook(after))

    if model_name == "baseline":
        backbone_module = __import__("layers.TVCFormer_backbone", fromlist=["module"])
        original = backbone_module.get_pearson_correlation_vector

        def timed_full_pearson(values, *call_args, **call_kwargs):
            sync(values.device)
            start = time.perf_counter()
            result = original(values, *call_args, **call_kwargs)
            sync(values.device)
            stats["pearson_seconds"] += time.perf_counter() - start
            batch, channels, _ = values.shape
            stats["pearson_pairs"] += batch * channels * (channels - 1) // 2
            stats["full_pearson_pair_equivalent"] += batch * channels * (channels - 1) // 2
            stats["pearson_scalar_evaluations"] += batch * channels * channels
            return result

        backbone_module.get_pearson_correlation_vector = timed_full_pearson
        handles.append((backbone_module, original))
    else:
        if selector is None:
            raise RuntimeError("clustering run requires a selector")
        # Selector counters time actual candidate scoring; derive scalar workload
        # in unique undirected pair-equivalents, matching the baseline comparison.
        handles.append(("selector_stats", stats, selector.stats))
    return stats, handles


def _remove_hooks(handles, model_name):
    for handle in handles:
        if isinstance(handle, tuple) and handle and handle[0] == "selector_stats":
            _, stats, selector_stats = handle
            stats["pearson_seconds"] = selector_stats.pearson_seconds
            stats["pearson_pairs"] = selector_stats.pearson_pairs
            stats["full_pearson_pair_equivalent"] = selector_stats.full_pairs
            # One Pearson scalar is produced for each unique candidate pair.
            stats["pearson_scalar_evaluations"] = selector_stats.pearson_pairs
        elif isinstance(handle, tuple):
            module, original = handle
            module.get_pearson_correlation_vector = original
        else:
            handle.remove()


def _train_and_evaluate(config, model_name, pred_len, seed, log):
    if config.get("use_gpu", True) and torch.cuda.is_available():
        torch.cuda.synchronize()
    total_start = time.perf_counter()
    args_dict = dict(config)
    args_dict["pred_len"] = pred_len
    args_dict["random_seed"] = seed
    args_dict["use_gpu"] = bool(args_dict["use_gpu"] and torch.cuda.is_available())
    args = argparse.Namespace(**args_dict)
    if model_name == "clustering" and any((
        getattr(args, "decomposition", False),
        getattr(args, "last_value_residual", 0),
        getattr(args, "na_head", 0),
    )):
        raise ValueError(
            "indexed history precompute requires the phase-5 raw-input backbone "
            "(no decomposition, last-value residual, or anchored head)"
        )
    seed_all(seed)

    print("Hardware:", torch.cuda.get_device_name(0) if args.use_gpu else "CPU")
    print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("Run config:", json.dumps(args_dict, default=str, sort_keys=True))

    experiment = Exp_Main(args)
    selector = None
    if model_name == "clustering":
        selector = ClusteringGraphSelector(
            n_clusters=args.n_clusters,
            keep_rate=args.subgraph_keep_rate,
            cross_edge_ratio=args.cross_edge_ratio,
            candidate_multiplier=args.candidate_multiplier,
            random_state=seed,
            batch_edge_assembly=getattr(args, "batch_edge_assembly", True),
        )
        attach_selector(experiment.model, selector)

    train_data, original_train_loader = data_provider(args, "train")
    val_data, original_val_loader = data_provider(args, "val")
    test_data, _ = data_provider(args, "test")
    train_loader = indexed_loader(original_train_loader, "train")
    val_loader = indexed_loader(original_val_loader, "val")
    # Full test-set evaluation includes the final partial batch, unlike the
    # repository helper which drops it. It remains sequential and unshuffled.
    test_loader = DataLoader(IndexedDataset(test_data, "test"), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, drop_last=False)

    precompute_time = 0.0
    if selector is not None:
        precompute_time = precompute_graph_selections(
            selector,
            (("train", train_data), ("val", val_data), ("test", test_data)),
            experiment.device,
            implementation=getattr(args, "precompute_implementation", "batched"),
        )
        print("precompute_seconds={:.3f} unique_history_windows={} kmeans_calls={}".format(
            precompute_time, selector.stats.cache_misses, selector.stats.kmeans_calls))

    stats, hook_handles = _create_hooks(experiment, model_name, selector, args)
    model = experiment.model
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=getattr(args, "weight_decay", 0.0))
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer, steps_per_epoch=len(train_loader),
        pct_start=args.pct_start, epochs=args.train_epochs,
        max_lr=args.learning_rate,
    )

    if experiment.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(experiment.device)
    epoch_seconds = []
    best_val = float("inf")
    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    patience_count = 0
    early_stopped = False
    sync(experiment.device)
    training_start = time.perf_counter()
    train_loop_seconds = 0.0
    validation_seconds = 0.0
    try:
        for epoch in range(1, args.train_epochs + 1):
            sync(experiment.device)
            epoch_start = time.perf_counter()
            model.train()
            train_losses = []
            sync(experiment.device)
            train_loop_start = time.perf_counter()
            for batch_x, batch_y, _, _, sample_indices in train_loader:
                if selector is not None:
                    selector.set_sample_ids([("train", int(index)) for index in sample_indices.tolist()])
                batch_x = batch_x.float().to(experiment.device)
                batch_y = batch_y.float().to(experiment.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch_x)
                target = batch_y[:, -pred_len:, :]
                loss = criterion(prediction, target)
                contrastive = getattr(model, "contrastive_loss", None)
                if contrastive is not None:
                    loss = loss + getattr(args, "sdei_lambda", 0.0) * contrastive
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss at epoch {}".format(epoch))
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in model.parameters() if p.requires_grad):
                    raise FloatingPointError("non-finite gradient at epoch {}".format(epoch))
                optimizer.step()
                train_losses.append(float(loss.detach().item()))

            sync(experiment.device)
            train_loop_seconds += time.perf_counter() - train_loop_start

            model.eval()
            val_losses = []
            val_maes = []
            sync(experiment.device)
            validation_start = time.perf_counter()
            with torch.no_grad():
                for batch_x, batch_y, _, _, sample_indices in val_loader:
                    if selector is not None:
                        selector.set_sample_ids([("val", int(index)) for index in sample_indices.tolist()])
                    batch_x = batch_x.float().to(experiment.device)
                    batch_y = batch_y.float().to(experiment.device)
                    prediction = model(batch_x)
                    target = batch_y[:, -pred_len:, :]
                    val_losses.append(float(criterion(prediction, target).item()))
                    val_maes.append(float(torch.mean(torch.abs(prediction - target)).item()))
            sync(experiment.device)
            validation_seconds += time.perf_counter() - validation_start
            if not val_losses:
                raise RuntimeError("validation loader produced no batches")
            val_loss = float(np.mean(val_losses))
            val_mae = float(np.mean(val_maes))
            if not np.isfinite(val_loss):
                raise FloatingPointError("non-finite validation loss at epoch {}".format(epoch))
            if not np.isfinite(val_mae):
                raise FloatingPointError("non-finite validation MAE at epoch {}".format(epoch))

            sync(experiment.device)
            elapsed_epoch = time.perf_counter() - epoch_start
            epoch_seconds.append(elapsed_epoch)
            print("epoch={}/{} train_mse={:.7f} val_mse={:.7f} epoch_seconds={:.3f}".format(
                epoch, args.train_epochs, float(np.mean(train_losses)), val_loss, elapsed_epoch))

            if val_loss < best_val:
                best_val = val_loss
                best_val_mae = val_mae
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                print("early_stopping_counter={}/{}".format(patience_count, args.patience))
                if patience_count >= args.patience:
                    early_stopped = True
                    print("early_stopping=true at epoch {}".format(epoch))
                    break

            if args.lradj != "TST":
                adjust_learning_rate(optimizer, scheduler, epoch, args, printout=False)
            else:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = scheduler.get_last_lr()[0]
                scheduler.step()

        if best_state is None:
            raise RuntimeError("training did not produce a best validation checkpoint")
        model.load_state_dict(best_state)
        sync(experiment.device)
        training_time = time.perf_counter() - training_start

        eval_start = time.perf_counter()
        model.eval()
        predictions, truths = [], []
        with torch.no_grad():
            for batch_x, batch_y, _, _, sample_indices in test_loader:
                if selector is not None:
                    selector.set_sample_ids([("test", int(index)) for index in sample_indices.tolist()])
                batch_x = batch_x.float().to(experiment.device)
                batch_y = batch_y.float().to(experiment.device)
                prediction = model(batch_x)
                truth = batch_y[:, -pred_len:, :]
                predictions.append(prediction.detach().cpu().numpy())
                truths.append(truth.detach().cpu().numpy())
        pred = np.concatenate(predictions, axis=0)
        true = np.concatenate(truths, axis=0)
        if pred.shape != true.shape or pred.shape[0] != len(test_data):
            raise RuntimeError("test evaluation did not cover the complete dataset")
        test_mse = float(np.mean((pred - true) ** 2))
        test_mae = float(np.mean(np.abs(pred - true)))
        sync(experiment.device)
        eval_time = time.perf_counter() - eval_start
        if not np.isfinite(test_mse) or not np.isfinite(test_mae):
            raise FloatingPointError("non-finite test metric")
        peak_memory = (torch.cuda.max_memory_allocated(experiment.device) / (1024 ** 2)
                       if experiment.device.type == "cuda" else 0.0)
        sync(experiment.device)
        total_time = time.perf_counter() - total_start

        if model_name == "clustering":
            _remove_hooks(hook_handles, model_name)
            hook_handles = []
            clustering_time = selector.stats.cluster_seconds
            if selector.stats.pearson_pairs != stats["pearson_pairs"]:
                raise RuntimeError("selector Pearson counter mismatch")
        else:
            _remove_hooks(hook_handles, model_name)
            hook_handles = []
            clustering_time = 0.0
        if stats["pearson_pairs"] <= 0 or stats["graph_calls"] <= 0:
            raise RuntimeError("Pearson or graph instrumentation observed no work")
        if model_name == "clustering" and stats["pearson_pairs"] >= stats["full_pearson_pair_equivalent"]:
            raise RuntimeError("clustering did not reduce Pearson pair count")

        result = {
            "dataset": args.data, "model": model_name, "pred_len": pred_len,
            "seed": seed, "validation_mse": best_val, "validation_mae": best_val_mae,
            "test_mse": test_mse, "test_mae": test_mae,
            "best_epoch": best_epoch, "training_time": training_time,
            "train_loop_time": train_loop_seconds,
            "validation_time": validation_seconds,
            "epoch_time": float(np.mean(epoch_seconds)) if epoch_seconds else 0.0,
            "eval_time": eval_time,
            "clustering_time": clustering_time,
            "pearson_time": stats["pearson_seconds"],
            "graph_time": stats["graph_seconds"],
            "pearson_pairs": stats["pearson_pairs"],
            "pearson_reduction_pct": "", # filled after pairing with baseline
            "peak_gpu_memory": peak_memory, "status": "completed",
            "pearson_scalar_evaluations": stats["pearson_scalar_evaluations"],
            "full_pearson_pair_equivalent": stats["full_pearson_pair_equivalent"],
            "test_samples": int(pred.shape[0]),
            "train_epochs_completed": len(epoch_seconds),
            "early_stopped": early_stopped, "graph_calls": stats["graph_calls"],
            "graph_nodes": stats["graph_nodes"], "graph_edges": stats["graph_edges"],
            "precompute_time": precompute_time,
            "feature_time": selector.stats.feature_seconds if selector is not None else 0.0,
            "kmeans_time": selector.stats.kmeans_seconds if selector is not None else 0.0,
            "candidate_time": selector.stats.candidate_seconds if selector is not None else 0.0,
            "cache_lookup_time": selector.stats.cache_seconds if selector is not None else 0.0,
            "transfer_time": selector.stats.transfer_seconds if selector is not None else 0.0,
            "kmeans_calls": selector.stats.kmeans_calls if selector is not None else 0,
            "cache_hits": selector.stats.cache_hits if selector is not None else 0,
            "cache_misses": selector.stats.cache_misses if selector is not None else 0,
            "total_time": total_time,
        }
        print("test_samples={} test_mse={:.8f} test_mae={:.8f}".format(
            result["test_samples"], test_mse, test_mae))
        print("best_validation_mse={:.8f} best_validation_mae={:.8f}".format(
            best_val, best_val_mae))
        print("best_epoch={} early_stopped={} training_seconds={:.3f} train_loop_seconds={:.3f} validation_seconds={:.3f} test_seconds={:.3f} mean_epoch_seconds={:.3f}".format(
            best_epoch, early_stopped, training_time, train_loop_seconds,
            validation_seconds, eval_time, result["epoch_time"]))
        print("pearson_pairs={} full_pair_equivalent={} scalar_evaluations={}".format(
            stats["pearson_pairs"], stats["full_pearson_pair_equivalent"], stats["pearson_scalar_evaluations"]))
        if selector is not None:
            print("clustering_cache_hits={} clustering_cache_misses={} kmeans_calls={}".format(
                selector.stats.cache_hits, selector.stats.cache_misses, selector.stats.kmeans_calls))
            print("feature_seconds={:.3f} kmeans_seconds={:.3f} candidate_seconds={:.3f} cache_seconds={:.3f} transfer_seconds={:.3f} precompute_seconds={:.3f} total_seconds={:.3f}".format(
                selector.stats.feature_seconds, selector.stats.kmeans_seconds,
                selector.stats.candidate_seconds, selector.stats.cache_seconds,
                selector.stats.transfer_seconds, precompute_time, total_time))
        print("clustering_seconds={:.3f} pearson_seconds={:.3f} graph_seconds={:.3f} peak_gpu_mb={:.3f}".format(
            clustering_time, stats["pearson_seconds"], stats["graph_seconds"], peak_memory))
        return result
    finally:
        if hook_handles:
            _remove_hooks(hook_handles, model_name)


def run_one(config, pred_len, model_name, seed, result_csv, log_dir, results_dir):
    run_id = "ETTh1_p{}_{}_s{}_{}".format(
        pred_len, model_name, seed, datetime.now().strftime("%Y%m%dT%H%M%S%f"))
    log_path = log_dir / (run_id + ".log")
    config_path = results_dir / (run_id + "_config.yaml")
    run_config = dict(config, pred_len=pred_len, random_seed=seed, model_name=model_name)
    config_path.write_text(yaml.safe_dump(run_config, sort_keys=True), encoding="utf-8")
    row = {"dataset": "ETTh1", "model": model_name, "pred_len": pred_len,
           "seed": seed, "status": "failed", "log_file": str(log_path.relative_to(ROOT)),
           "config_file": str(config_path.relative_to(ROOT)), "failure_reason": ""}
    log_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        tee = Tee(sys.__stdout__, log)
        try:
            import contextlib
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                print("run_id:", run_id)
                print("start_time:", datetime.now().isoformat(timespec="seconds"))
                print("config_file:", config_path)
                result = _train_and_evaluate(config, model_name, pred_len, seed, log)
                row.update(result)
                if model_name == "baseline":
                    row["pearson_reduction_pct"] = 0.0
                    row["baseline_pearson_pairs"] = result["pearson_pairs"]
                else:
                    prior_rows = []
                    if result_csv.is_file():
                        with result_csv.open("r", newline="", encoding="utf-8") as stream:
                            prior_rows = list(csv.DictReader(stream))
                    baseline_rows = [candidate for candidate in prior_rows
                                     if candidate.get("dataset") == "ETTh1"
                                     and candidate.get("model") == "baseline"
                                     and candidate.get("pred_len") == str(pred_len)
                                     and candidate.get("seed") == str(seed)
                                     and candidate.get("status") == "completed"]
                    # Optimized-only reruns may reference the verified baseline
                    # records from the original phase-5 CSV without rerunning it.
                    if not baseline_rows:
                        baseline_path = results_dir / "phase5_etth1_results.csv"
                        if baseline_path.resolve() != result_csv.resolve() and baseline_path.is_file():
                            with baseline_path.open("r", newline="", encoding="utf-8") as stream:
                                historic_rows = list(csv.DictReader(stream))
                            baseline_rows = [candidate for candidate in historic_rows
                                             if candidate.get("dataset") == "ETTh1"
                                             and candidate.get("model") == "baseline"
                                             and candidate.get("pred_len") == str(pred_len)
                                             and candidate.get("seed") == str(seed)
                                             and candidate.get("status") == "completed"]
                    if not baseline_rows:
                        raise RuntimeError("completed baseline row is missing for pair reduction")
                    baseline_pairs = int(baseline_rows[-1]["pearson_pairs"])
                    row["baseline_pearson_pairs"] = baseline_pairs
                    row["pearson_reduction_pct"] = (
                        1.0 - result["pearson_pairs"] / max(1, baseline_pairs)
                    ) * 100.0
                row["log_file"] = str(log_path.relative_to(ROOT))
                row["config_file"] = str(config_path.relative_to(ROOT))
                row["status"] = "completed"
                append_result(result_csv, row)
                print("status: completed")
                print("end_time:", datetime.now().isoformat(timespec="seconds"))
        except Exception as error:
            with __import__("contextlib").redirect_stdout(tee), __import__("contextlib").redirect_stderr(tee):
                row["failure_reason"] = "{}: {}".format(type(error).__name__, error)
                print("status: failed")
                traceback.print_exc()
                print("end_time:", datetime.now().isoformat(timespec="seconds"))
                append_result(result_csv, row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs" / "phase5_etth1.yaml"))
    parser.add_argument("--pred_lens", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--models", nargs="+", choices=("baseline", "clustering"),
                        default=("baseline", "clustering"))
    parser.add_argument("--result_csv", default="phase5_etth1_results.csv")
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument("--log_dir", default="")
    args = parser.parse_args()
    os.chdir(ROOT)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    config = _load_config(config_path)
    config["root_path"] = str((ROOT / config["root_path"]).resolve())
    if not Path(config["root_path"], config["data_path"]).is_file():
        raise FileNotFoundError(Path(config["root_path"], config["data_path"]))

    results_dir = PROJECT_DIR / "results"
    log_dir = (Path(args.log_dir).resolve() if args.log_dir else
               ROOT / "experiments" / "results" / "Clustering_TVC")
    if args.log_dir and not log_dir.is_relative_to(PROJECT_DIR.resolve()):
        raise ValueError("custom log_dir must be inside FinalProject/Clustering_TVC/")
    result_csv = results_dir / args.result_csv
    if result_csv.resolve().parent != results_dir.resolve():
        raise ValueError("--result_csv must be a filename inside FinalProject/Clustering_TVC/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    previous_rows = []
    if result_csv.is_file():
        with result_csv.open("r", newline="", encoding="utf-8") as stream:
            previous_rows = list(csv.DictReader(stream))
    completed_keys = {
        (row.get("dataset"), row.get("model"), row.get("pred_len"), row.get("seed"))
        for row in previous_rows if row.get("status") == "completed"
    }
    for pred_len in args.pred_lens:
        for model_name in args.models:
            key = ("ETTh1", model_name, str(pred_len), str(args.seed))
            if key in completed_keys:
                print("Skipping previously completed run:", key, flush=True)
                continue
            print("\n===== ETTh1 pred_len={} model={} seed={} =====".format(
                pred_len, model_name, args.seed), flush=True)
            row = run_one(config, pred_len, model_name, args.seed,
                          result_csv, log_dir, results_dir)
            if row.get("status") == "completed":
                completed_keys.add(key)
            print("run_status:", row["status"], "log:", row["log_file"], flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Analyze all recorded rows, including earlier phase-5 rows, without
    # overwriting any existing CSV/log. Only complete pairs are compared.
    if not args.skip_analysis:
        all_rows = list(csv.DictReader(result_csv.open("r", newline="", encoding="utf-8")))
        make_figures_and_report(all_rows, results_dir, PROJECT_DIR / "figures")


def make_figures_and_report(rows, results_dir, figures_dir):
    import matplotlib.pyplot as plt

    by_key = {}
    for row in rows:
        if row.get("status") == "completed":
            by_key.setdefault((row["dataset"], row["pred_len"], row["seed"]), {})[row["model"]] = row
    paired = [(key, models) for key, models in by_key.items()
              if "baseline" in models and "clustering" in models]
    figures_dir.mkdir(parents=True, exist_ok=True)
    if paired:
        horizons = [int(key[1]) for key, _ in sorted(paired, key=lambda item: int(item[0][1]))]
        metrics = [
            ("test_mse", "Test MSE", "phase5_etth1_mse.png"),
            ("test_mae", "Test MAE", "phase5_etth1_mae.png"),
            ("pearson_pairs", "Pearson pair evaluations", "phase5_etth1_pearson_pairs.png"),
            ("training_time", "Training time (seconds)", "phase5_etth1_training_time.png"),
            ("peak_gpu_memory", "Peak GPU memory (MiB)", "phase5_etth1_peak_gpu_memory.png"),
        ]
        for field, title, filename in metrics:
            fig, ax = plt.subplots(figsize=(7, 4))
            base_values, cluster_values = [], []
            for key, models in sorted(paired, key=lambda item: int(item[0][1])):
                base_values.append(float(models["baseline"][field]))
                cluster_values.append(float(models["clustering"][field]))
            ax.plot(horizons, base_values, marker="o", label="TVCFormer")
            ax.plot(horizons, cluster_values, marker="o", label="Clustering-TVCFormer")
            ax.set_xlabel("Prediction length")
            ax.set_ylabel(title)
            ax.set_title("ETTh1: " + title)
            ax.legend()
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures_dir / filename, dpi=160)
            plt.close(fig)

    report_path = results_dir / "phase5_etth1_comparison.md"
    lines = [
        "# Phase 5 – ETTh1 TVCFormer comparison", "",
        "## Experimental protocol", "",
        "Both modes use the same `configs/phase5_etth1.yaml`, seed, split, scaling, "
        "model architecture, optimizer (Adam), learning-rate policy (`type3`), "
        "batch size, epoch limit, and patience. Runs execute sequentially. The "
        "baseline leaves `graph_selector=None`; clustering attaches the per-window "
        "K-Means selector. Best weights are kept in RAM using validation MSE; no "
        "checkpoint is written outside this project folder.", "",
        "The clustering selector memoizes deterministic feature/label/candidate "
        "results by an exact hash of the observed history window. Repeated windows "
        "reuse K-Means output without changing selected candidates; Pearson is "
        "still recomputed on every forward. Cache hit/miss totals are recorded in "
        "each clustering run log. This optimization is included in measured "
        "clustering and training time.", "",
        "Each epoch timer covers train batches plus validation. `training_time` "
        "covers all train/validation epochs through best-weight restore and excludes "
        "the final test pass. Pearson and GraphGenerator timers include train, "
        "validation, and test forwards and synchronize CUDA before/after measured "
        "GPU sections. Peak memory is reset before training and sampled after full "
        "test evaluation. Test metrics use all windows, including the final partial "
        "batch; the repository test loader's `drop_last=True` is overridden only "
        "in this isolated runner.", "",
        "`pearson_pairs` counts unique undirected pair-equivalents. The baseline "
        "implementation also computes an ordered `C × C` tensor (including the "
        "diagonal); that scalar count is stored separately as "
        "`pearson_scalar_evaluations`. Clustering counts each candidate pair whose "
        "Pearson value is actually evaluated.", "",
        "## Results", "",
    ]
    if not paired:
        lines += ["No complete baseline/clustering horizon pair is currently available.", ""]
    else:
        lines += [
            "| pred_len | baseline MSE | clustering MSE | MSE change % | baseline MAE | clustering MAE | MAE change % | pair reduction % | training time change % | peak memory change % |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key, models in sorted(paired, key=lambda item: int(item[0][1])):
            base, cluster = models["baseline"], models["clustering"]
            def pct(new, old):
                old = float(old)
                return "n/a" if old == 0 else "{:.2f}".format((float(new) - old) / old * 100.0)
            pair_reduction = (1.0 - float(cluster["pearson_pairs"]) /
                              max(1.0, float(base["pearson_pairs"]))) * 100.0
            lines.append("| {} | {:.6f} | {:.6f} | {} | {:.6f} | {:.6f} | {} | {:.2f} | {} | {} |".format(
                key[1], float(base["test_mse"]), float(cluster["test_mse"]),
                pct(cluster["test_mse"], base["test_mse"]),
                float(base["test_mae"]), float(cluster["test_mae"]),
                pct(cluster["test_mae"], base["test_mae"]), pair_reduction,
                pct(cluster["training_time"], base["training_time"]),
                pct(cluster["peak_gpu_memory"], base["peak_gpu_memory"])))
        lines.append("")
    failures = [row for row in rows if row.get("status") != "completed"]
    lines += ["## Run status", ""]
    if failures:
        lines += ["| model | pred_len | status | reason | log |", "|---|---:|---|---|---|"]
        for row in failures:
            lines.append("| {} | {} | {} | {} | `{}` |".format(
                row.get("model", ""), row.get("pred_len", ""), row.get("status", "failed"),
                (row.get("failure_reason") or "").replace("|", "\\|"), row.get("log_file", "")))
        lines.append("")
    else:
        lines += ["No failed rows are present in the phase-5 CSV.", ""]
    lines += [
        "## Interpretation and limitations", "",
        "Only horizons with both completed runs are compared. MSE/MAE changes are "
        "computed as `(clustering - baseline) / baseline × 100`; positive values "
        "mean worse forecast error. Runtime and memory changes use the same formula, "
        "so negative values indicate lower cost. No acceptability threshold is "
        "assumed; the conclusion must weigh measured forecast changes against the "
        "measured runtime and memory changes. Results use one seed and one ETTh1 "
        "split, so they do not establish robustness across seeds or datasets.", "",
        "The source CSV contains one row per completed or failed run, exact log path, "
        "and per-run configuration snapshot. No missing metric is imputed.", "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
