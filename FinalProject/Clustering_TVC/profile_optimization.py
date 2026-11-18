"""Matched ETTh1 selector and one-step model smoke profile."""

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
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(ROOT / "official"))
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from FinalProject.Clustering_TVC.clustering_utils import ClusteringGraphSelector
from FinalProject.Clustering_TVC.run_phase5 import attach_selector, seed_all, sync


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index] + (index,)


def timer_cuda(device):
    sync(device)
    return time.perf_counter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT / "configs" / "phase5_etth1.yaml"))
    parser.add_argument("--batch_size", type=int, default=16)
    args_cli = parser.parse_args()
    os.chdir(ROOT)
    smoke_total_start = time.perf_counter()
    with open(args_cli.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["data"] = config.pop("dataset", "ETTh1")
    config.update(pred_len=96, random_seed=2021, batch_size=args_cli.batch_size,
                  use_gpu=torch.cuda.is_available(), gpu=0, use_multi_gpu=False,
                  model="TVCFormer", device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    args = argparse.Namespace(**config)
    seed_all(args.random_seed)
    run_id = "ETTh1_optimization_smoke_{}".format(datetime.now().strftime("%Y%m%dT%H%M%S%f"))
    log_dir = ROOT / "experiments" / "results" / "Clustering_TVC"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (run_id + ".log")
    results_dir = PROJECT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("x", encoding="utf-8") as log:
        original_out, original_err = sys.stdout, sys.stderr
        class Tee:
            def write(self, text):
                original_out.write(text); original_out.flush()
                log.write(text); log.flush()
            def flush(self):
                original_out.flush(); log.flush()
        sys.stdout = sys.stderr = Tee()
        try:
            print("run_id:", run_id)
            print("hardware:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
            print("torch:", torch.__version__, "cuda:", torch.version.cuda)
            print("config:", json.dumps({k: str(v) for k, v in vars(args).items()}, sort_keys=True))
            experiment = Exp_Main(args)
            train_data, train_loader = data_provider(args, "train")
            test_data, test_loader = data_provider(args, "test")
            train_indexed = DataLoader(IndexedDataset(train_data), batch_size=args.batch_size,
                                       shuffle=True, drop_last=True, num_workers=0)
            test_indexed = DataLoader(IndexedDataset(test_data), batch_size=args.batch_size,
                                      shuffle=False, drop_last=False, num_workers=0)
            train_sample = next(iter(train_indexed))
            test_sample = next(iter(test_indexed))
            train_x, train_y, _, _, train_ids = train_sample
            test_x, test_y, _, _, test_ids = test_sample
            device = experiment.device
            test_x = test_x.float().to(device)
            test_y = test_y.float().to(device)

            common = dict(n_clusters=args.n_clusters, keep_rate=args.subgraph_keep_rate,
                          cross_edge_ratio=args.cross_edge_ratio,
                          candidate_multiplier=args.candidate_multiplier,
                          random_state=args.random_seed)
            # Warm CUDA/cuDNN/DGL kernels outside the timed comparison.
            warmup_selector = ClusteringGraphSelector(**common)
            attach_selector(experiment.model, warmup_selector)
            experiment.model.eval()
            with torch.no_grad():
                _ = experiment.model(test_x[:1])
            legacy = ClusteringGraphSelector(**common)
            attach_selector(experiment.model, legacy)
            t0 = timer_cuda(device)
            with torch.no_grad():
                legacy_pred = experiment.model(test_x)
            legacy_cold_forward = time.perf_counter() - t0
            legacy_cold_pairs = legacy.stats.pearson_pairs
            legacy_cold_transfer = legacy.stats.transfer_seconds
            legacy_cold_hash = legacy.stats.cache_seconds
            legacy_cold_feature = legacy.stats.feature_seconds
            legacy_cold_kmeans = legacy.stats.kmeans_seconds
            legacy_cold_pearson = legacy.stats.pearson_seconds
            t0 = timer_cuda(device)
            with torch.no_grad():
                legacy_warm_pred = experiment.model(test_x)
            legacy_warm_forward = time.perf_counter() - t0
            legacy_warm_transfer = legacy.stats.transfer_seconds - legacy_cold_transfer
            legacy_warm_hash = legacy.stats.cache_seconds - legacy_cold_hash
            legacy_warm_pearson = legacy.stats.pearson_seconds - legacy_cold_pearson
            if not torch.allclose(legacy_pred, legacy_warm_pred, atol=1e-6, rtol=1e-6):
                raise AssertionError("legacy cold/warm predictions differ beyond float32 tolerance")
            legacy_mse = float(torch.nn.functional.mse_loss(legacy_pred, test_y[:, -args.pred_len:, :]).item())
            legacy_mae = float(torch.nn.functional.l1_loss(legacy_pred, test_y[:, -args.pred_len:, :]).item())

            optimized = ClusteringGraphSelector(**common)
            precompute_start = time.perf_counter()
            for sample_id, history in zip(test_ids.tolist(), test_x.detach().cpu().numpy()):
                optimized.prepare_indexed_window(("test", int(sample_id)), history)
            sync(device)
            test_precompute = time.perf_counter() - precompute_start
            test_pairs_precomputed = optimized.stats.pearson_pairs
            attach_selector(experiment.model, optimized)
            optimized.set_sample_ids([("test", int(i)) for i in test_ids.tolist()])
            t0 = timer_cuda(device)
            with torch.no_grad():
                optimized_pred = experiment.model(test_x)
            optimized_forward = time.perf_counter() - t0
            optimized_mse = float(torch.nn.functional.mse_loss(optimized_pred, test_y[:, -args.pred_len:, :]).item())
            optimized_mae = float(torch.nn.functional.l1_loss(optimized_pred, test_y[:, -args.pred_len:, :]).item())
            if not torch.allclose(legacy_pred, optimized_pred, atol=1e-6, rtol=1e-6):
                max_delta = float((legacy_pred - optimized_pred).abs().max().item())
                raise AssertionError("legacy/indexed predictions differ beyond tolerance; max abs delta {}".format(max_delta))
            if legacy_cold_pairs != test_pairs_precomputed:
                raise AssertionError("Pearson pair counts differ for identical test windows")

            train_precompute_start = time.perf_counter()
            for sample_id, history in zip(train_ids.tolist(), train_x.numpy()):
                optimized.prepare_indexed_window(("train", int(sample_id)), history)
            sync(device)
            train_precompute = time.perf_counter() - train_precompute_start

            optimizer = torch.optim.Adam(experiment.model.parameters(), lr=args.learning_rate)
            experiment.model.train()
            optimized.set_sample_ids([("train", int(i)) for i in train_ids.tolist()])
            t0 = timer_cuda(device)
            optimizer.zero_grad(set_to_none=True)
            train_x = train_x.float().to(device)
            train_y = train_y.float().to(device)
            prediction = experiment.model(train_x)
            loss = torch.nn.functional.mse_loss(prediction, train_y[:, -args.pred_len:, :])
            loss.backward()
            grads = [p.grad for p in experiment.model.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise AssertionError("training smoke gradients are missing or non-finite")
            optimizer.step()
            sync(device)
            train_step_time = time.perf_counter() - t0

            # One graph-generator timing sample with CUDA synchronization.
            graph_stats = {"seconds": 0.0, "calls": 0, "edges": 0, "nodes": 0}
            root_model = experiment.model.module if isinstance(experiment.model, torch.nn.DataParallel) else experiment.model
            generator = root_model.model.graph_generator
            timer = {"start": None}
            def before(module, inputs):
                sync(device); timer["start"] = time.perf_counter()
            def after(module, inputs, output):
                sync(device); graph_stats["seconds"] += time.perf_counter() - timer["start"]
                graph_stats["calls"] += 1
                graph_stats["edges"] += int(output[0].shape[0])
                graph_stats["nodes"] += int(inputs[0].shape[0] * args.enc_in * args.time_node_num)
            h1 = generator.register_forward_pre_hook(before)
            h2 = generator.register_forward_hook(after)
            experiment.model.eval()
            optimized.set_sample_ids([("test", int(i)) for i in test_ids.tolist()])
            t0 = timer_cuda(device)
            with torch.no_grad():
                final_pred = experiment.model(test_x)
            sync(device)
            final_eval_time = time.perf_counter() - t0
            h1.remove(); h2.remove()
            final_mse = float(torch.nn.functional.mse_loss(final_pred, test_y[:, -args.pred_len:, :]).item())
            final_mae = float(torch.nn.functional.l1_loss(final_pred, test_y[:, -args.pred_len:, :]).item())
            expected_candidates = optimized.stats.pearson_pairs
            unique_windows = optimized.stats.cache_misses
            if optimized.stats.kmeans_calls != unique_windows:
                raise AssertionError("K-Means call count differs from unique prepared windows")

            print("legacy_cold_eval_forward_s={:.6f} legacy_warm_eval_forward_s={:.6f} indexed_eval_forward_s={:.6f}".format(
                legacy_cold_forward, legacy_warm_forward, optimized_forward))
            print("legacy_cold_transfer_s={:.6f} legacy_warm_transfer_s={:.6f} legacy_cold_hash_s={:.6f} legacy_warm_hash_s={:.6f}".format(
                legacy_cold_transfer, legacy_warm_transfer, legacy_cold_hash, legacy_warm_hash))
            print("legacy_cold_feature_s={:.6f} legacy_cold_kmeans_s={:.6f} legacy_cold_pearson_s={:.6f} legacy_warm_pearson_s={:.6f}".format(
                legacy_cold_feature, legacy_cold_kmeans, legacy_cold_pearson, legacy_warm_pearson))
            print("test_precompute_s={:.6f} train_precompute_s={:.6f} train_step_s={:.6f} final_eval_s={:.6f}".format(
                test_precompute, train_precompute, train_step_time, final_eval_time))
            print("feature_s={:.6f} kmeans_s={:.6f} candidate_s={:.6f} pearson_s={:.6f} cache_lookup_s={:.6f} transfer_s={:.6f}".format(
                optimized.stats.feature_seconds, optimized.stats.kmeans_seconds,
                optimized.stats.candidate_seconds, optimized.stats.pearson_seconds,
                optimized.stats.cache_seconds, optimized.stats.transfer_seconds))
            print("cache_hits={} cache_misses={} kmeans_calls={} precomputed_pearson_pairs={}".format(
                optimized.stats.cache_hits, optimized.stats.cache_misses,
                optimized.stats.kmeans_calls, expected_candidates))
            print("legacy_pearson_pairs={} indexed_test_precompute_pairs={}".format(
                legacy_cold_pairs, test_pairs_precomputed))
            print("legacy_mse={:.9f} indexed_mse={:.9f} final_one_step_mse={:.9f}".format(
                legacy_mse, optimized_mse, final_mse))
            print("legacy_indexed_max_abs_prediction_delta={:.9g}".format(
                float((legacy_pred - optimized_pred).abs().max().item())))
            print("legacy_mae={:.9f} indexed_mae={:.9f} final_one_step_mae={:.9f}".format(
                legacy_mae, optimized_mae, final_mae))
            print("graph_generator_s={:.6f} calls={} nodes={} edges={}".format(
                graph_stats["seconds"], graph_stats["calls"], graph_stats["nodes"], graph_stats["edges"]))
            sync(device)
            smoke_total_time = time.perf_counter() - smoke_total_start
            print("smoke_total_wall_s={:.6f}".format(smoke_total_time))
            print("status: completed_smoke_profile")

            row = {
                "run_id": run_id, "dataset": args.data, "pred_len": args.pred_len,
                "seed": args.random_seed, "batch_size": args.batch_size,
                "legacy_cold_eval_forward_s": legacy_cold_forward,
                "legacy_warm_eval_forward_s": legacy_warm_forward,
                "legacy_cold_transfer_s": legacy_cold_transfer,
                "legacy_warm_transfer_s": legacy_warm_transfer,
                "legacy_cold_hash_s": legacy_cold_hash,
                "legacy_warm_hash_s": legacy_warm_hash,
                "legacy_cold_feature_s": legacy_cold_feature,
                "legacy_cold_kmeans_s": legacy_cold_kmeans,
                "legacy_cold_pearson_s": legacy_cold_pearson,
                "legacy_warm_pearson_s": legacy_warm_pearson,
                "indexed_eval_forward_s": optimized_forward,
                "test_precompute_s": test_precompute,
                "train_precompute_s": train_precompute,
                "train_step_s": train_step_time, "final_eval_s": final_eval_time,
                "feature_s": optimized.stats.feature_seconds,
                "kmeans_s": optimized.stats.kmeans_seconds,
                "candidate_s": optimized.stats.candidate_seconds,
                "pearson_s": optimized.stats.pearson_seconds,
                "cache_lookup_s": optimized.stats.cache_seconds,
                "transfer_s": optimized.stats.transfer_seconds,
                "cache_hits": optimized.stats.cache_hits,
                "cache_misses": optimized.stats.cache_misses,
                "kmeans_calls": optimized.stats.kmeans_calls,
                "pearson_pairs": expected_candidates,
                "legacy_test_pearson_pairs": legacy_cold_pairs,
                "legacy_mse": legacy_mse, "indexed_mse": optimized_mse,
                "legacy_mae": legacy_mae, "indexed_mae": optimized_mae,
                "final_one_step_mse": final_mse, "final_one_step_mae": final_mae,
                "graph_generator_s": graph_stats["seconds"],
                "smoke_total_wall_s": smoke_total_time,
                "status": "completed_smoke_profile", "log_file": str(log_path.relative_to(ROOT)),
            }
            csv_path = results_dir / "optimization_smoke_profiles_final.csv"
            exists = csv_path.exists() and csv_path.stat().st_size > 0
            with csv_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception:
            print("status: failed")
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = original_out, original_err


if __name__ == "__main__":
    main()
