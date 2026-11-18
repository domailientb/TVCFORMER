"""One-batch diagnostic for separating Clustering-TVCFormer pipeline costs.

This script is for profiling only; it does not update weights or append to an
existing benchmark CSV. Each invocation writes a new uniquely named directory.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

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
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return self.base[index] + (index,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT / "configs" / "phase5_etth1.yaml"))
    parser.add_argument("--batch_size", type=int, default=128)
    cli = parser.parse_args()
    os.chdir(ROOT)
    with open(cli.config, "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg["data"] = cfg.pop("dataset", "ETTh1")
    cfg.update(pred_len=96, random_seed=2021, batch_size=cli.batch_size,
               use_gpu=torch.cuda.is_available(), gpu=0, use_multi_gpu=False,
               devices="0", model="TVCFormer")
    args = argparse.Namespace(**cfg)
    seed_all(args.random_seed)
    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    run_id = "full_pipeline_profile_{}".format(datetime.now().strftime("%Y%m%dT%H%M%S%f"))
    out_dir = PROJECT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    report_path = out_dir / "profile.txt"
    output = ["run_id=" + run_id,
              "hardware=" + (torch.cuda.get_device_name(0) if args.use_gpu else "CPU"),
              "torch=" + torch.__version__, "config=" + json.dumps(vars(args), default=str, sort_keys=True)]

    experiment = Exp_Main(args)
    train_data, train_loader = data_provider(args, "train")
    indexed_loader = DataLoader(IndexedDataset(train_data), batch_size=args.batch_size,
                                shuffle=True, drop_last=True, num_workers=args.num_workers,
                                pin_memory=bool(args.use_gpu))
    loader_start = time.perf_counter()
    batch = next(iter(indexed_loader))
    sync(device)
    loader_seconds = time.perf_counter() - loader_start
    batch_x_cpu, batch_y_cpu, _, _, ids = batch

    selector = ClusteringGraphSelector(
        n_clusters=args.n_clusters, keep_rate=args.subgraph_keep_rate,
        cross_edge_ratio=args.cross_edge_ratio,
        candidate_multiplier=args.candidate_multiplier,
        random_state=args.random_seed)
    attach_selector(experiment.model, selector)
    precompute_start = time.perf_counter()
    for index, history in zip(ids.tolist(), batch_x_cpu.numpy()):
        selector.prepare_indexed_window(("train", int(index)), history)
    sync(device)
    precompute_seconds = time.perf_counter() - precompute_start
    selector.set_sample_ids([("train", int(index)) for index in ids.tolist()])

    graph_seconds = [0.0]
    handles = []
    root_model = experiment.model.module if isinstance(experiment.model, torch.nn.DataParallel) else experiment.model
    for name in ("model", "model_res", "model_trend"):
        backbone = getattr(root_model, name, None)
        generator = getattr(backbone, "graph_generator", None) if backbone is not None else None
        if generator is None:
            continue
        clock = {"start": 0.0}
        def before(module, inputs, clock=clock):
            sync(device)
            clock["start"] = time.perf_counter()
        def after(module, inputs, result, clock=clock):
            sync(device)
            graph_seconds[0] += time.perf_counter() - clock["start"]
        handles += [generator.register_forward_pre_hook(before),
                    generator.register_forward_hook(after)]

    selector_seconds = [0.0]
    original_select = selector.select
    def timed_select(*call_args, **call_kwargs):
        start = time.perf_counter()
        result = original_select(*call_args, **call_kwargs)
        selector_seconds[0] += time.perf_counter() - start
        return result
    selector.select = timed_select

    # Exclude CUDA context, cuDNN/DGL kernel loading, and allocator warm-up from
    # the measured batch. Warm-up uses the same cached sample IDs and model path.
    warm_x = batch_x_cpu.float().to(device)
    warm_y = batch_y_cpu.float().to(device)
    selector.set_sample_ids([("train", int(index)) for index in ids.tolist()])
    model = experiment.model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=getattr(args, "weight_decay", 0.0))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    warm_prediction = model(warm_x)
    warm_loss = torch.nn.functional.mse_loss(warm_prediction, warm_y[:, -args.pred_len:, :])
    warm_loss.backward()
    optimizer.step()
    sync(device)
    graph_seconds[0] = 0.0
    selector_seconds[0] = 0.0

    transfer_start = time.perf_counter()
    batch_x = batch_x_cpu.float().to(device, non_blocking=bool(args.use_gpu))
    batch_y = batch_y_cpu.float().to(device, non_blocking=bool(args.use_gpu))
    sync(device)
    transfer_seconds = time.perf_counter() - transfer_start

    model.train()
    optimizer.zero_grad(set_to_none=True)
    profiler_activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)
    sync(device)
    forward_start = time.perf_counter()
    with torch.profiler.profile(activities=profiler_activities, record_shapes=True,
                                profile_memory=True) as prof:
        prediction = model(batch_x)
        sync(device)
        forward_seconds = time.perf_counter() - forward_start
        loss = torch.nn.functional.mse_loss(prediction, batch_y[:, -args.pred_len:, :])
        backward_start = time.perf_counter()
        loss.backward()
        sync(device)
        backward_seconds = time.perf_counter() - backward_start
        optimizer_start = time.perf_counter()
        optimizer.step()
        sync(device)
        optimizer_seconds = time.perf_counter() - optimizer_start

    output.extend([
        "data_loader_next_batch_seconds={:.6f}".format(loader_seconds),
        "precompute_batch_seconds={:.6f}".format(precompute_seconds),
        "host_to_device_seconds={:.6f}".format(transfer_seconds),
        "selector_select_seconds={:.6f}".format(selector_seconds[0]),
        "graph_generator_seconds={:.6f}".format(graph_seconds[0]),
        "forward_total_seconds={:.6f}".format(forward_seconds),
        "backward_total_seconds={:.6f}".format(backward_seconds),
        "optimizer_step_seconds={:.6f}".format(optimizer_seconds),
        "loss_mse={:.9f}".format(float(loss.detach().cpu())),
        "cache_hits={} cache_misses={} kmeans_calls={} pearson_pairs={}".format(
            selector.stats.cache_hits, selector.stats.cache_misses,
            selector.stats.kmeans_calls, selector.stats.pearson_pairs),
        "feature_seconds={:.6f} kmeans_seconds={:.6f} candidate_seconds={:.6f} pearson_seconds={:.6f} cache_lookup_seconds={:.6f} edge_pack_seconds={:.6f} edge_transfer_seconds={:.6f}".format(
            selector.stats.feature_seconds, selector.stats.kmeans_seconds,
            selector.stats.candidate_seconds, selector.stats.pearson_seconds,
            selector.stats.cache_seconds, selector.stats.edge_pack_seconds,
            selector.stats.edge_transfer_seconds),
        "--- top CPU operators ---",
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=35),
    ])
    for handle in handles:
        handle.remove()
    report_path.write_text("\n".join(output), encoding="utf-8")
    print(report_path)
    print("\n".join(output[:14]))


if __name__ == "__main__":
    main()
