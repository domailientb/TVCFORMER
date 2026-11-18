"""One-batch inference parity check for retained and vectorized K-Means."""

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "official"))

from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from FinalProject.Clustering_TVC.clustering_utils import ClusteringGraphSelector
from FinalProject.Clustering_TVC.run_phase5 import _load_config, attach_selector


def main():
    config_path = PROJECT / "configs" / "phase5_etth1.yaml"
    config = _load_config(config_path)
    config["root_path"] = str((ROOT / config["root_path"]).resolve())
    config.update(data="ETTh1", pred_len=336, random_seed=2021, batch_size=16,
                  use_gpu=False, is_training=1)
    args = argparse.Namespace(**config)
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)

    test_data, loader = data_provider(args, "test")
    batch_x, batch_y, _, _ = next(iter(DataLoader(
        test_data, batch_size=16, shuffle=False, num_workers=0, drop_last=False)))
    ids = [("test", index) for index in range(batch_x.shape[0])]
    history = batch_x.detach().cpu().numpy()

    selectors = {
        backend: ClusteringGraphSelector(
            n_clusters=args.n_clusters, keep_rate=args.subgraph_keep_rate,
            cross_edge_ratio=args.cross_edge_ratio,
            candidate_multiplier=args.candidate_multiplier,
            random_state=2021, kmeans_backend=backend,
        ) for backend in ("numpy", "numpy_vectorized")
    }
    for index, sample_id in enumerate(ids):
        window = history[index]
        selectors["numpy"].prepare_indexed_window(sample_id, window)
        selectors["numpy_vectorized"].prepare_indexed_window(sample_id, window)

    experiment = Exp_Main(args)
    experiment.model.eval()
    predictions = {}
    with torch.no_grad():
        for backend, selector in selectors.items():
            attach_selector(experiment.model, selector)
            selector.set_sample_ids(ids)
            predictions[backend] = experiment.model(batch_x.float().to(experiment.device))
    target = batch_y[:, -args.pred_len:, :].float().to(experiment.device)
    metrics = {}
    for backend, output in predictions.items():
        metrics[backend] = {
            "mse": float(torch.mean((output - target) ** 2).item()),
            "mae": float(torch.mean(torch.abs(output - target)).item()),
        }
    delta = float(torch.max(torch.abs(predictions["numpy"] - predictions["numpy_vectorized"])).item())
    if delta != 0.0:
        raise RuntimeError(f"one-batch prediction mismatch: max_abs_delta={delta}")
    if metrics["numpy"] != metrics["numpy_vectorized"]:
        raise RuntimeError("one-batch MSE/MAE differ despite prediction parity")

    results = PROJECT / "results"
    output_path = results / "kmeans_vectorized_model_smoke.md"
    if output_path.exists():
        from datetime import datetime
        output_path = results / f"kmeans_vectorized_model_smoke_{datetime.now():%Y%m%dT%H%M%S%f}.md"
    output_path.write_text(
        "# Vectorized K-Means one-batch model parity check\n\n"
        "- Dataset: ETTh1, seq_len=336, pred_len=336, seed=2021.\n"
        "- Same randomly initialized model weights were used sequentially in eval mode.\n"
        "- Selector saw only the 16 `batch_x` history windows; `batch_y` was used only as the metric target.\n"
        f"- Maximum absolute prediction difference: `{delta}`.\n"
        f"- One-batch MSE/MAE diagnostic (not a test-set result): `{metrics}`.\n"
        "- No optimizer step or training was run.\n",
        encoding="utf-8",
    )
    print("smoke report:", output_path)
    print("max_abs_prediction_delta:", delta)
    print("one_batch_metrics:", metrics)


if __name__ == "__main__":
    main()
