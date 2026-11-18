"""Whole-ETTh1 artifact parity check for sequential vs batched precompute."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(ROOT / "official"))
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from FinalProject.Clustering_TVC.clustering_utils import ClusteringGraphSelector
from FinalProject.Clustering_TVC.run_phase5 import _load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT / "configs" / "phase5_etth1.yaml"))
    parser.add_argument("--batch_size", type=int, default=128)
    args_cli = parser.parse_args()
    config = _load_config(Path(args_cli.config))
    config["root_path"] = str((ROOT / config["root_path"]).resolve())
    config.update(pred_len=96, random_seed=2021, use_gpu=False, use_multi_gpu=False,
                  device="cpu", model="TVCFormer")
    args = argparse.Namespace(**config)
    datasets = [(split, data_provider(args, split)[0]) for split in ("train", "val", "test")]
    count = sum(len(dataset) for _, dataset in datasets)
    if count != 13779:
        raise RuntimeError(f"expected 13779 windows, got {count}")

    settings = dict(n_clusters=config["n_clusters"], keep_rate=config["subgraph_keep_rate"],
                    cross_edge_ratio=config["cross_edge_ratio"],
                    candidate_multiplier=config["candidate_multiplier"], random_state=2021)
    reference = ClusteringGraphSelector(**settings, batch_edge_assembly=True)
    batched = ClusteringGraphSelector(**settings, batch_edge_assembly=True)
    results = {"dataset": "ETTh1", "seq_len": config["seq_len"], "pred_len": 96,
               "seed": 2021, "windows": count, "batch_size": args_cli.batch_size,
               "status": "running", "features_close": 0, "max_feature_abs_diff": 0.0,
               "labels_equal": 0,
               "candidate_pairs_equal": 0, "selected_pairs_equal": 0,
               "pearson_score_arrays_equal": 0, "pearson_values_compared": 0,
               "coo_batches_equal": 0, "coo_edges_compared": 0,
               "reference_precompute_pearson_pairs": 0,
               "batched_precompute_pearson_pairs": 0}
    compared = 0
    for split, dataset in datasets:
        for start in range(0, len(dataset), args_cli.batch_size):
            end = min(start + args_cli.batch_size, len(dataset))
            ids = [(split, index) for index in range(start, end)]
            histories = np.stack([
                dataset.data_x[index:index + dataset.seq_len]
                for index in range(start, end)
            ], axis=0)
            if histories.shape[1] != config["seq_len"]:
                raise AssertionError("only history windows may enter precompute")
            for sample_id, window in zip(ids, histories):
                reference.prepare_indexed_window(sample_id, window)
            batched.prepare_indexed_windows_batch(ids, histories)

            for sample_id, window in zip(ids, histories):
                old = reference._indexed_cache[sample_id]
                new = batched._indexed_cache[sample_id]
                results["reference_precompute_pearson_pairs"] += len(old["intra"]) + len(old["inter"])
                results["batched_precompute_pearson_pairs"] += len(new["intra"]) + len(new["inter"])
                feature_diff = np.abs(new["features"] - old["features"])
                np.testing.assert_allclose(new["features"], old["features"], rtol=1e-12, atol=1e-12)
                results["features_close"] += 1
                results["max_feature_abs_diff"] = max(
                    results["max_feature_abs_diff"], float(feature_diff.max(initial=0.0)))
                np.testing.assert_array_equal(new["labels"], old["labels"])
                results["labels_equal"] += 1
                assert new["intra"] == old["intra"] and new["inter"] == old["inter"]
                results["candidate_pairs_equal"] += 1
                assert new["selected_pairs"] == old["selected_pairs"]
                results["selected_pairs_equal"] += 1
                for pairs in (old["intra"], old["inter"]):
                    if pairs:
                        old_scores = reference._pearson(window.T, pairs)
                        new_scores = batched._pearson(window.T, pairs)
                        np.testing.assert_array_equal(new_scores, old_scores)
                        results["pearson_score_arrays_equal"] += 1
                        results["pearson_values_compared"] += len(pairs)
                compared += 1

            values = torch.as_tensor(histories.transpose(0, 2, 1), dtype=torch.float32)
            reference.set_sample_ids(ids)
            batched.set_sample_ids(ids)
            old_coo = reference.select(values).cpu().numpy()
            new_coo = batched.select(values).cpu().numpy()
            np.testing.assert_array_equal(new_coo, old_coo)
            results["coo_batches_equal"] += 1
            results["coo_edges_compared"] += int(old_coo.shape[0] * old_coo.shape[1])

    if compared != count:
        raise AssertionError(f"compared {compared} windows, expected {count}")
    if reference.stats.kmeans_calls != batched.stats.kmeans_calls:
        raise AssertionError("K-Means call counters differ")
    if reference.stats.pearson_pairs != batched.stats.pearson_pairs:
        raise AssertionError("candidate Pearson counters differ")
    results.update(status="passed", windows_compared=compared,
                    reference_kmeans_calls=reference.stats.kmeans_calls,
                    batched_kmeans_calls=batched.stats.kmeans_calls,
                    reference_pearson_pairs_including_verifier_rechecks=reference.stats.pearson_pairs,
                    batched_pearson_pairs_including_verifier_rechecks=batched.stats.pearson_pairs,
                    cache_misses=batched.stats.cache_misses,
                    cache_hits_from_model_select=batched.stats.cache_hits)
    out_dir = PROJECT / "results" / f"precompute_parity_{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    output = out_dir / "whole_etth1_parity.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
