"""Compare the NumPy and sklearn K-Means paths on ETTh1 input windows."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
OFFICIAL = ROOT / "official"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from data_provider.data_factory import data_provider
from FinalProject.Clustering_TVC.clustering_utils import ClusteringGraphSelector
from FinalProject.Clustering_TVC.run_phase5 import _load_config


def same_partition(left, right):
    return all((left[i] == left[j]) == (right[i] == right[j])
               for i in range(len(left)) for j in range(i + 1, len(left)))


def check_symmetric(pairs):
    return all(0 <= i < j for i, j in pairs) and len(set(pairs)) == len(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT / "configs" / "phase5_etth1.yaml"))
    parser.add_argument("--limit", type=int, default=0,
                        help="optional history-window limit for a quick diagnostic; 0 uses all splits")
    parser.add_argument("--thread_limit", type=int, default=1,
                        help="BLAS/OpenMP thread cap; tiny 7-variable K-Means is benchmarked at 1 by default")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    config = _load_config(config_path)
    config["root_path"] = str((ROOT / config["root_path"]).resolve())
    config["pred_len"] = 336
    config["random_seed"] = 2021
    config["use_gpu"] = False
    config["is_training"] = 1
    data_args = argparse.Namespace(**config)

    windows = []
    split_sizes = {}
    for split in ("train", "val", "test"):
        dataset, _ = data_provider(data_args, split)
        split_sizes[split] = len(dataset)
        for index in range(len(dataset)):
            history = np.asarray(dataset.data_x[index:index + dataset.seq_len], dtype=np.float32)
            if history.shape != (dataset.seq_len, config["enc_in"]):
                raise RuntimeError(f"bad history window {split}:{index}: {history.shape}")
            windows.append(((split, index), history))
    if args.limit > 0:
        windows = windows[:args.limit]
    if not windows:
        raise RuntimeError("no ETTh1 history windows loaded")

    common = dict(
        n_clusters=config["n_clusters"], keep_rate=config["subgraph_keep_rate"],
        cross_edge_ratio=config["cross_edge_ratio"],
        candidate_multiplier=config["candidate_multiplier"], random_state=2021,
    )
    backend_names = ("numpy", "numpy_vectorized", "sklearn")
    selectors = {name: ClusteringGraphSelector(**common, kmeans_backend=name)
                 for name in backend_names}
    comparisons = {name: {"pair": 0, "selected": 0, "partition": 0, "raw": 0}
                   for name in backend_names if name != "numpy"}
    examples = {name: [] for name in comparisons}
    wall = {name: 0.0 for name in selectors}
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    thread_context = threadpool_limits(limits=args.thread_limit)
    thread_context.__enter__()
    pool_info = threadpool_info()

    # Rotate execution order to reduce systematic thermal/thread-pool bias.
    for position, (sample_id, history) in enumerate(windows):
        shift = position % len(backend_names)
        order = backend_names[shift:] + backend_names[:shift]
        prepared = {}
        for name in order:
            call_start = time.perf_counter()
            selectors[name].prepare_indexed_window(sample_id, history)
            wall[name] += time.perf_counter() - call_start
            prepared[name] = selectors[name]._indexed_cache[sample_id]
        np_candidates = set(prepared["numpy"]["intra"]) | set(prepared["numpy"]["inter"])
        np_selected = set(prepared["numpy"]["selected_pairs"])
        for name in backend_names:
            pairs = set(prepared[name]["selected_pairs"])
            if not check_symmetric(pairs):
                raise RuntimeError(f"asymmetric selected pair set for {name} at {sample_id}")
        for name in comparisons:
            candidate_set = set(prepared[name]["intra"]) | set(prepared[name]["inter"])
            selected_set = set(prepared[name]["selected_pairs"])
            labels_equal = same_partition(prepared["numpy"]["labels"], prepared[name]["labels"])
            raw_equal = np.array_equal(prepared["numpy"]["labels"], prepared[name]["labels"])
            comparisons[name]["partition"] += int(not labels_equal)
            comparisons[name]["raw"] += int(not raw_equal)
            comparisons[name]["pair"] += int(np_candidates != candidate_set)
            comparisons[name]["selected"] += int(np_selected != selected_set)
            if (not labels_equal or np_candidates != candidate_set or np_selected != selected_set) and len(examples[name]) < 3:
                examples[name].append({
                    "sample_id": sample_id,
                    "partition_equal": labels_equal,
                    "raw_labels_equal": raw_equal,
                    "numpy_labels": prepared["numpy"]["labels"].tolist(),
                    "other_labels": prepared[name]["labels"].tolist(),
                    "numpy_candidate_count": len(np_candidates),
                    "other_candidate_count": len(candidate_set),
                    "candidate_symmetric_difference": sorted(np_candidates ^ candidate_set)[:8],
                    "selected_symmetric_difference": sorted(np_selected ^ selected_set)[:8],
                })

    # Validate the Graph Generator's expected selector COO batch shape and
    # reciprocal edges on an identical small slice of prepared histories.
    graph_check = {}
    graph_batch = min(32, len(windows))
    values = torch.from_numpy(np.stack([history.T for _, history in windows[:graph_batch]]))
    graph_edge_sets = {}
    for name, selector in selectors.items():
        ids = [sample_id for sample_id, _ in windows[:graph_batch]]
        selector.set_sample_ids(ids)
        edge_index = selector.select(values)
        if edge_index.ndim != 3 or edge_index.shape[0] != graph_batch or edge_index.shape[2] != 3:
            raise RuntimeError(f"invalid Graph Generator selector COO format for {name}: {tuple(edge_index.shape)}")
        for row, sample_edges in enumerate(edge_index.cpu().numpy()):
            observed = {(int(src), int(dst)) for batch_id, src, dst in sample_edges
                        if int(batch_id) == row}
            if any((dst, src) not in observed for src, dst in observed):
                raise RuntimeError(f"asymmetric selector COO output for {name}, batch={row}")
        graph_edge_sets[name] = [
            {(int(src), int(dst)) for batch_id, src, dst in sample_edges if int(batch_id) == row}
            for row, sample_edges in enumerate(edge_index.cpu().numpy())
        ]
        graph_check[name] = {
            "batch": graph_batch,
            "shape": list(edge_index.shape),
            "directed_edges": int(edge_index.shape[1]),
            "symmetric": True,
            "edge_sets_match_numpy": name == "numpy" or graph_edge_sets[name] == graph_edge_sets["numpy"],
        }
        # Restore each selector's sample IDs after the cross-check selector call.
        selector.set_sample_ids(ids)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    results_dir = PROJECT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"kmeans_backend_benchmark_{timestamp}.csv"
    report_path = results_dir / f"kmeans_backend_benchmark_{timestamp}.md"
    fields = ["backend", "windows", "wall_seconds", "feature_seconds", "kmeans_seconds",
              "kmeans_init_seconds", "kmeans_fit_seconds", "candidate_seconds",
              "pearson_seconds", "cache_lookup_seconds", "transfer_seconds",
              "kmeans_calls", "cache_hits", "cache_misses", "pearson_pairs",
              "candidate_intra", "candidate_inter", "selected_intra", "selected_inter"]
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, selector in selectors.items():
            stats = selector.stats
            writer.writerow({
                "backend": name, "windows": len(windows), "wall_seconds": wall[name],
                "feature_seconds": stats.feature_seconds, "kmeans_seconds": stats.kmeans_seconds,
                "kmeans_init_seconds": stats.kmeans_init_seconds,
                "kmeans_fit_seconds": stats.kmeans_fit_seconds,
                "candidate_seconds": stats.candidate_seconds,
                "pearson_seconds": stats.pearson_seconds,
                "cache_lookup_seconds": stats.cache_seconds,
                "transfer_seconds": stats.transfer_seconds,
                "kmeans_calls": stats.kmeans_calls, "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses, "pearson_pairs": stats.pearson_pairs,
                "candidate_intra": stats.candidate_intra_pairs,
                "candidate_inter": stats.candidate_inter_pairs,
                "selected_intra": stats.selected_intra_pairs,
                "selected_inter": stats.selected_inter_pairs,
            })

    numpy_stats = selectors["numpy"].stats
    report = [
        "# ETTh1 K-Means backend benchmark",
        "",
        f"- Timestamp: `{timestamp}`",
        f"- Dataset histories loaded: `{len(windows)}` (configured split lengths: `{split_sizes}`); each sample uses only `data_x[index:index+seq_len]`.",
        f"- Configuration: ETTh1, seq_len={config['seq_len']}, pred_len=336, seed=2021, K={config['n_clusters']}, NumPy float64 statistic features after float32 model-input conversion.",
        "- All backends use K-Means++ initialization, 10 starts, Lloyd assignment/update, and the same selected-candidate and Pearson ranking code. sklearn reference is explicit `n_init=10`, `max_iter=50`, `algorithm='lloyd'`; `numpy` is the retained scalar reference and `numpy_vectorized` is the new production default.",
        "- Timings were collected sequentially with backend order alternated per window. They are single-process measurements on the current machine; no GPU model forward or MSE/MAE training evaluation is included.",
        f"- BLAS/OpenMP thread cap: `{args.thread_limit}`; detected pools: `{pool_info}`. The small 7-variable problem is particularly sensitive to OpenMP launch/synchronization overhead.",
        "",
        "## Measured timing and work",
        "",
        "| Backend | Total loop (s) | Feature (s) | K-Means total (s) | Init (s) | Lloyd fit (s) | Candidate (s) | Pearson (s) | Cache lookup (s) | Transfer (s) | Pearson pairs | K-Means calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, selector in selectors.items():
        st = selector.stats
        report.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(
            name, wall[name], st.feature_seconds, st.kmeans_seconds,
            st.kmeans_init_seconds, st.kmeans_fit_seconds, st.candidate_seconds,
            st.pearson_seconds, st.cache_seconds, st.transfer_seconds,
            st.pearson_pairs, st.kmeans_calls))
    report.extend([
        "",
        "## Functional parity",
        "",
        "- Partition/candidate/selected-set mismatches are measured against the retained NumPy implementation; cluster IDs may differ only by permutation without constituting a partition mismatch.",
        "| Comparison | Partition mismatches | Label-ID-only diffs | Candidate-set mismatches | Selected-set mismatches | Pearson evaluations |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, counts in comparisons.items():
        report.append("| NumPy vs {} | {}/{} | {}/{} | {}/{} | {}/{} | {} vs {} |".format(
            name, counts["partition"], len(windows), counts["raw"], len(windows),
            counts["pair"], len(windows), counts["selected"], len(windows),
            numpy_stats.pearson_pairs, selectors[name].stats.pearson_pairs))
    report.extend([
        "",
        f"- Directed selector COO/Graph Generator input check: `{graph_check}`; selected pairs were symmetric in both backends.",
        f"- Runtime ratios (scalar NumPy time / backend time): `{ {name: round(wall['numpy'] / wall[name], 3) for name in selectors if name != 'numpy'} }x`; values above 1 indicate the compared backend is faster.",
        "- MSE/MAE were not measured in this algorithm-level benchmark. A separate 16-sample eval-mode parity smoke is recorded in `results/kmeans_vectorized_model_smoke.md`; it is not a full test-set evaluation.",
        "",
        "## Mismatch examples",
        "",
    ])
    if any(examples.values()):
        report.append("```json")
        import json
        report.append(json.dumps(examples, indent=2))
        report.append("```")
    else:
        report.append("No label partition, candidate-pair, or selected-pair mismatches were observed for any compared backend.")
    report.extend([
        "",
        "## Interpretation limits",
        "",
        "- sklearn is retained as a benchmark-only selector option. The vectorized NumPy backend became the production default after matching the scalar reference over all loaded ETTh1 histories; no encoder, decoder, GraphSAGE, loss, or graph edge policy is changed.",
        "- sklearn K-Means++ initialization and NumPy's one-trial K-Means++ may converge to different local optima. If any partition/candidate/selected pair mismatch is reported, the implementations are not exactly functionally interchangeable for that window, even though both use the same feature matrix and K-Means assignment/update rule.",
        "- No full experiment was run. Compare component times and pair/edge parity before considering an end-to-end training benchmark.",
        "",
        f"Detailed CSV: `{csv_path.relative_to(ROOT).as_posix()}`",
    ])
    report_path.write_text("\n".join(report), encoding="utf-8")
    thread_context.__exit__(None, None, None)
    print("benchmark csv:", csv_path)
    print("benchmark report:", report_path)
    print("windows:", len(windows), "backend wall seconds:", wall)
    print("mismatch counts vs numpy:", comparisons)


if __name__ == "__main__":
    main()
