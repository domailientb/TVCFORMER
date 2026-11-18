"""Per-input-window K-Means edge selection for Clustering-TVCFormer."""

from dataclasses import dataclass, field
import hashlib
import math
import time
from typing import List, Tuple

import numpy as np


def extract_variable_features(values: np.ndarray) -> np.ndarray:
    """Describe variables using only one observed window [time, channels]."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("window must have shape [time>=2, channels>=2]")
    if not np.isfinite(x).all():
        raise ValueError("window contains non-finite values")
    mean = x.mean(axis=0)
    centered = x - mean
    std = x.std(axis=0)
    safe_std = np.where(std > 1e-12, std, 1.0)
    t = np.linspace(-1.0, 1.0, x.shape[0])
    trend = np.sum(centered * t[:, None], axis=0) / max(np.sum(t * t), 1e-12)
    normalized = centered / safe_std[None, :]
    skew = np.mean(normalized ** 3, axis=0)
    quantiles = np.quantile(x, [0.1, 0.5, 0.9], axis=0).T
    left, right = centered[:-1], centered[1:]
    denominator = np.sqrt(np.sum(left * left, axis=0) * np.sum(right * right, axis=0))
    autocorr = np.divide(np.sum(left * right, axis=0), denominator,
                         out=np.zeros(x.shape[1]), where=denominator > 1e-12)
    features = np.column_stack((mean, std, trend, skew, quantiles, autocorr))
    # Standardize feature dimensions per window. Constant dimensions map to zero.
    fmean = features.mean(axis=0)
    fstd = features.std(axis=0)
    return np.where(fstd > 1e-12, (features - fmean) / np.maximum(fstd, 1e-12), 0.0)


def extract_variable_features_batch(windows: np.ndarray) -> np.ndarray:
    """Batch equivalent of ``extract_variable_features`` for [windows,time,channels]."""
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] < 2:
        raise ValueError("windows must have shape [batch,time>=2,channels>=2]")
    if not np.isfinite(x).all():
        raise ValueError("windows contain non-finite values")
    mean = x.mean(axis=1)
    centered = x - mean[:, None, :]
    std = x.std(axis=1)
    safe_std = np.where(std > 1e-12, std, 1.0)
    t = np.linspace(-1.0, 1.0, x.shape[1])
    trend = np.sum(centered * t[None, :, None], axis=1) / max(np.sum(t * t), 1e-12)
    normalized = centered / safe_std[:, None, :]
    skew = np.mean(normalized ** 3, axis=1)
    quantiles = np.quantile(x, [0.1, 0.5, 0.9], axis=1).transpose(1, 2, 0)
    left, right = centered[:, :-1], centered[:, 1:]
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    autocorr = np.divide(np.sum(left * right, axis=1), denominator,
                         out=np.zeros_like(mean), where=denominator > 1e-12)
    features = np.stack((mean, std, trend, skew,
                         quantiles[:, :, 0], quantiles[:, :, 1], quantiles[:, :, 2],
                         autocorr), axis=-1)
    fmean = features.mean(axis=1, keepdims=True)
    fstd = features.std(axis=1, keepdims=True)
    return np.where(fstd > 1e-12,
                    (features - fmean) / np.maximum(fstd, 1e-12), 0.0)


def _numpy_kmeans_batch_vectorized(features: np.ndarray, n_clusters: int,
                                   random_state: int, n_init: int = 10,
                                   max_iter: int = 50, timing=None) -> np.ndarray:
    """Vectorize existing seeded K-Means across windows and starts.

    Every window uses the same independently reset RandomState as the scalar
    implementation. Standard K-Means++ steps use the same start index and
    random scalar as their per-window counterparts. Degenerate windows are
    delegated to the established implementation to preserve its fallback RNG.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 3:
        raise ValueError("features must have shape [batch,channels,dimensions]")
    batch_size, n_points, _ = features.shape
    k = min(int(n_clusters), n_points)
    if k == 1:
        return np.zeros((batch_size, n_points), dtype=np.int64)
    if batch_size == 0:
        return np.empty((0, n_points), dtype=np.int64)

    # Identify windows with fewer than K distinct descriptors without running
    # a separate np.unique sort for every window.
    equal = np.all(features[:, :, None, :] == features[:, None, :, :], axis=-1)
    prior_equal = np.tril(equal, k=-1).any(axis=2)
    nondegenerate = (~prior_equal).sum(axis=1) >= k
    output = np.empty((batch_size, n_points), dtype=np.int64)
    degenerate_ids = np.flatnonzero(~nondegenerate)
    for index in degenerate_ids:
        output[index] = _numpy_kmeans_vectorized(
            features[index], k, random_state, n_init=n_init, max_iter=max_iter)
    valid_ids = np.flatnonzero(nondegenerate)
    if not len(valid_ids):
        return output
    x = features[valid_ids]
    n_valid = len(valid_ids)

    init_start = time.perf_counter()
    rng = np.random.RandomState(random_state)
    starts = np.empty((n_init, n_valid, k, features.shape[-1]), dtype=np.float64)
    for run in range(n_init):
        first = int(rng.randint(n_points))
        starts[run, :, 0] = x[:, first]
        chosen = [np.full(n_valid, first, dtype=np.int64)]
        for center_id in range(1, k):
            selected = np.stack(chosen, axis=1)
            selected_centers = x[np.arange(n_valid)[:, None], selected]
            distances = np.sum((x[:, :, None, :] - selected_centers[:, None, :, :]) ** 2,
                               axis=-1).min(axis=2)
            totals = distances.sum(axis=1)
            if np.any(totals <= 0) or not np.isfinite(totals).all():
                # Rare near-degenerate geometry: the reference fallback has a
                # different randint draw path, so recompute these windows via
                # the proven single-window implementation.
                bad_local = np.flatnonzero((totals <= 0) | ~np.isfinite(totals))
                for local in bad_local:
                    output[valid_ids[local]] = _numpy_kmeans_vectorized(
                        x[local], k, random_state, n_init=n_init, max_iter=max_iter)
                keep = np.ones(n_valid, dtype=bool)
                keep[bad_local] = False
                if not keep.any():
                    if timing is not None:
                        timing["init"] += time.perf_counter() - init_start
                    return output
                # This rare mixed case is deliberately delegated as a whole
                # rather than risk advancing the seeded draw stream unevenly.
                if not keep.all():
                    for local in np.flatnonzero(keep):
                        output[valid_ids[local]] = _numpy_kmeans_vectorized(
                            x[local], k, random_state, n_init=n_init, max_iter=max_iter)
                    if timing is not None:
                        timing["init"] += time.perf_counter() - init_start
                    return output
            probabilities = distances / totals[:, None]
            cumulative = np.cumsum(probabilities, axis=1)
            cumulative /= cumulative[:, -1, None]
            draw = rng.random_sample()
            next_center = np.sum(cumulative <= draw, axis=1)
            next_center = np.minimum(next_center, n_points - 1)
            chosen.append(next_center.astype(np.int64))
            starts[run, :, center_id] = x[np.arange(n_valid), next_center]
    if timing is not None:
        timing["init"] += time.perf_counter() - init_start

    fit_start = time.perf_counter()
    centers = starts
    labels = np.zeros((n_init, n_valid, n_points), dtype=np.int64)
    run_index = np.arange(n_init)[:, None, None]
    for _iteration in range(max_iter):
        squared = np.sum(
            (x[None, :, :, None, :] - centers[:, :, None, :, :]) ** 2,
            axis=-1,
        )
        next_labels = np.argmin(squared, axis=-1)
        new_centers = centers.copy()
        for cluster_id in range(k):
            members = next_labels == cluster_id
            counts = members.sum(axis=2)
            sums = np.einsum("rbn,bnd->rbd", members.astype(np.float64), x,
                             optimize=False)
            nonempty = counts > 0
            new_centers[:, :, cluster_id, :] = np.where(
                nonempty[:, :, None],
                sums / np.maximum(counts[:, :, None], 1),
                centers[:, :, cluster_id, :],
            )
        stable = np.all(next_labels == labels, axis=2)
        labels, centers = next_labels, new_centers
        if np.all(stable):
            break
    b_index = np.arange(n_valid)[None, :, None]
    assigned = centers[run_index, b_index, labels]
    inertia = np.sum((x[None, :, :, :] - assigned) ** 2, axis=(2, 3))
    best_runs = np.argmin(inertia, axis=0)
    output[valid_ids] = labels[best_runs, np.arange(n_valid)]
    if timing is not None:
        timing["fit"] += time.perf_counter() - fit_start
    return output


def _rank_pairs(features: np.ndarray, pairs: List[Tuple[int, int]], limit: int):
    if not pairs or limit <= 0:
        return []
    ranked = []
    for left, right in pairs:
        delta = features[left] - features[right]
        ranked.append((float(np.dot(delta, delta)), left, right))
    ranked.sort()
    return [(left, right) for _, left, right in ranked[:limit]]


def _candidate_pairs(features, labels, pair_budget, cross_ratio):
    channels = len(labels)
    full_pairs = channels * (channels - 1) // 2
    # Keep at least one pair out of the Pearson pool whenever possible.
    max_pruned_budget = full_pairs if full_pairs <= 1 else full_pairs - 1
    pair_budget = min(max(0, pair_budget), max_pruned_budget)
    all_intra, all_cross = [], []
    for i in range(channels):
        for j in range(i + 1, channels):
            (all_intra if labels[i] == labels[j] else all_cross).append((i, j))
    desired_cross = int(math.ceil(pair_budget * cross_ratio)) if cross_ratio else 0
    cross_count = min(desired_cross, len(all_cross))
    intra_count = min(pair_budget - cross_count, len(all_intra))
    remaining = pair_budget - cross_count - intra_count
    # Fill shortages with the other partition, retaining the requested ratio when feasible.
    if remaining and cross_ratio > 0:
        extra_cross = min(remaining, len(all_cross) - cross_count)
        cross_count += extra_cross
        remaining -= extra_cross
    if remaining:
        intra_count += min(remaining, len(all_intra) - intra_count)
    intra = _rank_pairs(features, all_intra, intra_count)
    cross = _rank_pairs(features, all_cross, cross_count)
    return intra, cross


def _numpy_kmeans(features: np.ndarray, n_clusters: int, random_state: int,
                  n_init: int = 10, max_iter: int = 50, timing=None) -> np.ndarray:
    """Small-data Lloyd K-Means used for a handful of variable descriptors."""
    n_points = features.shape[0]
    k = min(int(n_clusters), n_points)
    if k == 1:
        return np.zeros(n_points, dtype=np.int64)
    rng = np.random.RandomState(random_state)
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        init_start = time.perf_counter()
        centers = [features[rng.randint(n_points)]]
        while len(centers) < k:
            center_array = np.stack(centers)
            distances = np.sum(
                (features[:, None, :] - center_array[None, :, :]) ** 2, axis=-1
            ).min(axis=1)
            total = distances.sum()
            if total <= 0 or not np.isfinite(total):
                remaining = [i for i in range(n_points)
                             if not any(np.array_equal(features[i], c) for c in centers)]
                if not remaining:
                    break
                centers.append(features[remaining[rng.randint(len(remaining))]])
            else:
                centers.append(features[rng.choice(n_points, p=distances / total)])
        centers = np.stack(centers)
        if timing is not None:
            timing["init"] += time.perf_counter() - init_start
        fit_start = time.perf_counter()
        labels = np.zeros(n_points, dtype=np.int64)
        for _iteration in range(max_iter):
            squared = np.sum(
                (features[:, None, :] - centers[None, :, :]) ** 2, axis=-1
            )
            next_labels = np.argmin(squared, axis=1)
            new_centers = centers.copy()
            for cluster_id in range(centers.shape[0]):
                members = features[next_labels == cluster_id]
                if members.size:
                    new_centers[cluster_id] = members.mean(axis=0)
            stable = np.array_equal(next_labels, labels)
            labels, centers = next_labels, new_centers
            if stable:
                break
        inertia = float(np.sum((features - centers[labels]) ** 2))
        if timing is not None:
            timing["fit"] += time.perf_counter() - fit_start
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()
    if best_labels is None:
        raise RuntimeError("NumPy K-Means did not produce cluster labels")
    return best_labels.astype(np.int64)


def _numpy_kmeans_vectorized(features: np.ndarray, n_clusters: int, random_state: int,
                             n_init: int = 10, max_iter: int = 50, timing=None) -> np.ndarray:
    """Vectorize Lloyd assignment/update across the small set of K-Means starts.

    K-Means++ center draws intentionally follow the reference implementation's
    RNG call order. Only the per-start assignment and centroid updates are
    batched, preserving the same initialization and objective.
    """
    n_points, n_features = features.shape
    k = min(int(n_clusters), n_points)
    if k == 1:
        return np.zeros(n_points, dtype=np.int64)
    if np.unique(features, axis=0).shape[0] < k:
        # Match the scalar reference's reduced center count when repeated
        # descriptors leave too few distinct K-Means++ seeds.
        return _numpy_kmeans(features, k, random_state, n_init=n_init,
                             max_iter=max_iter, timing=timing)
    rng = np.random.RandomState(random_state)
    starts = []
    init_elapsed = 0.0
    for _ in range(n_init):
        init_start = time.perf_counter()
        centers = [features[rng.randint(n_points)]]
        while len(centers) < k:
            center_array = np.stack(centers)
            distances = np.sum(
                (features[:, None, :] - center_array[None, :, :]) ** 2, axis=-1
            ).min(axis=1)
            total = distances.sum()
            if total <= 0 or not np.isfinite(total):
                remaining = [i for i in range(n_points)
                             if not any(np.array_equal(features[i], c) for c in centers)]
                if not remaining:
                    break
                centers.append(features[remaining[rng.randint(len(remaining))]])
            else:
                centers.append(features[rng.choice(n_points, p=distances / total)])
        starts.append(np.stack(centers))
        init_elapsed += time.perf_counter() - init_start
    if timing is not None:
        timing["init"] += init_elapsed

    fit_start = time.perf_counter()
    centers = np.stack(starts)  # [n_init, K, feature_dim]
    labels = np.zeros((n_init, n_points), dtype=np.int64)
    run_ids = np.arange(n_init)[:, None]
    for _iteration in range(max_iter):
        squared = np.sum(
            (features[None, :, None, :] - centers[:, None, :, :]) ** 2,
            axis=-1,
        )
        next_labels = np.argmin(squared, axis=-1)
        new_centers = centers.copy()
        for cluster_id in range(k):
            member_mask = next_labels == cluster_id
            counts = member_mask.sum(axis=1)
            sums = np.einsum("rn,nd->rd", member_mask.astype(np.float64), features,
                             optimize=False)
            nonempty = counts > 0
            new_centers[nonempty, cluster_id] = sums[nonempty] / counts[nonempty, None]
        stable = np.all(next_labels == labels, axis=1)
        labels, centers = next_labels, new_centers
        if np.all(stable):
            break
    inertia = np.empty(n_init, dtype=np.float64)
    for run_id in range(n_init):
        assigned_centers = centers[run_id, labels[run_id]]
        inertia[run_id] = np.sum((features - assigned_centers) ** 2)
    best_run = int(np.argmin(inertia))
    if timing is not None:
        timing["fit"] += time.perf_counter() - fit_start
    return labels[best_run].astype(np.int64)


def _sync_if_cuda(tensor):
    import torch
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


@dataclass
class SelectorStats:
    samples: int = 0
    pearson_pairs: int = 0
    full_pairs: int = 0
    cluster_seconds: float = 0.0
    feature_seconds: float = 0.0
    kmeans_seconds: float = 0.0
    kmeans_init_seconds: float = 0.0
    kmeans_fit_seconds: float = 0.0
    cache_seconds: float = 0.0
    candidate_seconds: float = 0.0
    edge_selection_seconds: float = 0.0
    transfer_seconds: float = 0.0
    edge_pack_seconds: float = 0.0
    edge_transfer_seconds: float = 0.0
    pearson_seconds: float = 0.0
    candidate_intra_pairs: int = 0
    candidate_inter_pairs: int = 0
    selected_intra_pairs: int = 0
    selected_inter_pairs: int = 0
    selected_directed_pairs: int = 0
    clusters_per_sample: List[int] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    kmeans_calls: int = 0


class ClusteringGraphSelector:
    """Run K-Means independently on each input sample, then score candidates only."""

    def __init__(self, n_clusters: int, keep_rate: float = 0.5,
                 cross_edge_ratio: float = 0.1, candidate_multiplier: float = 1.25,
                 random_state: int = 2021, pearson_chunk_size: int = 1024,
                 kmeans_backend: str = "numpy_vectorized",
                 batch_edge_assembly: bool = True):
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if not 0.0 < keep_rate <= 1.0:
            raise ValueError("keep_rate must be in (0, 1]")
        if not 0.0 <= cross_edge_ratio <= 1.0:
            raise ValueError("cross_edge_ratio must be in [0, 1]")
        if candidate_multiplier < 1.0:
            raise ValueError("candidate_multiplier must be >= 1")
        if kmeans_backend not in ("numpy", "numpy_vectorized", "sklearn"):
            raise ValueError("kmeans_backend must be 'numpy', 'numpy_vectorized', or 'sklearn'")
        self.n_clusters = int(n_clusters)
        self.keep_rate = float(keep_rate)
        self.cross_edge_ratio = float(cross_edge_ratio)
        self.candidate_multiplier = float(candidate_multiplier)
        self.random_state = int(random_state)
        self.pearson_chunk_size = int(pearson_chunk_size)
        self.kmeans_backend = kmeans_backend
        self.batch_edge_assembly = bool(batch_edge_assembly)
        self.stats = SelectorStats()
        self.last_labels = []
        self._window_cache = {}
        self._indexed_cache = {}
        self.active_sample_ids = None

    def set_sample_ids(self, sample_ids):
        """Attach stable IDs for input history windows in the next forward."""
        self.active_sample_ids = tuple(sample_ids)

    def _prepare_window(self, window):
        features, labels = self._kmeans(window)
        return self._prepare_from_features(window, features, labels)

    def _prepare_from_features(self, window, features, labels):
        """Build candidate and selected-edge cache entries from window stats."""
        channels = window.shape[1]
        full_pairs = channels * (channels - 1) // 2
        target = int(self.keep_rate * full_pairs)
        candidate_start = time.perf_counter()
        max_budget = full_pairs if full_pairs <= 1 else full_pairs - 1
        pool_budget = min(max_budget, int(math.ceil(target * self.candidate_multiplier)))
        intra, inter = _candidate_pairs(features, labels, pool_budget, self.cross_edge_ratio)
        self.stats.candidate_seconds += time.perf_counter() - candidate_start
        if len(set(intra).intersection(inter)) or len(set(intra)) != len(intra) or len(set(inter)) != len(inter):
            raise RuntimeError("candidate pairs contain duplicates or cross-partition overlap")
        self.stats.candidate_intra_pairs += len(intra)
        self.stats.candidate_inter_pairs += len(inter)

        edge_selection_start = time.perf_counter()
        want_inter = int(math.ceil(target * self.cross_edge_ratio)) if self.cross_edge_ratio else 0
        inter_take = min(want_inter, len(inter))
        intra_take = min(target - inter_take, len(intra))
        remaining = target - inter_take - intra_take
        if remaining and self.cross_edge_ratio > 0:
            extra = min(remaining, len(inter) - inter_take)
            inter_take += extra
            remaining -= extra
        if remaining:
            intra_take += min(remaining, len(intra) - intra_take)
        self.stats.edge_selection_seconds += time.perf_counter() - edge_selection_start

        selected = []
        for candidates, count in ((intra, intra_take), (inter, inter_take)):
            if count:
                scores = self._pearson(window.T, candidates)
                selection_start = time.perf_counter()
                positions = np.argsort(-scores, kind="mergesort")[:count].tolist()
                selected.extend(candidates[position] for position in positions)
                self.stats.edge_selection_seconds += time.perf_counter() - selection_start
        self.stats.selected_intra_pairs += intra_take
        self.stats.selected_inter_pairs += inter_take
        return {"labels": labels.copy(), "selected_pairs": tuple(selected),
                "features": features.copy(), "intra": tuple(intra), "inter": tuple(inter)}

    def prepare_indexed_windows_batch(self, sample_ids, history_windows):
        """Precompute a batch of history-only windows with batched stats/K-Means."""
        if self.kmeans_backend != "numpy_vectorized":
            raise ValueError("batched indexed precompute requires numpy_vectorized K-Means")
        ids = tuple(sample_ids)
        windows = np.asarray(history_windows, dtype=np.float32)
        if windows.ndim != 3:
            raise ValueError("history_windows must have shape [batch,time,channels]")
        if len(ids) != windows.shape[0]:
            raise ValueError("sample ID count must match the history batch")
        if len(set(ids)) != len(ids):
            raise ValueError("sample IDs must be unique within a precompute batch")
        if windows.shape[1] < 2 or windows.shape[2] < 2:
            raise ValueError("history windows need at least two time points and channels")

        lookup_start = time.perf_counter()
        missing = [(sample_id, index) for index, sample_id in enumerate(ids)
                   if sample_id not in self._indexed_cache]
        hits = len(ids) - len(missing)
        self.stats.cache_hits += hits
        self.stats.cache_misses += len(missing)
        self.stats.cache_seconds += time.perf_counter() - lookup_start
        if not missing:
            return 0

        missing_windows = np.ascontiguousarray(windows[[index for _, index in missing]])
        prepare_start = time.perf_counter()
        pearson_before = self.stats.pearson_seconds
        feature_start = time.perf_counter()
        batch_features = extract_variable_features_batch(missing_windows)
        self.stats.feature_seconds += time.perf_counter() - feature_start

        k = min(self.n_clusters, batch_features.shape[1])
        kmeans_start = time.perf_counter()
        if k == 1:
            batch_labels = np.zeros((len(missing), batch_features.shape[1]), dtype=np.int64)
        elif self.kmeans_backend == "numpy_vectorized":
            timing = {"init": 0.0, "fit": 0.0}
            batch_labels = _numpy_kmeans_batch_vectorized(
                batch_features, k, self.random_state, timing=timing)
            self.stats.kmeans_init_seconds += timing["init"]
            self.stats.kmeans_fit_seconds += timing["fit"]
            self.stats.kmeans_calls += len(missing)
        else:
            # Preserve alternative reference backends for controlled parity.
            rows = [self._kmeans(window) for window in missing_windows]
            batch_features = np.stack([row[0] for row in rows])
            batch_labels = np.stack([row[1] for row in rows])
        self.stats.kmeans_seconds += time.perf_counter() - kmeans_start

        for local_index, (sample_id, window_index) in enumerate(missing):
            window = missing_windows[local_index]
            prepared = self._prepare_from_features(
                window, batch_features[local_index], batch_labels[local_index])
            self._indexed_cache[sample_id] = prepared
            self.stats.samples += 1
            self.stats.clusters_per_sample.append(int(np.unique(prepared["labels"]).size))
        elapsed = time.perf_counter() - prepare_start
        self.stats.cluster_seconds += max(
            0.0, elapsed - (self.stats.pearson_seconds - pearson_before))
        return len(missing)

    def prepare_indexed_window(self, sample_id, history_window):
        """Precompute graph candidates and Pearson Top-K from history-only input."""
        cache_start = time.perf_counter()
        if sample_id in self._indexed_cache:
            self.stats.cache_hits += 1
            self.stats.cache_seconds += time.perf_counter() - cache_start
            return False
        self.stats.cache_seconds += time.perf_counter() - cache_start
        start = time.perf_counter()
        # The model runner passes encoder input as float32; align offline
        # Pearson ranking with that exact representation before precomputation.
        window = np.asarray(history_window, dtype=np.float32)
        if window.ndim != 2:
            raise ValueError("history_window must have shape [time, channels]")
        pearson_before = self.stats.pearson_seconds
        prepared = self._prepare_window(window)
        self._indexed_cache[sample_id] = prepared
        self.stats.cache_misses += 1
        self.stats.samples += 1
        self.stats.clusters_per_sample.append(int(np.unique(prepared["labels"]).size))
        elapsed = time.perf_counter() - start
        self.stats.cluster_seconds += max(0.0, elapsed - (self.stats.pearson_seconds - pearson_before))
        return True

    def _kmeans(self, window):
        feature_start = time.perf_counter()
        features = extract_variable_features(window)
        self.stats.feature_seconds += time.perf_counter() - feature_start
        k = min(self.n_clusters, features.shape[0])
        if k == 1:
            return features, np.zeros(features.shape[0], dtype=np.int64)
        self.stats.kmeans_calls += 1
        kmeans_start = time.perf_counter()
        if self.kmeans_backend in ("numpy", "numpy_vectorized"):
            timing = {"init": 0.0, "fit": 0.0}
            implementation = (_numpy_kmeans if self.kmeans_backend == "numpy"
                              else _numpy_kmeans_vectorized)
            best_labels = implementation(features, k, self.random_state, timing=timing)
            self.stats.kmeans_init_seconds += timing["init"]
            self.stats.kmeans_fit_seconds += timing["fit"]
        else:
            # Retained as a reference path for controlled implementation
            # comparisons. n_init/max_iter match the NumPy implementation.
            from sklearn.cluster import KMeans
            init_start = time.perf_counter()
            estimator = KMeans(
                n_clusters=k, random_state=self.random_state,
                n_init=10, max_iter=50, algorithm="lloyd",
            )
            self.stats.kmeans_init_seconds += time.perf_counter() - init_start
            fit_start = time.perf_counter()
            best_labels = estimator.fit_predict(features)
            self.stats.kmeans_fit_seconds += time.perf_counter() - fit_start
        self.stats.kmeans_seconds += time.perf_counter() - kmeans_start
        return features, best_labels.astype(np.int64)

    def _pearson(self, values, pairs):
        if not pairs:
            return np.empty((0,), dtype=np.float64)
        start_time = time.perf_counter()
        pair_array = np.asarray(pairs, dtype=np.int64)
        result = np.empty(len(pair_array), dtype=np.float64)
        # values [channels, time], and only candidate channel columns are read.
        for start in range(0, len(pair_array), self.pearson_chunk_size):
            end = min(start + self.pearson_chunk_size, len(pair_array))
            chunk = pair_array[start:end]
            left = values[chunk[:, 0]]
            right = values[chunk[:, 1]]
            left = left - left.mean(axis=-1, keepdims=True)
            right = right - right.mean(axis=-1, keepdims=True)
            numerator = np.sum(left * right, axis=-1)
            denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
            result[start:end] = numerator / np.where(denominator > 0, denominator, 1.0)
        self.stats.pearson_seconds += time.perf_counter() - start_time
        self.stats.pearson_pairs += len(pairs)
        return result

    def select(self, values, keep_rate=None, keep_self_cycles=False):
        """Return [batch, directed_edges, (batch_id, source, destination)]."""
        import torch
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, channels, time]")
        batch_size, channels, time_steps = values.shape
        if channels < 2:
            raise ValueError("at least two variables are required")
        rate = self.keep_rate if keep_rate is None else float(keep_rate)
        target = int(rate * channels * (channels - 1) / 2)
        full_pairs = channels * (channels - 1) // 2
        batch_edges = []
        batch_edge_rows = []
        edge_pack_start = time.perf_counter()
        self.last_labels = []
        sample_ids = self.active_sample_ids
        if sample_ids is not None and len(sample_ids) != batch_size:
            raise ValueError("sample ID count must match batch size")
        if sample_ids is None:
            transfer_start = time.perf_counter()
            _sync_if_cuda(values)
            batch_windows = values.detach().transpose(1, 2).cpu().numpy()
            _sync_if_cuda(values)
            self.stats.transfer_seconds += time.perf_counter() - transfer_start
        else:
            batch_windows = None

        for b in range(batch_size):
            if sample_ids is not None:
                cache_start = time.perf_counter()
                prepared = self._indexed_cache.get(sample_ids[b])
                self.stats.cache_seconds += time.perf_counter() - cache_start
                if prepared is None:
                    raise KeyError("unprepared history window ID: {}".format(sample_ids[b]))
                self.stats.cache_hits += 1
                labels = prepared["labels"]
                selected = prepared["selected_pairs"]
                self.stats.full_pairs += full_pairs
                self.stats.clusters_per_sample.append(int(np.unique(labels).size))
            else:
                cluster_start = time.perf_counter()
                window = batch_windows[b]
                cache_start = time.perf_counter()
                cache_key = hashlib.blake2b(np.ascontiguousarray(window).tobytes(), digest_size=16).digest()
                prepared = self._window_cache.get(cache_key)
                self.stats.cache_seconds += time.perf_counter() - cache_start
                if prepared is None:
                    features, labels = self._kmeans(window)
                    candidate_start = time.perf_counter()
                    max_budget = full_pairs if full_pairs <= 1 else full_pairs - 1
                    pool_budget = min(max_budget, int(math.ceil(target * self.candidate_multiplier)))
                    intra, inter = _candidate_pairs(features, labels, pool_budget, self.cross_edge_ratio)
                    self.stats.candidate_seconds += time.perf_counter() - candidate_start
                    self.stats.candidate_intra_pairs += len(intra)
                    self.stats.candidate_inter_pairs += len(inter)
                    prepared = {"labels": labels.copy(), "features": features.copy(),
                                "intra": tuple(intra), "inter": tuple(inter)}
                    self._window_cache[cache_key] = prepared
                    self.stats.cache_misses += 1
                else:
                    self.stats.cache_hits += 1
                labels = prepared["labels"]
                intra, inter = prepared["intra"], prepared["inter"]
                pearson_before = self.stats.pearson_seconds
                desired_inter = int(math.ceil(target * self.cross_edge_ratio)) if self.cross_edge_ratio else 0
                inter_take = min(desired_inter, len(inter))
                intra_take = min(target - inter_take, len(intra))
                remaining = target - inter_take - intra_take
                if remaining and self.cross_edge_ratio > 0:
                    extra = min(remaining, len(inter) - inter_take)
                    inter_take += extra
                    remaining -= extra
                if remaining:
                    intra_take += min(remaining, len(intra) - intra_take)
                selected = []
                for candidates, count in ((intra, intra_take), (inter, inter_take)):
                    if count:
                        scores = self._pearson(window.T, candidates)
                        positions = np.argsort(-scores, kind="mergesort")[:count].tolist()
                        selected.extend(candidates[position] for position in positions)
                self.stats.selected_intra_pairs += intra_take
                self.stats.selected_inter_pairs += inter_take
                elapsed = time.perf_counter() - cluster_start
                pearson_elapsed = self.stats.pearson_seconds - pearson_before
                self.stats.cluster_seconds += max(0.0, elapsed - pearson_elapsed)
                self.stats.samples += 1
                self.stats.full_pairs += full_pairs
                self.stats.clusters_per_sample.append(int(np.unique(labels).size))

            self.last_labels.append(labels.copy())
            self.stats.selected_intra_pairs += sum(1 for i, j in selected if labels[i] == labels[j])
            self.stats.selected_inter_pairs += sum(1 for i, j in selected if labels[i] != labels[j])
            if self.batch_edge_assembly:
                # Preserve the original order (all forward directions, then
                # their reverses) while assembling the whole batch on CPU.
                if selected:
                    src = [pair[0] for pair in selected] + [pair[1] for pair in selected]
                    dst = [pair[1] for pair in selected] + [pair[0] for pair in selected]
                else:
                    src, dst = [], []
                if keep_self_cycles:
                    src.extend(range(channels))
                    dst.extend(range(channels))
                self.stats.selected_directed_pairs += len(src)
                rows = np.empty((len(src), 3), dtype=np.int64)
                if src:
                    rows[:, 0] = b
                    rows[:, 1] = src
                    rows[:, 2] = dst
                batch_edge_rows.append(rows)
            else:
                # Retained sample-wise reference path for parity and benchmark.
                if selected:
                    pair_tensor = torch.as_tensor(selected, device=values.device, dtype=torch.long)
                    src = torch.cat((pair_tensor[:, 0], pair_tensor[:, 1]))
                    dst = torch.cat((pair_tensor[:, 1], pair_tensor[:, 0]))
                else:
                    src = torch.empty((0,), device=values.device, dtype=torch.long)
                    dst = torch.empty_like(src)
                if keep_self_cycles:
                    loops = torch.arange(channels, device=values.device)
                    src, dst = torch.cat((src, loops)), torch.cat((dst, loops))
                batch_ids = torch.full_like(src, b)
                self.stats.selected_directed_pairs += int(src.numel())
                batch_edges.append(torch.stack((batch_ids, src, dst), dim=-1))

        if not self.batch_edge_assembly:
            return torch.stack(batch_edges, dim=0)
        edge_array = np.stack(batch_edge_rows, axis=0)
        self.stats.edge_pack_seconds += time.perf_counter() - edge_pack_start
        edge_transfer_start = time.perf_counter()
        edge_tensor = torch.as_tensor(edge_array, device=values.device, dtype=torch.long)
        _sync_if_cuda(values)
        self.stats.edge_transfer_seconds += time.perf_counter() - edge_transfer_start
        return edge_tensor
