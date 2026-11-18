import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "official"))
sys.path.insert(0, str(ROOT))

from FinalProject.Clustering_TVC.clustering_utils import (
    ClusteringGraphSelector,
    _candidate_pairs,
    _numpy_kmeans_batch_vectorized,
    _numpy_kmeans_vectorized,
    extract_variable_features,
    extract_variable_features_batch,
)


def sample_window(seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 48)
    return np.stack((t, t + 0.01 * rng.normal(size=t.size),
                     np.sin(6 * t), np.cos(5 * t)), axis=1)


def test_features_use_only_supplied_window_and_are_finite():
    features = extract_variable_features(sample_window(1))
    assert features.shape[0] == 4
    assert np.isfinite(features).all()


def test_batch_feature_extraction_matches_window_reference():
    windows = np.stack([sample_window(seed) for seed in range(60, 68)])
    batched = extract_variable_features_batch(windows)
    expected = np.stack([extract_variable_features(window) for window in windows])
    np.testing.assert_array_equal(batched, expected)


def test_batch_kmeans_matches_per_window_seeded_reference_including_degenerate_features():
    rng = np.random.RandomState(73)
    features = rng.normal(size=(24, 7, 8))
    # Include windows that must take the established duplicate-descriptor path.
    features[2] = np.tile(features[2, :1], (7, 1))
    features[9] = np.round(features[9], decimals=2)
    expected = np.stack([_numpy_kmeans_vectorized(row, 3, 2021) for row in features])
    actual = _numpy_kmeans_batch_vectorized(features, 3, 2021)
    np.testing.assert_array_equal(actual, expected)


def test_indexed_batched_precompute_matches_reference_cache_artifacts():
    windows = np.stack([sample_window(seed) for seed in range(80, 100)])
    reference = ClusteringGraphSelector(n_clusters=3, keep_rate=0.7,
                                        cross_edge_ratio=0.1, random_state=2021)
    batched = ClusteringGraphSelector(n_clusters=3, keep_rate=0.7,
                                      cross_edge_ratio=0.1, random_state=2021)
    ids = [("train", i) for i in range(len(windows))]
    for sample_id, window in zip(ids, windows):
        reference.prepare_indexed_window(sample_id, window)
    batched.prepare_indexed_windows_batch(ids, windows)
    for sample_id in ids:
        expected = reference._indexed_cache[sample_id]
        actual = batched._indexed_cache[sample_id]
        np.testing.assert_array_equal(actual["features"], expected["features"])
        np.testing.assert_array_equal(actual["labels"], expected["labels"])
        assert actual["intra"] == expected["intra"]
        assert actual["inter"] == expected["inter"]
        assert actual["selected_pairs"] == expected["selected_pairs"]
    assert batched.stats.cache_misses == reference.stats.cache_misses == len(ids)
    assert batched.stats.pearson_pairs == reference.stats.pearson_pairs


def test_constant_variables_and_singleton_clusters_are_handled_without_duplicate_candidates():
    window = np.column_stack((np.ones(32), np.arange(32), np.ones(32) * 3, np.sin(np.arange(32))))
    features = extract_variable_features(window)
    assert np.isfinite(features).all()
    intra, inter = _candidate_pairs(features, np.arange(4), 3, 0.0)
    assert intra == [] and inter == []  # zero cross budget is a strict no-cross policy


def test_n_clusters_above_channel_count_is_clamped_and_pair_shortfall_is_explicit():
    values = torch.tensor(np.stack((sample_window(3)[:, :4].T,)), dtype=torch.float32)
    selector = ClusteringGraphSelector(n_clusters=9, keep_rate=1.0,
                                       cross_edge_ratio=0.0, random_state=11)
    # Force singleton labels to make the no-intra-edge case deterministic.
    selector._kmeans = lambda window: (np.eye(4), np.arange(4))
    edges = selector.select(values)
    assert len(selector.last_labels[0]) == 4
    assert edges.shape == (1, 0, 3)
    assert selector.stats.pearson_pairs == 0
    assert selector.stats.selected_intra_pairs == 0
    assert selector.stats.selected_inter_pairs == 0


def test_two_variables_can_use_the_only_available_pair():
    values = torch.tensor(np.stack((sample_window(4)[:, :2].T,)), dtype=torch.float32)
    selector = ClusteringGraphSelector(n_clusters=5, keep_rate=1.0,
                                       cross_edge_ratio=1.0, random_state=2)
    edges = selector.select(values)
    assert selector.stats.full_pairs == selector.stats.pearson_pairs == 1
    assert edges.shape == (1, 2, 3)


def test_zero_variance_channel_produces_finite_pearson_scores():
    t = np.linspace(-1.0, 1.0, 32)
    values = torch.tensor(
        np.stack((np.ones_like(t), t, np.sin(5 * t)), axis=0)[None],
        dtype=torch.float32,
    )
    selector = ClusteringGraphSelector(n_clusters=2, keep_rate=1.0,
                                       cross_edge_ratio=1.0, random_state=5)
    original_pearson = selector._pearson
    evaluated_scores = []
    def capture_scores(window, pairs):
        scores = original_pearson(window, pairs)
        evaluated_scores.append(scores)
        return scores
    selector._pearson = capture_scores
    edges = selector.select(values)
    assert edges.shape[0] == 1
    assert selector.stats.pearson_pairs > 0
    assert all(np.isfinite(scores).all() for scores in evaluated_scores)


def test_kmeans_runs_per_sample_and_only_scores_bounded_candidates():
    selector = ClusteringGraphSelector(
        n_clusters=2, keep_rate=0.5, cross_edge_ratio=0.25,
        candidate_multiplier=1.5, random_state=19,
    )
    x = torch.tensor(np.stack((sample_window(1).T, sample_window(8).T)), dtype=torch.float32)
    seen_windows = []
    original_kmeans = selector._kmeans
    def capture_window(window):
        seen_windows.append(window.copy())
        return original_kmeans(window)
    selector._kmeans = capture_window
    edges = selector.select(x)
    assert len(selector.last_labels) == 2
    np.testing.assert_allclose(seen_windows[0], x[0].numpy().T)
    np.testing.assert_allclose(seen_windows[1], x[1].numpy().T)
    assert selector.stats.samples == 2
    assert 0 < selector.stats.pearson_pairs < selector.stats.full_pairs
    assert selector.stats.pearson_pairs == (
        selector.stats.candidate_intra_pairs + selector.stats.candidate_inter_pairs
    )
    assert edges.shape[0] == 2 and edges.shape[2] == 3
    first_pearson_count = selector.stats.pearson_pairs
    selector.select(x)
    assert selector.stats.cache_misses == 2
    assert selector.stats.cache_hits == 2
    assert len(seen_windows) == 2  # cached repeats skip K-Means
    assert selector.stats.pearson_pairs == 2 * first_pearson_count  # legacy path rescored Pearson on each forward


def test_indexed_precomputation_matches_content_cache_without_runtime_transfer():
    values = torch.tensor(np.stack((sample_window(13).T, sample_window(21).T)), dtype=torch.float32)
    legacy = ClusteringGraphSelector(n_clusters=2, keep_rate=0.5,
                                     cross_edge_ratio=0.25, random_state=31)
    expected_edges = legacy.select(values)
    expected_pairs = legacy.stats.pearson_pairs

    indexed = ClusteringGraphSelector(n_clusters=2, keep_rate=0.5,
                                      cross_edge_ratio=0.25, random_state=31)
    ids = [("train", 2), ("train", 8)]
    for sample_id, history in zip(ids, values.numpy().transpose(0, 2, 1)):
        assert indexed.prepare_indexed_window(sample_id, history)
    indexed.set_sample_ids(ids)
    actual_edges = indexed.select(values)
    np.testing.assert_array_equal(actual_edges.cpu().numpy(), expected_edges.cpu().numpy())
    assert indexed.stats.pearson_pairs == expected_pairs
    assert indexed.stats.kmeans_calls == 2
    assert indexed.stats.transfer_seconds == 0.0
    assert indexed.stats.cache_hits == 2
    assert indexed.stats.cache_misses == 2


def test_batched_edge_assembly_preserves_direction_order_and_batch_mapping():
    windows = [sample_window(44).T, sample_window(45).T, sample_window(46).T]
    values = torch.tensor(np.stack(windows), dtype=torch.float32)
    selector = ClusteringGraphSelector(n_clusters=2, keep_rate=0.7,
                                       cross_edge_ratio=0.25, random_state=9)
    ids = [("train", i) for i in range(len(windows))]
    for sample_id, window in zip(ids, windows):
        selector.prepare_indexed_window(sample_id, window.T)
    selector.set_sample_ids(ids)

    expected = []
    for batch_id, sample_id in enumerate(ids):
        selected = selector._indexed_cache[sample_id]["selected_pairs"]
        src = [left for left, _ in selected] + [right for _, right in selected]
        dst = [right for _, right in selected] + [left for left, _ in selected]
        expected.append([[batch_id, left, right] for left, right in zip(src, dst)])

    actual = selector.select(values)
    np.testing.assert_array_equal(actual.cpu().numpy(), np.asarray(expected, dtype=np.int64))
    assert actual.shape[0] == len(windows)
    assert selector.stats.edge_transfer_seconds >= 0.0

    reference = ClusteringGraphSelector(n_clusters=2, keep_rate=0.7,
                                        cross_edge_ratio=0.25, random_state=9,
                                        batch_edge_assembly=False)
    for sample_id, window in zip(ids, windows):
        reference.prepare_indexed_window(sample_id, window.T)
    reference.set_sample_ids(ids)
    old_edges = reference.select(values)
    torch.testing.assert_close(actual, old_edges, rtol=0, atol=0)


def test_vectorized_kmeans_preserves_reference_labels_candidates_and_edges():
    windows = [sample_window(seed) for seed in range(20, 40)]
    reference = ClusteringGraphSelector(n_clusters=3, keep_rate=0.7,
                                        cross_edge_ratio=0.1, random_state=2021,
                                        kmeans_backend="numpy")
    vectorized = ClusteringGraphSelector(n_clusters=3, keep_rate=0.7,
                                         cross_edge_ratio=0.1, random_state=2021,
                                         kmeans_backend="numpy_vectorized")
    for index, window in enumerate(windows):
        reference.prepare_indexed_window(("train", index), window)
        vectorized.prepare_indexed_window(("train", index), window)
        old = reference._indexed_cache[("train", index)]
        new = vectorized._indexed_cache[("train", index)]
        np.testing.assert_array_equal(new["labels"], old["labels"])
        assert new["intra"] == old["intra"]
        assert new["inter"] == old["inter"]
        assert new["selected_pairs"] == old["selected_pairs"]

    assert vectorized.stats.pearson_pairs == reference.stats.pearson_pairs
    assert vectorized.stats.kmeans_calls == reference.stats.kmeans_calls == len(windows)


def test_default_selector_uses_vectorized_numpy_backend():
    selector = ClusteringGraphSelector(n_clusters=2, keep_rate=0.5,
                                       cross_edge_ratio=0.2, random_state=5)
    assert selector.kmeans_backend == "numpy_vectorized"


def test_vectorized_kmeans_delegates_degenerate_duplicate_features_to_reference():
    # Identical columns yield identical descriptors, fewer distinct feature
    # vectors than K, and exercise the reference's reduced-center fallback.
    window = np.tile(np.linspace(-1.0, 1.0, 48)[:, None], (1, 7))
    reference = ClusteringGraphSelector(n_clusters=3, keep_rate=0.7,
                                        cross_edge_ratio=0.1, random_state=17,
                                        kmeans_backend="numpy")
    vectorized = ClusteringGraphSelector(n_clusters=3, keep_rate=0.7,
                                         cross_edge_ratio=0.1, random_state=17,
                                         kmeans_backend="numpy_vectorized")
    old = reference._prepare_window(window)
    new = vectorized._prepare_window(window)
    np.testing.assert_array_equal(new["labels"], old["labels"])
    assert new["intra"] == old["intra"]
    assert new["inter"] == old["inter"]
    assert new["selected_pairs"] == old["selected_pairs"]


def test_selected_edges_are_symmetric():
    selector = ClusteringGraphSelector(n_clusters=2, keep_rate=0.5,
                                       cross_edge_ratio=0.2, random_state=7)
    edges = selector.select(torch.tensor(np.stack((sample_window(2).T,)), dtype=torch.float32))
    pairs = set(map(tuple, edges[0, :, 1:].tolist()))
    assert all((dst, src) in pairs for src, dst in pairs)


def test_original_graph_generator_and_graphsage_forward_backward():
    from layers.GraphGenerator import graph_generator
    from layers.Graph_layers import dgl_graph_for_trans
    from utils.graph_utils import genearte_graph_from_coo

    selector = ClusteringGraphSelector(n_clusters=2, keep_rate=0.5,
                                       cross_edge_ratio=0.2, random_state=5)
    batch, channels, time_nodes = 2, 4, 3
    edge_index = selector.select(torch.randn(batch, channels, 32), keep_self_cycles=True)
    generator = graph_generator(timenode_num=time_nodes, channel_num=channels,
                                embeded_dim=2, conv_hidden_size=2, conv_kernel_size=3)
    graph_features = torch.randn(batch, channels, time_nodes, 2, requires_grad=True)
    sparse_indices, edge_weights, adjacency = generator(graph_features, edge_index)
    assert adjacency.shape == (batch * channels * time_nodes,) * 2
    assert sparse_indices.ndim == 2 and sparse_indices.shape[1] == 2
    assert edge_weights.ndim == 1

    graph, weights = genearte_graph_from_coo(sparse_indices, edge_weights,
                                             batch * channels * time_nodes)
    assert graph.num_nodes() == batch * channels * time_nodes
    assert graph.num_edges() == sparse_indices.shape[0]
    selected = set(map(tuple, edge_index.reshape(-1, 3).tolist()))
    for source, destination in sparse_indices.tolist():
        sample = source // (channels * time_nodes)
        source_channel = (source % (channels * time_nodes)) // time_nodes
        destination_channel = (destination % (channels * time_nodes)) // time_nodes
        assert (sample, source_channel, destination_channel) in selected
        assert source // (channels * time_nodes) == destination // (channels * time_nodes)
    sage = dgl_graph_for_trans(d_model=4, graph_r=1, graph_hidden_size=8)
    node_features = torch.randn(batch, channels, time_nodes, 4, requires_grad=True)
    output = sage(node_features, graph, weights)
    assert output.shape == node_features.shape
    output.square().mean().backward()
    assert node_features.grad is not None
    assert graph_features.grad is not None
    assert generator.mode_capture_component.grad is not None
    assert torch.isfinite(output).all()
    assert torch.isfinite(node_features.grad).all()
    assert torch.isfinite(graph_features.grad).all()
    assert torch.isfinite(generator.mode_capture_component.grad).all()

    dense = adjacency.to_dense()
    assert torch.allclose(dense, dense.T, atol=1e-4, rtol=0)
    batch_ids_src = sparse_indices[:, 0] // (channels * time_nodes)
    batch_ids_dst = sparse_indices[:, 1] // (channels * time_nodes)
    assert torch.equal(batch_ids_src, batch_ids_dst)


def test_full_model_uses_the_same_selector_in_train_validation_and_test_modes():
    from models.TVCFormer import Model

    config = SimpleNamespace(
        enc_in=4, device=torch.device("cpu"), subgraph_keep_rate=0.5,
        graph_keep_self_cycles=False, eps=1e-5, seq_len=24, pred_len=4,
        e_layers=1, n_heads=2, d_model=8, d_ff=16, dropout=0.0,
        fc_dropout=0.0, head_dropout=0.0, individual=False, patch_len=4,
        stride=2, padding_patch="start", revins=True, affine=False,
        subtract_last=False, decomposition=False, decomposition_kernel_size=3,
        time_node_num=2, gnn_r_degree=1, gnn_hidden_size=8,
        graph_conv_hidden_size=2, graph_conv_kernel_size=3,
    )
    model = Model(config).float()
    selector = ClusteringGraphSelector(n_clusters=7, keep_rate=0.5,
                                       cross_edge_ratio=0.25, random_state=13)
    model.model.graph_selector = selector
    x = torch.randn(2, config.seq_len, config.enc_in)
    target = torch.randn(2, config.pred_len, config.enc_in)

    model.train()
    train_pred = model(x)
    torch.nn.functional.mse_loss(train_pred, target).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert selector.stats.samples == x.shape[0]

    model.eval()
    with torch.no_grad():
        val_pred = model(x)
    assert val_pred.shape == target.shape
    assert selector.stats.samples == 2 * x.shape[0]
    with torch.no_grad():
        test_pred = model(x)
    assert test_pred.shape == target.shape
    assert selector.stats.samples == 3 * x.shape[0]
    assert model.model.graph_selector is selector

    reference = ClusteringGraphSelector(n_clusters=7, keep_rate=0.5,
                                        cross_edge_ratio=0.25, random_state=13,
                                        batch_edge_assembly=False)
    model.model.graph_selector = reference
    with torch.no_grad():
        reference_pred = model(x)
    torch.testing.assert_close(test_pred, reference_pred, rtol=0, atol=0)


def test_batched_precomputed_history_cache_supports_model_forward_backward_and_reuse():
    from models.TVCFormer import Model

    config = SimpleNamespace(
        enc_in=4, device=torch.device("cpu"), subgraph_keep_rate=0.5,
        graph_keep_self_cycles=False, eps=1e-5, seq_len=24, pred_len=4,
        e_layers=1, n_heads=2, d_model=8, d_ff=16, dropout=0.0,
        fc_dropout=0.0, head_dropout=0.0, individual=False, patch_len=4,
        stride=2, padding_patch="start", revins=True, affine=False,
        subtract_last=False, decomposition=False, decomposition_kernel_size=3,
        time_node_num=2, gnn_r_degree=1, gnn_hidden_size=8,
        graph_conv_hidden_size=2, graph_conv_kernel_size=3,
    )
    model = Model(config).float()
    selector = ClusteringGraphSelector(n_clusters=2, keep_rate=0.5,
                                       cross_edge_ratio=0.25, random_state=2021)
    model.model.graph_selector = selector
    x = torch.randn(2, config.seq_len, config.enc_in)
    ids = [("train", 7), ("train", 19)]
    selector.prepare_indexed_windows_batch(ids, x.numpy())
    selector.set_sample_ids(ids)

    model.train()
    prediction = model(x)
    torch.nn.functional.mse_loss(prediction, torch.randn_like(prediction)).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert selector.stats.cache_misses == selector.stats.kmeans_calls == 2

    model.eval()
    with torch.no_grad():
        val_prediction = model(x)
        test_prediction = model(x)
    assert val_prediction.shape == test_prediction.shape == prediction.shape
    assert selector.stats.cache_hits == 6
    assert selector.stats.kmeans_calls == 2
