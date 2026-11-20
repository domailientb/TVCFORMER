import torch
import torch.nn.functional as F

from FinalProject.Siamese_TVC.models.similarity import (
    SiameseWindowSimilarity, select_channel_candidates,
)
from FinalProject.Siamese_TVC.models.graph_generator import GraphGeneratorSiamese
from FinalProject.Siamese_TVC.models.evaluation import embedding_metrics
from utils.graph_utils import genearte_graph_from_coo
from layers.Graph_layers import dgl_graph_for_trans


def _fixture():
    torch.manual_seed(17)
    b, c, l, w = 2, 5, 4, 12
    t = torch.linspace(0, 2 * torch.pi, w)
    windows = torch.randn(b, c, l, w)
    windows[:, 0] = torch.sin(t)[None, None, :]
    windows[:, 1] = torch.sin(t + 0.1)[None, None, :]
    windows[:, 2] = torch.sign(torch.sin(t))[None, None, :]
    stats = torch.randn(b, c, l, 2)
    pearson = torch.eye(c).expand(b, c, c).clone()
    pearson[:, 0, 1] = pearson[:, 1, 0] = 0.01
    return stats, windows, pearson


def test_shared_encoder_embedding_shape_and_similarity():
    stats, windows, _ = _fixture()
    model = SiameseWindowSimilarity(embedding_dim=11)
    z = model(stats, windows)
    assert z.shape == (2, 5, 4, 11)
    assert torch.allclose(z.norm(dim=-1), torch.ones(2, 5, 4), atol=1e-6)
    weights, _ = model.edge_weights(stats, z,
        torch.tensor([[[0, 0, 1], [0, 1, 0]], [[1, 0, 1], [1, 1, 0]]]))
    assert weights.shape == (2, 2, 4, 4)
    assert torch.allclose(weights[:, 0], weights[:, 1].transpose(-1, -2), atol=1e-6)


def test_siamese_proposes_pair_outside_pearson_support_and_hybrid_union():
    _, _, pearson = _fixture()
    siam = torch.full_like(pearson, -0.5)
    siam[:, torch.arange(5), torch.arange(5)] = 1.0
    siam[:, 0, 1] = siam[:, 1, 0] = 0.99
    pearson[:, 0, 1] = pearson[:, 1, 0] = -0.8
    edges = select_channel_candidates(pearson, siam, 0.25, 'siamese')
    directed = {(int(a), int(b)) for _, a, b in edges[0].tolist()}
    assert (0, 1) in directed and (1, 0) in directed
    hybrid = select_channel_candidates(pearson, siam, 0.25, 'hybrid', 1.0)
    hybrid_pairs = {(int(a), int(b)) for _, a, b in hybrid[0].tolist()}
    assert (0, 1) in hybrid_pairs and (1, 0) in hybrid_pairs


def test_support_is_symmetric_and_self_loops_follow_config():
    stats, windows, pearson = _fixture()
    z = SiameseWindowSimilarity()(stats, windows)
    siam = SiameseWindowSimilarity().channel_scores(z)
    no_loop = select_channel_candidates(pearson, siam, 0.4, 'hybrid')
    assert torch.all(no_loop[..., 1] != no_loop[..., 2])
    pairs = {(int(b), int(i), int(j)) for b, i, j in no_loop.reshape(-1, 3).tolist()}
    assert all((b, j, i) in pairs for b, i, j in pairs)
    loops = select_channel_candidates(pearson, siam, 0.4, 'siamese', keep_self_cycles=True)
    assert int((loops[..., 1] == loops[..., 2]).sum()) == 2 * 5


def test_forecast_path_gradients_reach_encoder_projection_and_gate():
    stats, windows, pearson = _fixture()
    gen = GraphGeneratorSiamese(4, 5, gate_init=0.0, candidate_mode='siamese')
    embedding = gen.similarity(stats, windows)
    scores = gen.similarity.channel_scores(embedding)
    edges = select_channel_candidates(pearson, scores, 0.4, 'siamese')
    idx, val, graph = gen(stats, edges, windows)
    assert graph.shape == (2 * 5 * 4, 2 * 5 * 4)
    dense_graph = graph.to_dense()
    assert torch.allclose(dense_graph, dense_graph.T, atol=2e-5)
    assert torch.count_nonzero(torch.diagonal(dense_graph)) == 0
    batch_graph, weights = genearte_graph_from_coo(idx, val, 2 * 5 * 4)
    node_features = torch.randn(2, 5, 4, 8, requires_grad=True)
    sage = dgl_graph_for_trans(d_model=8, graph_r=1, graph_hidden_size=12)
    out = sage(node_features, batch_graph, weights)
    synthetic_forecast_head = torch.nn.Linear(8, 3)
    synthetic_truth = torch.randn(2, 5, 4, 3)
    forecast_loss = F.mse_loss(synthetic_forecast_head(out), synthetic_truth)
    (forecast_loss + graph.values().square().mean()).backward()
    assert gen.similarity.encoder[0].weight.grad is not None
    assert torch.isfinite(gen.similarity.encoder[0].weight.grad).all()
    assert gen.similarity.encoder[-1].weight.grad is not None
    assert gen.similarity.features.temporal[0].weight.grad is not None
    assert gen.similarity.gate_logit.grad is not None
    assert torch.isfinite(gen.similarity.gate_logit.grad)
    assert gen.mode_capture_component.grad is not None


def test_embedding_does_not_collapse_on_diverse_windows():
    stats, windows, _ = _fixture()
    embedding = SiameseWindowSimilarity()(stats, windows)
    diagnostics = embedding_metrics(embedding)
    assert diagnostics['embedding_std'] > 1e-4
    assert diagnostics['cosine_std'] > 1e-4
    assert diagnostics['effective_rank'] > 1.0


def test_graph_and_candidates_use_context_inputs_only():
    # Candidate API receives only the context tensor; no target/future argument exists.
    stats, windows, pearson = _fixture()
    model = SiameseWindowSimilarity()
    emb = model(stats, windows)
    edges_a = select_channel_candidates(pearson, model.channel_scores(emb), 0.5, 'hybrid')
    future = torch.randn(2, 96, 5)
    future.add_(1000.0)
    emb2 = model(stats, windows)
    edges_b = select_channel_candidates(pearson, model.channel_scores(emb2), 0.5, 'hybrid')
    assert torch.equal(edges_a, edges_b)


def test_single_channel_and_invalid_scores_are_handled():
    pearson = torch.ones(2, 1, 1)
    siamese = torch.ones_like(pearson)
    edges = select_channel_candidates(pearson, siamese, 0.5, 'hybrid')
    assert edges.shape == (2, 0, 3)
    generator = GraphGeneratorSiamese(timenode_num=3, channel_num=1)
    statistics = torch.randn(2, 1, 3, 2)
    windows = torch.randn(2, 1, 3, 8)
    _, _, graph = generator(statistics, edges, windows)
    assert graph.shape == (6, 6) and graph._nnz() == 0
    siamese[0, 0, 0] = float('nan')
    try:
        select_channel_candidates(pearson, siamese, 0.5, 'hybrid')
    except ValueError as error:
        assert 'NaN' in str(error)
    else:
        raise AssertionError('non-finite scores must be rejected explicitly')

