"""Forecast-trained Siamese graph generator and learned candidate selector."""
import time
import torch
from layers.GraphGenerator import graph_generator
from utils.graph_utils import subgraph_tensor_to_full_graph_sparse_tensor
from utils.normal_utils import get_statistics_full
from utils.correlation_utils import get_pearson_correlation_vector
from FinalProject.Siamese_TVC.models.similarity import (
    SiameseWindowSimilarity, select_channel_candidates,
)


class GraphGeneratorSiamese(graph_generator):
    def __init__(self, timenode_num, channel_num, statistics_dim=2,
                 conv_hidden_size=4, conv_kernel_size=3, hidden_dim=32,
                 embedding_dim=16, gate_init=-3.0, gate_mode='learned',
                 candidate_mode='hybrid', candidate_max_keep_rate=1.0,
                 cnn_dim=32, measure_graph_time=False):
        super().__init__(timenode_num, channel_num, statistics_dim,
                         conv_hidden_size, conv_kernel_size, use_sdei=False)
        self.similarity = SiameseWindowSimilarity(
            statistics_dim, hidden_dim, embedding_dim, gate_init, gate_mode, cnn_dim)
        if candidate_mode not in ('pearson', 'siamese', 'hybrid'):
            raise ValueError('candidate_mode must be pearson, siamese, or hybrid')
        self.candidate_mode = candidate_mode
        self.candidate_max_keep_rate = candidate_max_keep_rate
        self.measure_graph_time = measure_graph_time
        self.last_gate = None
        self.last_edge_index = None
        self.last_graph = None
        self.last_embeddings = None
        self.last_local_weights = None
        self._prepared_embedding = None
        self.last_graph_build_seconds = 0.0
        self.last_pearson = None
        self.last_candidate_diagnostics = None
        self._candidate_seconds_pending = 0.0
        self.graph_build_seconds_total = 0.0
        self.last_candidate_diagnostics = None

    def prepare_candidates(self, z, backbone):
        """Select graph support from the current context and cache its embedding."""
        if self.measure_graph_time and z.is_cuda:
            torch.cuda.synchronize(z.device)
        started = time.perf_counter() if self.measure_graph_time else 0.0
        padded = backbone.padding_patch_layer(z)
        windows = torch.stack(torch.chunk(padded, self.timenode_num, dim=-1), dim=-2)
        stats = get_statistics_full(
            z.permute(0, 2, 1), padding_len=backbone.x_padding_len,
            padding_mode=backbone.padding_patch, block_num=self.timenode_num,
            subtract_last=backbone.subtract_last, eps=backbone.eps)
        stats = backbone.statistic_norm(stats.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        embedding = self.similarity(stats, windows)
        pearson = get_pearson_correlation_vector(z).reshape(
            z.shape[0], self.channel_num, self.channel_num)
        self.last_pearson = pearson.detach()
        siamese = self.similarity.channel_scores(embedding)
        edges, details = select_channel_candidates(
            pearson, siamese, backbone.subgraph_keep_rate,
            mode=self.candidate_mode,
            max_keep_rate=self.candidate_max_keep_rate,
            keep_self_cycles=backbone.graph_keep_self_cycles,
            return_details=True)
        self.last_candidate_diagnostics = {
            key: value.detach() for key, value in details.items()
        }
        # The official backbone calls the graph generator immediately after
        # selector.select, allowing this differentiable embedding to be reused.
        self._prepared_embedding = (stats, windows, embedding)
        if self.measure_graph_time:
            if z.is_cuda:
                torch.cuda.synchronize(z.device)
            self._candidate_seconds_pending = time.perf_counter() - started
        return edges

    def clear_candidate_cache(self):
        self._prepared_embedding = None

    def forward(self, x_in, index_vector, windows=None):
        if windows is None:
            raise ValueError('Siamese_TVC requires local windows from the input context')
        if self.measure_graph_time and x_in.is_cuda:
            torch.cuda.synchronize(x_in.device)
        started = time.perf_counter() if self.measure_graph_time else 0.0
        cached = self._prepared_embedding
        self._prepared_embedding = None
        if cached is not None and cached[0].shape == x_in.shape:
            embedding = cached[2]
        else:
            embedding = self.similarity(x_in, windows)
        edges = index_vector.long()
        gate = self.similarity.gate_logit.new_tensor(1.0)
        if self.similarity.gate_mode == 'learned':
            gate = torch.sigmoid(self.similarity.gate_logit)
        elif self.similarity.gate_mode == 'statistical':
            gate = gate.new_zeros(())
        if edges.shape[1] == 0:
            # A one-channel graph, or a floor-rounded zero Top-K budget, has no
            # cross-channel edges; still return a valid empty sparse graph.
            local_weights = x_in.new_empty((x_in.shape[0], 0, self.timenode_num, self.timenode_num))
            sparse_indices = torch.empty((0, 2), dtype=torch.long, device=x_in.device)
            sparse_values = x_in.new_empty((0,))
        else:
            local_weights, gate = self.similarity.edge_weights(x_in, embedding, edges)
            local_weights = self.sparse_module(local_weights)
            sparse_indices, sparse_values = subgraph_tensor_to_full_graph_sparse_tensor(
                local_weights, edges, self.channel_num, self.timenode_num)
        total_nodes = local_weights.shape[0] * self.channel_num * self.timenode_num
        adjacency = torch.sparse_coo_tensor(
            sparse_indices.T, sparse_values, (total_nodes, total_nodes),
            device=local_weights.device).coalesce()
        self.last_gate = gate.detach()
        self.last_edge_index = edges.detach()
        self.last_graph = adjacency.detach()
        self.last_embeddings = embedding.detach()
        self.last_local_weights = local_weights.detach()
        if self.measure_graph_time and local_weights.is_cuda:
            torch.cuda.synchronize(local_weights.device)
        self.last_graph_build_seconds = (
            self._candidate_seconds_pending + time.perf_counter() - started
            if self.measure_graph_time else 0.0)
        self._candidate_seconds_pending = 0.0
        self.graph_build_seconds_total += self.last_graph_build_seconds
        return sparse_indices, sparse_values, adjacency


class CandidateSupportSelector:
    """Plain selector adapter; deliberately does not register duplicate modules."""
    def __init__(self, graph_generator_module, backbone):
        self.graph_generator = graph_generator_module
        self.backbone = backbone

    def select(self, z, keep_rate, keep_self_cycles):
        return self.graph_generator.prepare_candidates(z, self.backbone)
