# Adaptive Time-Variable Coupling graph generator for ATVF.
# Keeps the exact sparse-graph interface expected by TVCFormer's DGL encoder.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.graph_utils import subgraph_tensor_to_full_graph_sparse_tensor


class ATVFGraphGenerator(nn.Module):
    def __init__(
        self,
        timenode_num,
        channel_num,
        embeded_dim=2,
        hidden_dim=16,
        edge_keep_rate=0.35,
        lambda_stability=0.01,
        lambda_sparsity=0.001,
        lambda_smooth=0.001,
    ):
        super().__init__()
        if not 0.0 < edge_keep_rate <= 1.0:
            raise ValueError("edge_keep_rate must be in (0, 1].")

        self.timenode_num = int(timenode_num)
        self.channel_num = int(channel_num)
        self.edge_keep_rate = float(edge_keep_rate)
        self.lambda_stability = float(lambda_stability)
        self.lambda_sparsity = float(lambda_sparsity)
        self.lambda_smooth = float(lambda_smooth)

        self.query = nn.Linear(embeded_dim, hidden_dim, bias=False)
        self.key = nn.Linear(embeded_dim, hidden_dim, bias=False)
        self.channel_bias = nn.Parameter(torch.zeros(channel_num, channel_num))
        self.mix_logit = nn.Parameter(torch.tensor(0.0))
        self.last_aux_loss = None

    def forward(self, x_in, index_vector):
        # x_in: [B, C, T, D]; index_vector: [B, S, 3] = (batch, src, dst)
        if x_in.ndim != 4:
            raise ValueError(f"x_in must be [B,C,T,D], got {tuple(x_in.shape)}")
        batch_size, channel_count, time_count, _ = x_in.shape
        if channel_count != self.channel_num or time_count != self.timenode_num:
            raise ValueError("ATVF graph dimensions do not match the configured channel/time-node counts.")

        src_idx = index_vector[:, :, 1].long()
        dst_idx = index_vector[:, :, 2].long()
        src = torch.gather(
            x_in, 1,
            src_idx[:, :, None, None].expand(-1, -1, time_count, x_in.size(-1))
        )
        dst = torch.gather(
            x_in, 1,
            dst_idx[:, :, None, None].expand(-1, -1, time_count, x_in.size(-1))
        )

        # Feature-dependent temporal coupling for every selected channel pair.
        q = self.query(src)
        k = self.key(dst)
        dynamic = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        dynamic = torch.tanh(dynamic)

        # Correlation-derived prior is retained, while learnable channel bias
        # and input-conditioned attention adapt the graph at every batch.
        corr = F.cosine_similarity(src.unsqueeze(-2), dst.unsqueeze(-3), dim=-1)
        bias = 0.5 * (
            self.channel_bias[src_idx, dst_idx] +
            self.channel_bias[dst_idx, src_idx]
        )
        mix = torch.sigmoid(self.mix_logit)
        logits = mix * dynamic + (1.0 - mix) * corr + bias[..., None, None]
        logits = 0.5 * (logits + logits.transpose(-1, -2))
        probabilities = torch.sigmoid(logits)

        # Symmetric top-k selection over undirected off-diagonal edges.
        tri = torch.triu_indices(time_count, time_count, offset=1, device=x_in.device)
        candidate_scores = probabilities[..., tri[0], tri[1]]
        keep_count = max(1, int(math.ceil(
            self.edge_keep_rate * candidate_scores.size(-1)
        )))
        keep_indices = torch.topk(candidate_scores, k=keep_count, dim=-1).indices
        mask_upper = torch.zeros_like(candidate_scores).scatter_(
            -1, keep_indices, 1.0
        )
        mask = probabilities.new_zeros(probabilities.shape)
        mask[..., tri[0], tri[1]] = mask_upper
        mask = mask + mask.transpose(-1, -2)
        selected_adj = probabilities * mask

        # Regularizers: stay near the correlation prior, avoid unnecessarily
        # large edge mass, and encourage smooth coupling over adjacent nodes.
        stability = F.mse_loss(probabilities, torch.sigmoid(corr.detach()))
        sparsity = selected_adj.mean()
        smooth_t = (selected_adj[:, :, 1:, :] - selected_adj[:, :, :-1, :]).abs().mean()
        smooth_s = (selected_adj[:, :, :, 1:] - selected_adj[:, :, :, :-1]).abs().mean()
        self.last_aux_loss = (
            self.lambda_stability * stability
            + self.lambda_sparsity * sparsity
            + self.lambda_smooth * (smooth_t + smooth_s)
        )

        sparse_indices, sparse_values = subgraph_tensor_to_full_graph_sparse_tensor(
            selected_adj, index_vector, self.channel_num, self.timenode_num
        )
        total_node_num = batch_size * self.channel_num * self.timenode_num
        adj = torch.sparse_coo_tensor(
            sparse_indices.T, sparse_values,
            size=(total_node_num, total_node_num),
            device=x_in.device,
        ).coalesce()
        return sparse_indices, sparse_values, adj
