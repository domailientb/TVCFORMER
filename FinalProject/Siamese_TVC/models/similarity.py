"""Shared encoder and candidate selection for local TVC time windows."""
import torch
from torch import nn
import torch.nn.functional as F


class WindowFeatures(nn.Module):
    """Scale-aware descriptors plus a shared temporal CNN for [B,C,L,W]."""
    def __init__(self, cnn_dim=32):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(16, cnn_dim, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, windows):
        eps = 1e-5
        mean = windows.mean(dim=-1, keepdim=True)
        std = windows.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        norm = (windows - mean) / std
        diff = windows[..., 1:] - windows[..., :-1]
        if diff.shape[-1] == 0:
            diff = torch.zeros_like(windows[..., :1])
        dstd = diff.std(dim=-1, keepdim=True, unbiased=False)
        if norm.shape[-1] > 1:
            left, right = norm[..., :-1], norm[..., 1:]
            lag = (left * right).mean(dim=-1, keepdim=True) / (
                left.std(dim=-1, keepdim=True, unbiased=False)
                * right.std(dim=-1, keepdim=True, unbiased=False) + eps)
        else:
            lag = torch.zeros_like(mean)
        slope = (norm[..., -1:] - norm[..., :1]) / max(windows.shape[-1] - 1, 1)
        descriptors = torch.cat((mean, torch.log(std), slope,
            dstd / std, diff.abs().mean(dim=-1, keepdim=True) / std, lag), dim=-1)
        # Shape CNN sees scale-normalized windows; level/scale remain available
        # explicitly in descriptors and the TVC statistics.
        flat = norm.reshape(-1, 1, norm.shape[-1])
        temporal = self.temporal(flat).squeeze(-1).reshape(*windows.shape[:3], -1)
        return torch.cat((descriptors, temporal), dim=-1)


class SiameseWindowSimilarity(nn.Module):
    """One shared temporal encoder; forecast loss learns its cosine geometry."""
    def __init__(self, statistics_dim=2, hidden_dim=32, embedding_dim=16,
                 gate_init=-3.0, gate_mode='learned', cnn_dim=32):
        super().__init__()
        if gate_mode not in ('learned', 'statistical', 'siamese'):
            raise ValueError('gate_mode must be learned, statistical, or siamese')
        self.features = WindowFeatures(cnn_dim)
        self.encoder = nn.Sequential(
            nn.Linear(statistics_dim + 6 + cnn_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.gate_mode = gate_mode
        if gate_mode != 'learned':
            self.gate_logit.requires_grad_(False)

    def forward(self, statistics, windows):
        if statistics.ndim != 4 or windows.ndim != 4:
            raise ValueError('Expected statistics [B,C,L,F] and windows [B,C,L,W]')
        if statistics.shape[:3] != windows.shape[:3]:
            raise ValueError('statistics and windows must share [B,C,L] dimensions')
        if not torch.isfinite(statistics).all() or not torch.isfinite(windows).all():
            raise ValueError('Siamese inputs contain NaN or infinite values')
        features = self.features(windows)
        embedding = F.normalize(self.encoder(torch.cat((statistics, features), dim=-1)), dim=-1)
        return embedding

    def channel_scores(self, embedding):
        """Aligned-window mean cosine used only to choose channel candidates."""
        # [B,C,C,L], without materializing [B,C,C,L,L].
        pairwise = torch.einsum('bcld,bjld->bcjl', embedding, embedding)
        return pairwise.mean(dim=-1)

    def edge_weights(self, statistics, embedding, edges):
        batch, source, destination = edges.long().unbind(-1)
        z_src, z_dst = embedding[batch, source], embedding[batch, destination]
        e_src, e_dst = statistics[batch, source], statistics[batch, destination]
        learned = torch.matmul(z_src, z_dst.transpose(-1, -2)).clamp(-1, 1)
        statistical = torch.matmul(F.normalize(e_src, dim=-1),
                                    F.normalize(e_dst, dim=-1).transpose(-1, -2)).clamp(-1, 1)
        gate = torch.sigmoid(self.gate_logit)
        if self.gate_mode == 'statistical':
            gate = gate.new_zeros(())
        elif self.gate_mode == 'siamese':
            gate = gate.new_ones(())
        return (1.0 - gate) * statistical + gate * learned, gate


def select_channel_candidates(pearson, siamese, keep_rate, mode='hybrid',
                              max_keep_rate=1.0, keep_self_cycles=False,
                              return_details=False):
    """Return symmetric directed support [B,2K,3]. Self loops are excluded.

    Pearson selects the baseline's top positive correlations. Siamese selects
    aligned-window embedding similarity. Hybrid takes their union, then applies
    the baseline Top-K edge budget by Borda rank fusion.
    """
    if mode not in ('pearson', 'siamese', 'hybrid'):
        raise ValueError('candidate_mode must be pearson, siamese, or hybrid')
    if not (0 < keep_rate <= 1 and 0 < max_keep_rate <= 1):
        raise ValueError('candidate rates must be in (0, 1]')
    if pearson.ndim != 3 or siamese.shape != pearson.shape:
        raise ValueError('pearson and siamese scores must both have shape [B,C,C]')
    if pearson.shape[-1] != pearson.shape[-2]:
        raise ValueError('channel score matrices must be square')
    if pearson.shape[-1] < 1:
        raise ValueError('at least one channel is required')
    if not torch.isfinite(pearson).all() or not torch.isfinite(siamese).all():
        raise ValueError('candidate scores contain NaN or infinite values')
    batch, channels, _ = pearson.shape
    upper = torch.triu_indices(channels, channels, offset=1, device=pearson.device)
    count = upper.shape[1]
    p = pearson[:, upper[0], upper[1]]
    s = siamese[:, upper[0], upper[1]]
    k = min(count, int(keep_rate * count), int(max_keep_rate * count))
    p_idx = torch.topk(p, k, dim=-1, largest=True, sorted=False).indices
    s_idx = torch.topk(s, k, dim=-1, largest=True, sorted=False).indices
    p_mask = torch.zeros(batch, count, dtype=torch.bool, device=pearson.device)
    s_mask = torch.zeros_like(p_mask)
    p_mask.scatter_(1, p_idx, True)
    s_mask.scatter_(1, s_idx, True)
    if mode == 'pearson':
        selected = p_idx
    elif mode == 'siamese':
        selected = s_idx
    else:
        union_mask = p_mask | s_mask
        # Keep the same edge budget as E0/E1, independent of batch overlap.
        budget = min(k, int(union_mask.sum(dim=-1).min().item()))
        # Borda-style rank fusion avoids comparing Pearson and cosine scales.
        p_rank = torch.argsort(torch.argsort(p, dim=-1), dim=-1).float()
        s_rank = torch.argsort(torch.argsort(s, dim=-1), dim=-1).float()
        combined = (p_rank + s_rank) / max(count - 1, 1)
        combined = combined.masked_fill(~union_mask, float('-inf'))
        selected = torch.topk(combined, budget, dim=-1,
                              largest=True, sorted=False).indices
    pair_i = upper[0][selected]
    pair_j = upper[1][selected]
    selected_mask = torch.zeros(batch, count, dtype=torch.bool, device=pearson.device)
    selected_mask.scatter_(1, selected, True)
    selected_p = selected_mask & p_mask
    selected_s = selected_mask & s_mask
    denom = selected_mask.sum(dim=-1).clamp_min(1)
    details = {
        'selected_undirected': selected_mask.sum(dim=-1),
        'union_candidates': (p_mask | s_mask).sum(dim=-1),
        'pearson_only': (selected_p & ~selected_s).sum(dim=-1),
        'siamese_only': (selected_s & ~selected_p).sum(dim=-1),
        'shared': (selected_p & selected_s).sum(dim=-1),
        'pearson_fraction': selected_p.sum(dim=-1).float() / denom,
        'siamese_fraction': selected_s.sum(dim=-1).float() / denom,
        'pearson_missed_fraction': (selected_s & ~selected_p).sum(dim=-1).float() / denom,
    }
    bidx = torch.arange(batch, device=pearson.device).unsqueeze(1).expand_as(pair_i)
    fwd = torch.stack((bidx, pair_i, pair_j), dim=-1)
    rev = torch.stack((bidx, pair_j, pair_i), dim=-1)
    directed = torch.cat((fwd, rev), dim=1)
    if keep_self_cycles:
        nodes = torch.arange(channels, device=pearson.device)
        bnodes = torch.arange(batch, device=pearson.device)[:, None].expand(batch, channels)
        loops = torch.stack((bnodes, nodes[None, :].expand(batch, -1),
                             nodes[None, :].expand(batch, -1)), dim=-1)
        directed = torch.cat((directed, loops), dim=1)
    return (directed, details) if return_details else directed
