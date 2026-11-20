"""Embedding and graph quality diagnostics (descriptive, not proof of utility)."""
import torch
import torch.nn.functional as F


def embedding_metrics(embedding, near_duplicate_threshold=0.98):
    """Summarize embedding spread and cosine geometry for [B,C,L,D]."""
    z = F.normalize(embedding.detach().reshape(-1, embedding.shape[-1]), dim=-1)
    if z.shape[0] < 2:
        cos = z.new_empty(0)
    else:
        sim = z @ z.T
        mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
        cos = sim[mask]
    return {
        'embedding_mean': float(embedding.mean().item()),
        'embedding_std': float(embedding.std(unbiased=False).item()),
        'cosine_mean': float(cos.mean().item()) if cos.numel() else float('nan'),
        'cosine_std': float(cos.std(unbiased=False).item()) if cos.numel() else float('nan'),
        'near_duplicate_fraction': float((cos >= near_duplicate_threshold).float().mean().item()) if cos.numel() else 0.0,
        'effective_rank': _effective_rank(z),
    }


def _effective_rank(z):
    centered = z - z.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    power = singular.square()
    probs = power / power.sum().clamp_min(torch.finfo(power.dtype).eps)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().item())


def graph_metrics(adjacency, channel_count=None, time_node_count=None):
    """Statistics for a dense or sparse adjacency; excludes diagonal from density."""
    if adjacency.is_sparse:
        a = adjacency.detach().coalesce()
        n = a.shape[-1]
        indices, values = a.indices(), a.values()
        offdiag = (indices[0] != indices[1]) & values.ne(0)
        degree = torch.bincount(indices[0, offdiag], minlength=n).float()
        weights = values[offdiag]
        symmetry_error = _sparse_symmetry_error(a)
        self_loop_count = int((indices[0] == indices[1]).sum().item())
        edge_count = int(offdiag.sum().item())
    else:
        a = adjacency.detach()
        if a.ndim == 3:
            a = a[0]
        n = a.shape[-1]
        offdiag = ~torch.eye(n, dtype=torch.bool, device=a.device)
        nz = (a.abs() > 0) & offdiag
        degree = nz.sum(dim=-1).float()
        weights = a[offdiag & (a.abs() > 0)]
        symmetry_error = float((a - a.T).abs().max().item())
        self_loop_count = int(torch.diagonal(a).ne(0).sum().item())
        edge_count = int(nz.sum().item())
    if channel_count and time_node_count:
        nodes_per_sample = int(channel_count) * int(time_node_count)
        sample_count = max(n // nodes_per_sample, 1)
        possible_edges = sample_count * nodes_per_sample * (nodes_per_sample - 1)
    else:
        possible_edges = n * (n - 1)
    return {
        'node_count': int(n), 'edge_count_directed': edge_count,
        'density_offdiag': float(edge_count / max(possible_edges, 1)),
        'degree_mean': float(degree.mean().item()),
        'degree_std': float(degree.std(unbiased=False).item()),
        'weight_mean': float(weights.mean().item()) if weights.numel() else 0.0,
        'weight_std': float(weights.std(unbiased=False).item()) if weights.numel() else 0.0,
        'weight_min': float(weights.min().item()) if weights.numel() else 0.0,
        'weight_max': float(weights.max().item()) if weights.numel() else 0.0,
        'symmetry_max_error': symmetry_error,
        'self_loop_count': self_loop_count,
        'channel_count': channel_count, 'time_node_count': time_node_count,
    }


def graph_similarity(first, second):
    """Cosine similarity and edge Jaccard for equally shaped adjacency matrices."""
    if first.shape != second.shape:
        raise ValueError('graphs must have identical shapes')
    if first.is_sparse and second.is_sparse:
        a, b = first.detach().coalesce(), second.detach().coalesce()
        n = a.shape[-1]
        ka = a.indices()[0] * n + a.indices()[1]
        kb = b.indices()[0] * n + b.indices()[1]
        keys = torch.unique(torch.cat((ka, kb)), sorted=True)
        av = torch.zeros(keys.numel(), device=a.device, dtype=a.values().dtype)
        bv = torch.zeros(keys.numel(), device=b.device, dtype=b.values().dtype)
        av[torch.searchsorted(keys, ka)] = a.values()
        bv[torch.searchsorted(keys, kb)] = b.values()
        ea, eb = av.ne(0), bv.ne(0)
        cosine = F.cosine_similarity(av, bv, dim=0, eps=1e-8)
    else:
        a = first.detach().to_dense() if first.is_sparse else first.detach()
        b = second.detach().to_dense() if second.is_sparse else second.detach()
        av, bv = a.flatten(), b.flatten()
        ea, eb = av.ne(0), bv.ne(0)
        cosine = F.cosine_similarity(av, bv, dim=0, eps=1e-8)
    union = (ea | eb).sum()
    jaccard = (ea & eb).sum().float() / union.clamp_min(1)
    return {'weight_cosine': float(cosine.item()), 'edge_jaccard': float(jaccard.item())}


def _sparse_symmetry_error(adjacency):
    indices, values = adjacency.indices(), adjacency.values()
    n = adjacency.shape[-1]
    keys = indices[0] * n + indices[1]
    reverse = indices[1] * n + indices[0]
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    pos = torch.searchsorted(sorted_keys, reverse)
    valid = pos < sorted_keys.numel()
    found = torch.zeros_like(valid)
    found[valid] = sorted_keys[pos[valid]] == reverse[valid]
    reverse_values = torch.zeros_like(values)
    reverse_values[found] = values[order[pos[found]]]
    return float((values - reverse_values).abs().max().item()) if values.numel() else 0.0


def pearson_missed_candidates(pearson, edge_index, threshold=0.2):
    """Count selected undirected pairs below absolute Pearson threshold."""
    edges = edge_index.detach().long()
    if edges.numel() == 0:
        return {'count': 0, 'fraction': 0.0}
    if edges.ndim == 3:
        pairs = edges.reshape(-1, 3)
    elif edges.ndim == 2 and edges.shape[-1] == 3:
        pairs = edges
    else:
        raise ValueError('edge_index must have shape [B,E,3] or [E,3]')
    batch, src, dst = pairs.unbind(-1)
    values = pearson[batch, src, dst].abs()
    mask = src != dst
    missed = mask & (values < threshold)
    denom = mask.sum().clamp_min(1)
    return {'count': int(missed.sum().item()), 'fraction': float((missed.sum() / denom).item())}
