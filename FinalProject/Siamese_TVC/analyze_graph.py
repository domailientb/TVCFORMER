"""Analyze and visualize saved *_graph.npz diagnostics."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from FinalProject.Siamese_TVC.models.evaluation import graph_metrics, graph_similarity


def load_graph(path):
    data = np.load(path)
    shape = tuple(data['graph_shape'].tolist())
    graph = torch.sparse_coo_tensor(torch.from_numpy(data['graph_indices']),
                                    torch.from_numpy(data['graph_values']), shape).coalesce()
    return data, graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graphs', nargs='+', help='Saved graph NPZ files')
    parser.add_argument('--output_dir', default='FinalProject/Siamese_TVC/graph_analysis')
    parser.add_argument('--channels', type=int, default=7)
    parser.add_argument('--time_nodes', type=int, default=7)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    loaded = [(Path(path), *load_graph(path)) for path in args.graphs]
    rows = []
    for path, data, graph in loaded:
        metrics = graph_metrics(graph, args.channels, args.time_nodes)
        metrics.update({'file': str(path), 'gate': float(data['gate']),
                        'graph_build_seconds': float(data['graph_build_seconds'])})
        rows.append(metrics)
        nodes = args.channels * args.time_nodes
        first = torch.zeros((nodes, nodes), dtype=graph.values().dtype)
        indices, values = graph.indices(), graph.values()
        in_first = (indices[0] < nodes) & (indices[1] < nodes)
        first[indices[0, in_first].cpu(), indices[1, in_first].cpu()] = values[in_first].cpu()
        first = first.numpy()
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(first, cmap='magma', aspect='auto')
        ax.set(title=f'{path.stem}: first sample TVC graph', xlabel='destination node', ylabel='source node')
        fig.colorbar(im, ax=ax, label='edge weight')
        fig.tight_layout(); fig.savefig(out / f'{path.stem}_graph.png', dpi=160); plt.close(fig)

        if 'embeddings' in data:
            emb = torch.from_numpy(data['embeddings']).float().mean(dim=2)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            similarity = (emb[0] @ emb[0].T).cpu().numpy()
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(similarity, vmin=-1, vmax=1, cmap='coolwarm')
            ax.set(title=f'{path.stem}: channel embedding similarity', xlabel='channel', ylabel='channel')
            fig.colorbar(im, ax=ax); fig.tight_layout()
            fig.savefig(out / f'{path.stem}_similarity.png', dpi=160); plt.close(fig)
            pearson = torch.from_numpy(data['pearson']).float()
            edges = torch.from_numpy(data['edge_index'])
            rows[-1]['low_pearson_candidate_fraction'] = _low_pearson_fraction(pearson, edges)
            rows[-1]['embedding_mean'] = float(data['embeddings'].mean())
            rows[-1]['embedding_std'] = float(data['embeddings'].std())
            rows[-1]['near_duplicate_cosine_fraction'] = _near_duplicate(data['embeddings'])
            for metric_key in ('pearson_only', 'siamese_only', 'shared', 'union_candidates',
                               'pearson_fraction', 'siamese_fraction',
                               'pearson_missed_fraction', 'selected_undirected'):
                key = f'candidate_{metric_key}'
                if key in data:
                    rows[-1][metric_key + '_mean'] = float(np.asarray(data[key]).mean())
        else:
            rows[-1]['low_pearson_candidate_fraction'] = float('nan')

        vals = graph.values().cpu().numpy()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(vals, bins=50)
        ax.set(title=f'{path.stem}: edge weight distribution', xlabel='weight', ylabel='count')
        fig.tight_layout(); fig.savefig(out / f'{path.stem}_weights.png', dpi=160); plt.close(fig)

    if len(loaded) > 1:
        base_path, _, base_graph = loaded[0]
        for path, _, graph in loaded[1:]:
            rows[[row['file'] for row in rows].index(str(path))].update(
                {f'vs_{base_path.stem}_{key}': value for key, value in graph_similarity(base_graph, graph).items()})
    columns = sorted({key for row in rows for key in row})
    with (out / 'graph_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    print(f'Wrote graph metrics and figures to {out.resolve()}')


def _low_pearson_fraction(pearson, edges, threshold=0.2):
    edges = edges.reshape(-1, 3).long()
    b, i, j = edges.unbind(-1)
    keep = i != j
    if not keep.any():
        return 0.0
    return float((pearson[b[keep], i[keep], j[keep]].abs() < threshold).float().mean())


def _near_duplicate(embeddings, threshold=0.98):
    z = torch.from_numpy(embeddings).float().mean(dim=2)
    z = torch.nn.functional.normalize(z, dim=-1)
    values = []
    for matrix in z:
        cosine = matrix @ matrix.T
        mask = torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)
        values.append(cosine[mask])
    values = torch.cat(values)
    return float((values > threshold).float().mean())


if __name__ == '__main__':
    main()
