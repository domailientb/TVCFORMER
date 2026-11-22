"""Extract and visualize the trained ETTh1 TVC graphs on held-out test data.

This is inference-only: it loads the validation-selected checkpoints produced
by run_etth1_ablation.py and never calls train(). The test inputs trigger the
dynamic graph generator; edge indices and values are averaged per test window.
"""
import csv
import gc
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "official"), str(ROOT / "FinalProject/TR_TVC/models"), str(ROOT / "FinalProject/TR_TVC")]

from run_etth1_ablation import FEATURES, build_args, seed_everything  # noqa: E402
from exp_tr_tvc import Exp_TR_TVC  # noqa: E402

HORIZONS = (96, 192, 336, 720)
NODE_COUNT = len(FEATURES) * 7
OUT = HERE / "results" / "graphs"


def load_checkpoint(exp, horizon, seed):
    setting = "ETTh1_TVCFormer_ablation_pl{}_seed{}".format(horizon, seed)
    path = HERE / "checkpoints" / setting / "checkpoint.pth"
    if not path.exists():
        raise FileNotFoundError("Missing trained checkpoint: {}".format(path))
    state = torch.load(path, map_location=exp.device)
    exp.model.load_state_dict(state)
    return setting


def extract_one(horizon, seed=2021):
    class Cli:
        random_seed = seed
        batch_size = 32
        epochs = None
        use_gpu = 1

    args = build_args(horizon, Cli())
    seed_everything(seed)
    exp = Exp_TR_TVC(args)
    setting = load_checkpoint(exp, horizon, seed)
    exp.model.eval()
    generator = exp.model.model.graph_generator
    captured = []

    def capture_graph(module, inputs, output):
        # TVC graph generator returns global COO node ids and their learned
        # nonnegative edge weights. Batch node ids are shifted by 49 per case.
        edge_index, edge_weight = output[0], output[1]
        captured.append((edge_index.detach().cpu(), edge_weight.detach().cpu()))

    hook = generator.register_forward_hook(capture_graph)
    _, test_loader = exp._get_data(flag="test")
    dense_sum = torch.zeros((NODE_COUNT, NODE_COUNT), dtype=torch.float64)
    samples_seen = 0
    with torch.inference_mode():
        for batch_x, _, _, _ in test_loader:
            x = batch_x.float().to(exp.device)
            exp.model(x)
            batch_size = x.shape[0]
            if not captured:
                raise RuntimeError("Graph generator hook did not capture an adjacency")
            edge_index, edge_weight = captured.pop()
            local_src = edge_index[:, 0] % NODE_COUNT
            local_dst = edge_index[:, 1] % NODE_COUNT
            local_batch = edge_index[:, 0] // NODE_COUNT
            if edge_index.numel() and (edge_index[:, 1] // NODE_COUNT != local_batch).any():
                raise RuntimeError("Cross-sample graph edge found in batched TVC graph")
            flat = (local_src * NODE_COUNT + local_dst).long()
            per_sample = torch.zeros((batch_size, NODE_COUNT * NODE_COUNT), dtype=torch.float64)
            per_sample.index_put_((local_batch.long(), flat), edge_weight.double(), accumulate=True)
            dense_sum += per_sample.sum(dim=0).reshape(NODE_COUNT, NODE_COUNT)
            samples_seen += batch_size
    hook.remove()
    del test_loader, captured, generator, hook
    gc.collect()

    if samples_seen == 0:
        raise RuntimeError("ETTh1 test loader returned no windows")
    graph49 = (dense_sum / samples_seen).numpy()
    # The official graph has undirected channel support, but averaging floating
    # point sparse outputs can leave tiny directional differences. Symmetrize
    # the visualization and record symmetric edge weight.
    graph49 = (graph49 + graph49.T) / 2.0
    graph7 = np.zeros((len(FEATURES), len(FEATURES)), dtype=np.float64)
    for source in range(len(FEATURES)):
        s = slice(source * 7, (source + 1) * 7)
        for target in range(len(FEATURES)):
            t = slice(target * 7, (target + 1) * 7)
            graph7[source, target] = graph49[s, t].mean()
    np.fill_diagonal(graph7, 0.0)
    return graph49, graph7, samples_seen, setting


def draw_heatmap(matrix, path, title, labels, xlabel, ylabel, cmap="magma"):
    fig, ax = plt.subplots(figsize=(12, 10) if matrix.shape[0] > 10 else (9, 7.5), constrained_layout=True)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90 if len(labels) > 10 else 45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label="Mean TVC edge weight")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_matrix_csv(matrix, path, names):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["node"] + names)
        for label, row in zip(names, matrix):
            writer.writerow([label] + ["{:.10g}".format(value) for value in row])


def write_edge_tables(graph_by_horizon):
    edge_rows, degree_rows, stability_rows = [], [], []
    supports = {}
    for horizon, graph in graph_by_horizon.items():
        upper = np.triu(graph, 1)
        pairs = [(FEATURES[i], FEATURES[j], float(upper[i, j]))
                 for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))]
        pairs.sort(key=lambda row: row[2], reverse=True)
        for rank, (left, right, weight) in enumerate(pairs, 1):
            edge_rows.append({"horizon": horizon, "rank": rank, "variable_a": left,
                              "variable_b": right, "mean_edge_weight": weight})
        strength = graph.sum(axis=1)
        for name, value in zip(FEATURES, strength):
            degree_rows.append({"horizon": horizon, "variable": name,
                                "weighted_degree": float(value),
                                "mean_strength_per_other_variable": float(value / (len(FEATURES) - 1))})
        threshold = np.quantile([p[2] for p in pairs], 0.70)
        supports[horizon] = {(a, b) for a, b, w in pairs if w >= threshold}

    for h1, h2 in zip(HORIZONS[:-1], HORIZONS[1:]):
        a, b = supports[h1], supports[h2]
        stability_rows.append({"horizon_a": h1, "horizon_b": h2,
                               "top30pct_support_jaccard": len(a & b) / len(a | b) if a | b else 1.0,
                               "matrix_pearson": float(np.corrcoef(graph_by_horizon[h1].ravel(),
                                                                  graph_by_horizon[h2].ravel())[0, 1])})

    def write(rows, filename):
        path = OUT / filename
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(edge_rows, "top_variable_edges_by_horizon.csv")
    write(degree_rows, "variable_weighted_degree.csv")
    write(stability_rows, "graph_horizon_stability.csv")


def draw_horizon_comparison(graphs):
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(18, 5.2), constrained_layout=True, sharey=True)
    vmax = max(float(np.max(graph)) for graph in graphs.values())
    for ax, horizon in zip(axes, HORIZONS):
        im = ax.imshow(graphs[horizon], cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title("Horizon {}".format(horizon))
        ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=70, ha="right")
        ax.set_yticks(range(len(FEATURES)), FEATURES)
        ax.set_xlabel("Destination variable")
    axes[0].set_ylabel("Source variable")
    fig.colorbar(im, ax=axes, label="Mean TVC edge weight", shrink=0.82)
    fig.savefig(OUT / "figure3_horizon_comparison.png", dpi=220)
    plt.close(fig)


def write_report(graphs, samples):
    stability = []
    for h1, h2 in zip(HORIZONS[:-1], HORIZONS[1:]):
        stability.append((h1, h2, float(np.corrcoef(graphs[h1].ravel(), graphs[h2].ravel())[0, 1])))
    strongest = {}
    weakest = {}
    degrees = {}
    for h, matrix in graphs.items():
        pairs = [(matrix[i, j], FEATURES[i], FEATURES[j])
                 for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))]
        strongest[h] = sorted(pairs, reverse=True)[:5]
        weakest[h] = sorted(pairs)[:3]
        degrees[h] = sorted(zip(matrix.sum(axis=1), FEATURES), reverse=True)
    lines = [
        "# ETTh1 TVC Graph extraction and structure analysis", "",
        "## Protocol", "",
        "Loaded the validation-selected TVCFormer checkpoints trained for the ETTh1 mean-replacement experiment. No training or parameter updates were performed. Graphs were extracted by forwarding all complete batches of ETTh1 test windows through each checkpoint. The test loader uses the official `drop_last=True` protocol; sample counts are listed below.", "",
        "For every graph edge, source and destination IDs were reduced modulo 49 to remove the batch offset; each sample adjacency was accumulated and divided by the number of test windows. Node IDs are ordered variable-major: HUFL-T1..T7, HULL-T1..T7, MUFL-T1..T7, MULL-T1..T7, LUFL-T1..T7, LULL-T1..T7, OT-T1..T7. The 7×7 matrix is the mean of the corresponding 7×7 temporal-node block, including zeros for absent edges. Diagonals are excluded from the inter-variable matrix.", "",
        "## Figures", "",
        "- Figure 1: [49×49 time-node graph heatmap](figure1_graph49_heatmaps.png)",
        "- Figure 2: [7×7 variable connection heatmaps](figure2_variable_heatmaps.png)",
        "- Figure 3: [Horizon comparison](figure3_horizon_comparison.png)", "",
        "## Test graph batches", "",
        "| Horizon | Windows aggregated | Checkpoint |", "|---:|---:|---|",
    ]
    for h in HORIZONS:
        lines.append("| {} | {} | `ETTh1_TVCFormer_ablation_pl{}_seed2021` |".format(h, samples[h], h))
    lines += ["", "## Strong and weak inter-variable edges", "",
              "The following are the largest and smallest mean block weights per horizon. They describe learned connectivity, not feature importance.", "",
              "| Horizon | Strongest pair | Weight | Weakest pair | Weight |", "|---:|---|---:|---|---:|"]
    for h in HORIZONS:
        a, b = strongest[h][0], weakest[h][0]
        lines.append("| {} | {}–{} | {:.6g} | {}–{} | {:.6g} |".format(h, a[1], a[2], a[0], b[1], b[2], b[0]))
    lines += ["", "## Stability across horizons", "",
              "Pearson correlation compares all 49 inter-variable block weights flattened from each 7×7 matrix. Values near one mean similar relative edge-weight patterns. It is descriptive, and because edge magnitudes are averaged across test windows, it does not imply identical per-window graphs.", "",
              "| Horizons | 7×7 weight matrix Pearson r | Top 30% support Jaccard |", "|---|---:|---:|"]
    for h1, h2, corr in stability:
        p1 = [(graphs[h1][i, j], i, j) for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))]
        p2 = [(graphs[h2][i, j], i, j) for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))]
        s1 = {(i, j) for _, i, j in sorted(p1, reverse=True)[:7]}
        s2 = {(i, j) for _, i, j in sorted(p2, reverse=True)[:7]}
        jaccard = len(s1 & s2) / len(s1 | s2) if s1 | s2 else 1.0
        lines.append("| {} vs {} | {:.4f} | {:.4f} |".format(h1, h2, corr, jaccard))
    lines += ["", "## Table 1. Five strongest variable pairs per horizon", "",
              "| Horizon | Rank | Variable pair | Mean block weight |", "|---:|---:|---|---:|"]
    for h in HORIZONS:
        for rank, (weight, left, right) in enumerate(strongest[h][:5], 1):
            lines.append("| {} | {} | {}–{} | {:.6f} |".format(h, rank, left, right, weight))
    lines += ["", "## Variable weighted degree", "",
              "Weighted degree is the row sum of the symmetric 7×7 mean-weight matrix; it summarizes average connection strength to all other variables.", "",
              "| Horizon | Highest weighted degree | Weight | Lowest weighted degree | Weight |", "|---:|---|---:|---|---:|"]
    for h in HORIZONS:
        top, low = degrees[h][0], degrees[h][-1]
        lines.append("| {} | {} | {:.6f} | {} | {:.6f} |".format(h, top[1], top[0], low[1], low[0]))
    lines += ["", "## Interpretation", "",
              "The graph is dynamically generated for each test window from TVC's Pearson-selected channel support, then transformed into time-node edge weights by the graph generator. The 49×49 matrix exposes where edges concentrate across temporal segments; the 7×7 block mean summarizes cross-variable connectivity. Weighted degree is the sum of a variable's mean block weights to the other six variables. Consult `variable_weighted_degree.csv` and `top_variable_edges_by_horizon.csv` for full rankings.", "",
              "These weights indicate connectivity within TVC Graph. They do not measure causal relations or a variable's contribution to forecast accuracy; use the separate mean-replacement ablation for predictive sensitivity.", ""]
    (OUT / "ETTh1_TVC_graph_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seed = 2021
    node_labels = ["{}-T{}".format(variable, time_id + 1) for variable in FEATURES for time_id in range(7)]
    graphs49, graphs7, samples = {}, {}, {}
    print("Inference-only ETTh1 TVC graph extraction", "device", "cuda" if torch.cuda.is_available() else "cpu")
    for horizon in HORIZONS:
        graph49, graph7, count, checkpoint = extract_one(horizon, seed)
        graphs49[horizon], graphs7[horizon], samples[horizon] = graph49, graph7, count
        np.savez_compressed(OUT / "ETTh1_graphs_pl{}.npz".format(horizon), graph49=graph49,
                            graph7=graph7, samples=count, variables=np.array(FEATURES), nodes=np.array(node_labels))
        write_matrix_csv(graph49, OUT / "ETTh1_graph49_pl{}.csv".format(horizon), node_labels)
        write_matrix_csv(graph7, OUT / "ETTh1_graph7_pl{}.csv".format(horizon), FEATURES)
        draw_heatmap(graph49, OUT / "ETTh1_graph49_pl{}.png".format(horizon),
                     "ETTh1 TVC Graph: 49 temporal nodes, horizon {}".format(horizon), node_labels,
                     "Destination node", "Source node")
        draw_heatmap(graph7, OUT / "ETTh1_graph7_pl{}.png".format(horizon),
                     "ETTh1 TVC Graph: variable connectivity, horizon {}".format(horizon), FEATURES,
                     "Destination variable", "Source variable", cmap="viridis")
        print("H{}: {} test windows | checkpoint {} | max 49x49 weight {:.6g}".format(
            horizon, count, checkpoint, float(graph49.max())))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    vmax = max(float(g.max()) for g in graphs49.values())
    for ax, horizon in zip(axes.flat, HORIZONS):
        im = ax.imshow(graphs49[horizon], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title("Horizon {}".format(horizon))
        ax.set_xticks(range(0, NODE_COUNT, 7), FEATURES, rotation=45, ha="right")
        ax.set_yticks(range(0, NODE_COUNT, 7), FEATURES)
        ax.set_xlabel("Destination (ticks mark variable starts)")
        ax.set_ylabel("Source (ticks mark variable starts)")
    fig.colorbar(im, ax=axes, label="Mean TVC edge weight", shrink=0.82)
    fig.savefig(OUT / "figure1_graph49_heatmaps.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    vmax = max(float(g.max()) for g in graphs7.values())
    for ax, horizon in zip(axes.flat, HORIZONS):
        im = ax.imshow(graphs7[horizon], cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title("Horizon {}".format(horizon))
        ax.set_xticks(range(7), FEATURES, rotation=45, ha="right")
        ax.set_yticks(range(7), FEATURES)
        ax.set_xlabel("Destination variable")
        ax.set_ylabel("Source variable")
    fig.colorbar(im, ax=axes, label="Mean TVC edge weight", shrink=0.82)
    fig.savefig(OUT / "figure2_variable_heatmaps.png", dpi=220)
    plt.close(fig)
    draw_horizon_comparison(graphs7)
    write_edge_tables(graphs7)
    write_report(graphs7, samples)
    print("\nArtifacts:", OUT)


if __name__ == "__main__":
    main()
