"""Compare ETTh1 prediction ablation sensitivity with TVC graph degree."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GRAPH_DIR = RESULTS / "graphs"
FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
HORIZONS = [96, 192, 336, 720]


def main():
    ablation = pd.read_csv(RESULTS / "ETTh1_mean_replacement_ablation.csv")
    ablation = ablation[ablation.experiment != "Baseline"]
    degree = pd.read_csv(GRAPH_DIR / "variable_weighted_degree.csv")
    comparison = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, horizon in zip(axes.flat, HORIZONS):
        imp = ablation[ablation.pred_len == horizon].set_index("masked_variable")
        deg = degree[degree.horizon == horizon].set_index("variable")
        xai_order = sorted(FEATURES, key=lambda f: float(imp.loc[f, "influence_mse"]), reverse=True)
        graph_order = sorted(FEATURES, key=lambda f: float(deg.loc[f, "weighted_degree"]), reverse=True)
        rho = float(np.corrcoef([xai_order.index(f) for f in FEATURES],
                                [graph_order.index(f) for f in FEATURES])[0, 1])
        comparison.append({
            "horizon": horizon,
            "largest_delta_mse_variable": xai_order[0],
            "highest_weighted_degree_variable": graph_order[0],
            "same_top_variable": xai_order[0] == graph_order[0],
            "spearman_rank_correlation": rho,
        })
        for feature in FEATURES:
            x = float(deg.loc[feature, "weighted_degree"])
            y = float(imp.loc[feature, "influence_mse"])
            ax.scatter(x, y, s=40)
            ax.annotate(feature, (x, y), fontsize=8)
        ax.set_title("ETTh1 horizon {}".format(horizon))
        ax.set_xlabel("TVC graph weighted degree")
        ax.set_ylabel("Feature replacement Delta MSE")
        ax.grid(alpha=0.25)
    fig.savefig(GRAPH_DIR / "figure_xai_vs_graph.png", dpi=220)
    plt.close(fig)

    out = GRAPH_DIR / "xai_method_comparison.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    lines = ["# ETTh1 comparison of feature ablation and TVC graph", "",
             "Delta MSE is the change in test error after replacing a feature with its standardized train mean. Weighted degree sums the feature's off-diagonal mean block weights in the TVC graph. They measure forecast sensitivity and connectivity, respectively.", "",
             "| Horizon | Largest Delta MSE | Highest weighted degree | Same feature? | Spearman rank correlation |",
             "|---:|---|---|---|---:|"]
    for row in comparison:
        lines.append("| {} | {} | {} | {} | {:.3f} |".format(
            row["horizon"], row["largest_delta_mse_variable"], row["highest_weighted_degree_variable"],
            "Yes" if row["same_top_variable"] else "No", row["spearman_rank_correlation"]))
    lines += ["", "![Ablation sensitivity versus graph connectivity](figure_xai_vs_graph.png)", "",
              "Ablation sensitivity and graph connectivity are different descriptive views; agreement does not establish causality.", ""]
    (GRAPH_DIR / "ETTh1_XAI_graph_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved ETTh1 XAI comparison to", GRAPH_DIR)


if __name__ == "__main__":
    main()
