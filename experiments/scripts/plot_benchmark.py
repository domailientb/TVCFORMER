
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
BENCHMARK_FILE = RESULTS_DIR / "benchmark_results.csv"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures" / "benchmark"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BENCHMARK_FILE)

    required_columns = {
        "Dataset", "Pred_len", "Model", "MSE", "MAE"
    }
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("Datasets:", sorted(df["Dataset"].unique()))
    print("Methods:", sorted(df["Model"].unique()))
    print("Rows:", len(df))

    metrics = ["MSE", "MAE"]

    for dataset in sorted(df["Dataset"].unique()):
        dataset_df = df[df["Dataset"] == dataset]

        for metric in metrics:
            pivot = dataset_df.pivot(
                index="Pred_len",
                columns="Model",
                values=metric,
            )

            ax = pivot.plot(
                kind="bar",
                figsize=(13, 6),
                width=0.85,
            )

            ax.set_title(f"{dataset} - {metric} comparison")
            ax.set_xlabel("Prediction length")
            ax.set_ylabel(metric)
            ax.legend(
                title="Method",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
            )

            ax.tick_params(axis="x", rotation=0)
            ax.grid(axis="y", linestyle="--", alpha=0.35)

            plt.tight_layout()

            output_file = FIGURES_DIR / f"{dataset}_{metric}.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            print("Saved:", output_file)


if __name__ == "__main__":
    main()
