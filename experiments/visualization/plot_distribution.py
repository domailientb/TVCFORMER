
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = PROJECT_ROOT / "experiments" / "datasets" / "ETT-small" / "ETTh1.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "figures" / "ETTh1"


def main():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_FILE)
    numeric_df = df.select_dtypes(include="number")
    columns = numeric_df.columns[:8]

    if len(columns) == 0:
        raise ValueError("Dataset contains no numeric columns.")

    fig, axes = plt.subplots(
        len(columns), 1, figsize=(10, 2.5 * len(columns))
    )
    if len(columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, columns):
        ax.hist(numeric_df[column].dropna(), bins=50, edgecolor="black")
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.2)

    fig.suptitle("ETTh1 - Feature Distributions")
    fig.tight_layout()

    output_file = OUTPUT_DIR / "distribution.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
