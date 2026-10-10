
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = (
    PROJECT_ROOT / "experiments" / "datasets" / "ETT-small" / "ETTh1.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "figures" / "ETTh1"


def main():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_FILE)
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        raise ValueError("Dataset contains no numeric columns.")

    correlation = numeric_df.corr(method="pearson")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
    )
    plt.title("ETTh1 - Pearson Correlation Matrix")
    plt.tight_layout()

    output_file = OUTPUT_DIR / "correlation_matrix.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
