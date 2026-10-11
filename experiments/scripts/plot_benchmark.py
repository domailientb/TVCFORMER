
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
BENCHMARK_FILE = RESULTS_DIR / "benchmark_results.csv"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures" / "benchmark"

METRICS = ["MSE", "MAE"]


def main():
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {BENCHMARK_FILE}"
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BENCHMARK_FILE)

    required_columns = {
        "Dataset", "Pred_len", "Model", "MSE", "MAE"
    }
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"File CSV thiếu các cột: {missing}")

    df["Pred_len"] = pd.to_numeric(df["Pred_len"], errors="raise")

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="raise")

    methods = sorted(df["Model"].dropna().unique())

    print("Datasets:", sorted(df["Dataset"].unique()))
    print("Methods:", methods)
    print("Number of methods:", len(methods))
    print("Rows:", len(df))

    for dataset in sorted(df["Dataset"].unique()):
        dataset_df = df[df["Dataset"] == dataset].copy()

        for metric in METRICS:
            pivot = dataset_df.pivot(
                index="Pred_len",
                columns="Model",
                values=metric,
            )

            # Chỉ so sánh những mốc có kết quả của tất cả mô hình.
            pivot = pivot.reindex(columns=methods)
            complete_pivot = pivot.dropna(axis=0, how="any")

            if complete_pivot.empty:
                print(
                    f"Skipped: {dataset} - {metric}: "
                    "không có mốc dự báo đủ dữ liệu của tất cả mô hình."
                )
                continue

            # Sắp xếp chú giải theo sai số trung bình tăng dần.
            model_order = (
                complete_pivot.mean(axis=0)
                .sort_values()
                .index
            )
            complete_pivot = complete_pivot[model_order]

            ax = complete_pivot.plot(
                kind="bar",
                figsize=(15, 7),
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

            output_file = FIGURES_DIR / (
                f"{dataset}_{metric.lower()}.png"
            )
            plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            print(f"Saved: {output_file}")

            # In mô hình có sai số thấp nhất tại từng mốc.
            print(f"\nBest model by {metric} - {dataset}:")
            for pred_len, row in complete_pivot.iterrows():
                best_model = row.idxmin()
                best_value = row.min()
                print(
                    f"  Pred_len={int(pred_len)}: "
                    f"{best_model} ({metric}={best_value:.6f})"
                )

            print()

    print("Hoàn thành vẽ biểu đồ.")


if __name__ == "__main__":
    main()
