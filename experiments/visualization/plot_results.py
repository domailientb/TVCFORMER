from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Xác định thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "reproduction_results.csv"
)

FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"


def plot_metric(data, metric, title, ylabel, output_file):
    """Vẽ biểu đồ so sánh một chỉ số giữa các bộ dữ liệu."""

    plt.figure(figsize=(9, 6))

    for dataset in data["Dataset"].unique():
        subset = data[data["Dataset"] == dataset].sort_values("Pred_len")

        plt.plot(
            subset["Pred_len"],
            subset[metric],
            marker="o",
            linewidth=2,
            markersize=6,
            label=dataset,
        )

    plt.title(title)
    plt.xlabel("Độ dài dự báo (Pred_len)")
    plt.ylabel(ylabel)
    plt.xticks([96, 192, 336, 720])
    plt.legend(title="Bộ dữ liệu")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path = FIGURES_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Đã lưu biểu đồ: {output_path}")


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file kết quả: {CSV_FILE}"
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(CSV_FILE, skipinitialspace=True)

    # Loại bỏ khoảng trắng thừa trong tên cột và dữ liệu dạng chuỗi
    data.columns = data.columns.str.strip()

    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].str.strip()

    required_columns = {
        "Dataset",
        "Pred_len",
        "MSE",
        "MAE",
        "RSE",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"CSV thiếu các cột: {sorted(missing_columns)}"
        )

    print(f"Đã đọc {len(data)} dòng kết quả.")
    print("Các bộ dữ liệu:", ", ".join(data["Dataset"].unique()))

    plot_metric(
        data,
        metric="MSE",
        title="So sánh MSE của TVCFormer",
        ylabel="MSE",
        output_file="comparison_mse.png",
    )

    plot_metric(
        data,
        metric="MAE",
        title="So sánh MAE của TVCFormer",
        ylabel="MAE",
        output_file="comparison_mae.png",
    )

    plot_metric(
        data,
        metric="RSE",
        title="So sánh RSE của TVCFormer",
        ylabel="RSE",
        output_file="comparison_rse.png",
    )

    print("\nHoàn thành vẽ 3 biểu đồ.")


if __name__ == "__main__":
    main()