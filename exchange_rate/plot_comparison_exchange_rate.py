# Vẽ biểu đồ so sánh MSE, MAE giữa bốn mô hình và phân tích ảnh hưởng của Pred_len đến TVCFormer

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "results" / "results_exchange_rate.csv"
OUTPUT_DIR = PROJECT_DIR / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_LENGTHS = [96, 192, 336, 720]
MODELS = ["TVCFormer", "TimesNet", "DLinear", "PatchTST"]

def load_results():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file CSV: {INPUT_FILE}"
        )
    df = pd.read_csv(INPUT_FILE, skipinitialspace=True)

    df.columns = df.columns.str.strip()
    df["Model"] = df["Model"].astype(str).str.strip()
    df["Dataset"] = df["Dataset"].astype(str).str.strip()

    # Chuyển các cột chỉ số về dạng số và loại bỏ dòng không hợp lệ
    for column in ["Seq_len", "Pred_len", "MSE", "MAE"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Pred_len", "MSE", "MAE"])

    # Chỉ giữ kết quả Exchange Rate của các mô hình và Pred_len cần so sánh
    df = df[
        (df["Dataset"] == "Exchange_Rate")
        & (df["Pred_len"].isin(PRED_LENGTHS))
        & (df["Model"].isin(MODELS))
    ]

    if df.empty:
        raise ValueError(
            "Không có dữ liệu Exchange_Rate hợp lệ trong file CSV."
        )
    return df

def plot_model_comparison(df):
    # Tính MSE và MAE trung bình của mỗi mô hình trên các Pred_len có dữ liệu
    mean_results = (
        df.groupby("Model")[["MSE", "MAE"]]
        .mean()
        .reindex(MODELS)
    )
    x = np.arange(len(MODELS))
    width = 0.6
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars_mse = axes[0].bar(
        x,
        mean_results["MSE"],
        width
    )
    axes[0].set_title("So sánh MSE trung bình")
    axes[0].set_xlabel("Phương pháp")
    axes[0].set_ylabel("MSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(MODELS, rotation=15)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].bar_label(bars_mse, fmt="%.3f", padding=3)

    bars_mae = axes[1].bar(
        x,
        mean_results["MAE"],
        width
    )
    axes[1].set_title("So sánh MAE trung bình")
    axes[1].set_xlabel("Phương pháp")
    axes[1].set_ylabel("MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(MODELS, rotation=15)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].bar_label(bars_mae, fmt="%.3f", padding=3)

    fig.suptitle(
        "So sánh 4 phương pháp trên Exchange_Rate",
        fontsize=14
    )
    fig.tight_layout()

    output_file = OUTPUT_DIR / "comparison_4_methods.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Đã lưu hình 1: {output_file}")

def plot_prediction_lengths(df):
    # Lấy MSE và MAE trung bình của TVCFormer theo từng Pred_len
    tvcformer = (
        df[df["Model"] == "TVCFormer"]
        .groupby("Pred_len")[["MSE", "MAE"]]
        .mean()
        .reindex(PRED_LENGTHS)
    )

    if tvcformer.isna().any().any():
        raise ValueError(
            "TVCFormer chưa có đủ kết quả cho cả 4 Pred_len: "
            "96, 192, 336, 720."
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        PRED_LENGTHS,
        tvcformer["MSE"],
        marker="o",
        linewidth=2,
        label="TVCFormer"
    )
    axes[0].set_title("TVCFormer - MSE theo độ dài dự báo")
    axes[0].set_xlabel("Độ dài dự báo (Pred_len)")
    axes[0].set_ylabel("MSE")
    axes[0].set_xticks(PRED_LENGTHS)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    # Hiển thị giá trị MSE tại từng độ dài dự báo
    for pred_len, value in zip(PRED_LENGTHS, tvcformer["MSE"]):
        axes[0].annotate(
            f"{value:.3f}",
            (pred_len, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center"
        )

    axes[1].plot(
        PRED_LENGTHS,
        tvcformer["MAE"],
        marker="o",
        linewidth=2,
        label="TVCFormer"
    )
    axes[1].set_title("TVCFormer - MAE theo độ dài dự báo")
    axes[1].set_xlabel("Độ dài dự báo (Pred_len)")
    axes[1].set_ylabel("MAE")
    axes[1].set_xticks(PRED_LENGTHS)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    # Hiển thị giá trị MAE tại từng độ dài dự báo
    for pred_len, value in zip(PRED_LENGTHS, tvcformer["MAE"]):
        axes[1].annotate(
            f"{value:.3f}",
            (pred_len, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center"
        )

    fig.suptitle(
        "Ảnh hưởng của độ dài dự báo đối với TVCFormer - Exchange_Rate",
        fontsize=14
    )
    fig.tight_layout()

    output_file = OUTPUT_DIR / "tvcformer_prediction_lengths.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Đã lưu hình 2: {output_file}")

if __name__ == "__main__":
    results = load_results()

    plot_model_comparison(results)
    plot_prediction_lengths(results)

    print("\nĐã vẽ xong cả 2 hình.")