# Vẽ biểu đồ MSE và MAE của PatchTST trên bộ dữ liệu Exchange Rate.
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RESULT_FILE = PROJECT_DIR / "results" / "results_exchange_rate.csv"
OUTPUT_DIR = PROJECT_DIR / "figures" / "patchtst"

PRED_LENGTHS = [96, 192, 336, 720]
MODEL_NAME = "PatchTST"


def main():
    if not RESULT_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {RESULT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULT_FILE)
    df.columns = df.columns.str.strip()

    # Kiểm tra các cột cần thiết trong file kết quả
    column_map = {col.lower(): col for col in df.columns}
    required = ["model", "pred_len", "mse", "mae"]

    if any(col not in column_map for col in required):
        raise ValueError(f"File thiếu cột cần thiết: {list(df.columns)}")
    df = df.rename(columns={
        column_map["model"]: "Model",
        column_map["pred_len"]: "Pred_len",
        column_map["mse"]: "MSE",
        column_map["mae"]: "MAE",
    })

    # Chuẩn hóa dữ liệu và chuyển các chỉ số về dạng số
    df["Model"] = df["Model"].astype(str).str.strip().str.lower()
    df["Pred_len"] = pd.to_numeric(df["Pred_len"], errors="coerce")

    for metric in ["MSE", "MAE"]:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = df[df["Model"] == MODEL_NAME.lower()].dropna(
        subset=["Pred_len", "MSE", "MAE"]
    )
    if df.empty:
        raise ValueError("Không có kết quả PatchTST trong file.")

    df["Pred_len"] = df["Pred_len"].astype(int)

    # Gộp các lần chạy cùng độ dài dự báo bằng giá trị trung bình
    df = df.groupby("Pred_len", as_index=False)[["MSE", "MAE"]].mean()
    for metric in ["MSE", "MAE"]:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            df["Pred_len"],
            df[metric],
            marker="o",
            linewidth=2,
            markersize=7,
            label="PatchTST",
        )
        ax.set_title(
            f"PatchTST - Exchange Rate - {metric}",
            fontweight="bold",
        )
        ax.set_xlabel("Độ dài dự báo (Pred_len)")
        ax.set_ylabel(metric)
        ax.set_xticks(PRED_LENGTHS)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()

        output_file = OUTPUT_DIR / f"patchtst_exchange_rate_{metric.lower()}.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Đã lưu: {output_file}")

if __name__ == "__main__":
    main()