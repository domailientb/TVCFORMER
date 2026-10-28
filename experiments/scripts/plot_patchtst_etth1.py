# vẽ biểu đồ đg thể hiện sự thay đổi của MSE , MAE theo độ dài dự báo
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILE = PROJECT_ROOT / "experiments" / "results" / "PatchTST_ETTh1_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "figures" / "patchtst"

PRED_LENGTHS = [96, 192, 336, 720]
MODEL_NAME = "PatchTST"


def main():
    if not RESULT_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {RESULT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULT_FILE)
    df.columns = df.columns.str.strip()

   # Chuẩn hóa tên cột và kiểm tra các cột bắt buộc.
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

    # lọc và tiền xly kq
    df["Model"] = df["Model"].astype(str).str.strip().str.lower()
    df["Pred_len"] = pd.to_numeric(df["Pred_len"], errors="coerce")

    for metric in ["MSE", "MAE"]:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    # Chỉ giữ kết quả PatchTST có đầy đủ các chỉ số cần thiết.
    df = df[df["Model"] == MODEL_NAME.lower()].dropna(
        subset=["Pred_len", "MSE", "MAE"]
    )

    if df.empty:
        raise ValueError("Không có kết quả PatchTST trong file.")

    df["Pred_len"] = df["Pred_len"].astype(int)

    # Nếu một độ dài dự báo có nhiều kết quả, lấy giá trị trung bình.
    df = df.groupby("Pred_len", as_index=False)[["MSE", "MAE"]].mean()

    # vẽ 
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

        ax.set_title(f"PatchTST - ETTh1 - {metric}", fontweight="bold")
        ax.set_xlabel("Độ dài dự báo (Pred_len)")
        ax.set_ylabel(metric)
        ax.set_xticks(PRED_LENGTHS)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()

        output_file = OUTPUT_DIR / f"patchtst_etth1_{metric.lower()}.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Đã lưu: {output_file}")


if __name__ == "__main__":
    main()
