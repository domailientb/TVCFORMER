# kq bài báo
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PAPER_FILE = RESULTS_DIR / "paper_comparison.csv"
REPRODUCED_FILE = RESULTS_DIR / "benchmark_results.csv"
OUTPUT_FILE = RESULTS_DIR / "paper_vs_reproduced.csv"

METRICS = ["MSE", "MAE"]


def main():
    if not PAPER_FILE.exists():
        print(f"ERROR: Paper file not found: {PAPER_FILE}")
        return

    if not REPRODUCED_FILE.exists():
        print(f"ERROR: Reproduced file not found: {REPRODUCED_FILE}")
        return

    paper = pd.read_csv(PAPER_FILE)
    reproduced = pd.read_csv(REPRODUCED_FILE)

    # Loại bỏ khoảng trắng thừa trong tên cột
    paper.columns = paper.columns.str.strip()
    reproduced.columns = reproduced.columns.str.strip()

    # Đảm bảo hai file có đủ cột cần thiết để so sánh
    required = {"Dataset", "Pred_len", "Model", *METRICS}

    if not required.issubset(paper.columns):
        print("ERROR: Paper CSV has missing columns.")
        print("Required:", sorted(required))
        print("Found:", paper.columns.tolist())
        return

    if not required.issubset(reproduced.columns):
        print("ERROR: Benchmark CSV has missing columns.")
        print("Required:", sorted(required))
        print("Found:", reproduced.columns.tolist())
        return

    keys = ["Dataset", "Pred_len", "Model"]

    for df in [paper, reproduced]:
        df["Dataset"] = df["Dataset"].astype(str).str.strip()
        df["Model"] = df["Model"].astype(str).str.strip()

        # Chuyển dữ liệu về dạng số; giá trị không hợp lệ thành NaN
        df["Pred_len"] = pd.to_numeric(
            df["Pred_len"], errors="coerce"
        )

        for metric in METRICS:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

    # Loại các dòng thiếu thông tin định danh để phục vụ ghép dữ liệu
    paper = paper.dropna(subset=keys).copy()
    reproduced = reproduced.dropna(subset=keys).copy()

    paper["Pred_len"] = paper["Pred_len"].astype(int)
    reproduced["Pred_len"] = reproduced["Pred_len"].astype(int)
    
     # Đổi tên cột để phân biệt kết quả trong bài báo và kết quả tái thực nghiệm
    paper = paper.rename(
        columns={
            "MSE": "Paper_MSE",
            "MAE": "Paper_MAE",
        }
    )

    reproduced = reproduced.rename(
        columns={
            "MSE": "Reproduced_MSE",
            "MAE": "Reproduced_MAE",
        }
    )

    # Ghép theo bộ dữ liệu, độ dài dự báo và mô hình
    # outer giữ cả những trường hợp chỉ xuất hiện ở một trong hai file
    comparison = paper[
        keys + ["Paper_MSE", "Paper_MAE"]
    ].merge(
        reproduced[
            keys + ["Reproduced_MSE", "Reproduced_MAE"]
        ],
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    
    # Hiệu số = kết quả tái thực nghiệm - kết quả trong bài báo
    comparison["MSE_Difference"] = (
        comparison["Reproduced_MSE"] - comparison["Paper_MSE"]
    )

    comparison["MAE_Difference"] = (
        comparison["Reproduced_MAE"] - comparison["Paper_MAE"]
    )

    # Xác định nguồn có dữ liệu tương ứng
    comparison["Match_Status"] = comparison["_merge"].map(
        {
            "both": "Both available",
            "left_only": "Paper only",
            "right_only": "Reproduced only",
        }
    )

    comparison = comparison.drop(columns=["_merge"])

    comparison = comparison.sort_values(
        ["Dataset", "Pred_len", "Model"]
    ).reset_index(drop=True)

    # Xuất bảng so sánh; ghi đè file cũ nếu đã tồn tại
    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 60)
    print("PAPER VS REPRODUCED RESULTS")
    print("=" * 60)
    print(f"Paper rows: {len(paper)}")
    print(f"Reproduced rows: {len(reproduced)}")
    print(f"Comparison rows: {len(comparison)}")
    print()
    print("Match status:")
    print(comparison["Match_Status"].value_counts().to_string())
    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
