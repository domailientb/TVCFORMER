 
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RESULTS_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "benchmark_results.csv"
)


TVCFORMER_RESULTS = [
    ["ETTh1", 96,  "TVCFormer", 0.36449319,  0.39083716, 336],
    ["ETTh1", 192, "TVCFormer", 0.39980584,  0.41383228, 336],
    ["ETTh1", 336, "TVCFormer", 0.42267010,  0.42998970, 336],
    ["ETTh1", 720, "TVCFormer", 0.43064615,  0.45328006, 336],

    ["ETTh2", 96,  "TVCFormer", 0.27344292,  0.33489630, 336],
    ["ETTh2", 192, "TVCFormer", 0.33650035, 0.37778836, 336],
    ["ETTh2", 336, "TVCFormer", 0.32352257, 0.38003573, 336],
    ["ETTh2", 720, "TVCFormer", 0.37429574, 0.41818500, 336],

    ["ETTm1", 96,  "TVCFormer", 0.28850044, 0.34578973, 336],
    ["ETTm1", 192, "TVCFormer", 0.33161572, 0.36587685, 336],
    ["ETTm1", 336, "TVCFormer", 0.36226815, 0.38391072, 336],
    ["ETTm1", 720, "TVCFormer", 0.41697350, 0.41518873, 336],

    ["ETTm2", 96,  "TVCFormer", 0.16428503, 0.25412986, 336],
    ["ETTm2", 192, "TVCFormer", 0.22106625, 0.29301935, 336],
    ["ETTm2", 336, "TVCFormer", 0.27556863, 0.32850865, 336],
    ["ETTm2", 720, "TVCFormer", 0.36042770, 0.380589395, 336],

    ["Weather", 96,  "TVCFormer", 0.15060541, 0.19990897, 336],
    ["Weather", 192, "TVCFormer", 0.19451171, 0.24292062, 336],
    ["Weather", 336, "TVCFormer", 0.24629523, 0.28291723, 336],
    ["Weather", 720, "TVCFormer", 0.31443340, 0.33348766, 336],
]


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file kết quả: {RESULTS_FILE}"
        )

    df = pd.read_csv(RESULTS_FILE)

    required_columns = [
        "Dataset", "Pred_len", "Model", "MSE", "MAE", "Seq_len"
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"File CSV thiếu các cột: {sorted(missing)}")

    tvcformer_df = pd.DataFrame(
        TVCFORMER_RESULTS,
        columns=required_columns
    )

    # Xóa kết quả TVCFormer cũ nếu đã tồn tại,
    # tránh thêm trùng khi chạy script nhiều lần.
    df = df[df["Model"] != "TVCFormer"]

    # Gộp kết quả TVCFormer vào bảng 9 phương pháp hiện có.
    df = pd.concat([df, tvcformer_df], ignore_index=True)

    df["Pred_len"] = df["Pred_len"].astype(int)
    df["Seq_len"] = df["Seq_len"].astype(int)

    df = df.sort_values(
        ["Dataset", "Pred_len", "Model"]
    ).reset_index(drop=True)

    df.to_csv(RESULTS_FILE, index=False, encoding="utf-8")

    print("Đã gộp kết quả TVCFormer thành công.")
    print(f"File: {RESULTS_FILE}")
    print(f"Tổng số dòng: {len(df)}")
    print(f"Số phương pháp: {df['Model'].nunique()}")
    print("\nSố dòng theo phương pháp:")
    print(df["Model"].value_counts().sort_index())


if __name__ == "__main__":
    main()
