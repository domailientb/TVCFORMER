# Tổng hợp kết quả và xếp hạng 
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "results" / "results_exchange_rate.csv"
OUTPUT_FILE = PROJECT_DIR / "results" / "comparison_exchange_rate.csv"

MODELS = ["TVCFormer", "TimesNet", "DLinear", "PatchTST"]
METRICS = ["MSE", "MAE"]

def main():
    print("Đường dẫn file đầu vào:", INPUT_FILE.resolve())

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    required_columns = ["Model", "Dataset", "Seq_len", "Pred_len", "MSE", "MAE"]
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"Thiếu cột: {missing}. Các cột hiện có: {df.columns.tolist()}"
        )

    df["Model"] = df["Model"].astype(str).str.strip()

    model_mapping = {
        "tvcformer": "TVCFormer",
        "timesnet": "TimesNet",
        "dlinear": "DLinear",
        "patchtst": "PatchTST",
    }

    df["Model"] = df["Model"].str.lower().map(model_mapping)

    for column in ["Seq_len", "Pred_len", "MSE", "MAE"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Model", "Pred_len", "MSE", "MAE"]).copy()
    df["Pred_len"] = df["Pred_len"].astype(int)

    print("\nSố dòng đọc được:", len(df))
    print("\nSố dòng theo mô hình:")
    print(df["Model"].value_counts().reindex(MODELS, fill_value=0).to_string())

    # Đảm bảo dữ liệu đầu vào có đủ cả bốn mô hình
    missing_models = [
        model for model in MODELS
        if model not in df["Model"].unique()
    ]

    if missing_models:
        print("\nCác dòng dữ liệu thực tế trong file:")
        print(pd.read_csv(INPUT_FILE, skipinitialspace=True).to_string(index=False))
        raise ValueError(
            f"File đầu vào đang thiếu dữ liệu của: {missing_models}. "
            "Hãy kiểm tra và lưu đúng file CSV."
        )

    comparison_rows = []

    # Xếp hạng mô hình riêng cho từng độ dài dự báo và từng chỉ số
    for pred_len, group in df.groupby("Pred_len"):
        for metric in METRICS:
            ranking = (
                group.groupby("Model", as_index=False)[metric]
                .mean()
                .sort_values([metric, "Model"], ascending=[True, True])
                .reset_index(drop=True)
            )

            if len(ranking) != 4:
                raise ValueError(
                    f"Pred_len={pred_len}, metric={metric}: "
                    f"chỉ có {len(ranking)} mô hình."
                )

            row = {
                "Pred_len": pred_len,
                "Metric": metric,
                "Num_models": len(ranking),
            }

            for rank in range(4):
                row[f"Rank_{rank + 1}_Model"] = ranking.loc[rank, "Model"]
                row[f"Rank_{rank + 1}_Value"] = ranking.loc[rank, metric]

            row["Best_model"] = ranking.loc[0, "Model"]
            row["Best_value"] = ranking.loc[0, metric]
            comparison_rows.append(row)

    result = pd.DataFrame(comparison_rows)
    result["_metric_order"] = result["Metric"].map({"MSE": 0, "MAE": 1})
    result = (
        result.sort_values(["Pred_len", "_metric_order"])
        .drop(columns="_metric_order")
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    print("\nBẢNG SO SÁNH 4 MÔ HÌNH")
    print(result.to_string(index=False))
    print("\nĐã lưu bảng tại:", OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()