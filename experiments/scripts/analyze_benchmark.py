# ptich và xếp hạg các mô hình, xác định top 3 mô hình tốt nhất, thứ hạg tvcformer
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
INPUT_FILE = RESULTS_DIR / "benchmark_results.csv"
OUTPUT_FILE = RESULTS_DIR / "model_comparison_summary.csv"

METRICS = ["MSE", "MAE"]
TVCFORMER_NAME = "TVCFormer"

# ptich và xếp hạg mô hình 
def analyze_metric(group, metric):
    # Chỉ giữ các mô hình có giá trị chỉ số đánh giá hợp lệ
    valid = group.dropna(subset=[metric]).copy()

    # Kiểm tra mô hình có nhiều kết quả trong cùng một nhóm hay không
    duplicated = valid[valid.duplicated(subset=["Model"], keep=False)]

    if not duplicated.empty:
        duplicate_models = sorted(duplicated["Model"].unique())
        print(
            f"WARNING: Duplicate results for "
            f"{group['Dataset'].iloc[0]}, "
            f"pred_len={group['Pred_len'].iloc[0]}, "
            f"metric={metric}: {duplicate_models}"
        )
        return None

    if valid.empty:
        return None

    # Xếp hạng theo chỉ số đánh giá: MSE và MAE càng thấp thì thứ hạng càng cao
    valid["Rank"] = valid[metric].rank(
        method="min",
        ascending=True,
    ).astype(int)

    valid = valid.sort_values(
        by=[metric, "Model"],
        ascending=[True, True],
    ).reset_index(drop=True)

    # Lấy ba mô hình có kết quả tốt nhất
    top1 = valid.iloc[0]
    top2 = valid.iloc[1] if len(valid) >= 2 else None
    top3 = valid.iloc[2] if len(valid) >= 3 else None

    # Lấy giá trị và thứ hạng của TVCFormer nếu có
    tvc_rows = valid[valid["Model"] == TVCFORMER_NAME]

    if not tvc_rows.empty:
        tvc_value = float(tvc_rows.iloc[0][metric])
        tvc_rank = int(tvc_rows.iloc[0]["Rank"])
    else:
        tvc_value = None
        tvc_rank = None

    # Tìm mô hình đối chứng (baseline) tốt nhất, không tính TVCFormer
    baselines = valid[valid["Model"] != TVCFORMER_NAME]

    if not baselines.empty:
        best_baseline = baselines.iloc[0]
        best_baseline_name = best_baseline["Model"]
        best_baseline_value = float(best_baseline[metric])
    else:
        best_baseline_name = None
        best_baseline_value = None

    # Tính phần trăm cải thiện của TVCFormer so với baseline tốt nhất
    # Kết quả dương: TVCFormer có sai số thấp hơn baseline
    if (
        tvc_value is not None
        and best_baseline_value is not None
        and best_baseline_value != 0
    ):
        improvement = (
            (best_baseline_value - tvc_value)
            / best_baseline_value
            * 100
        )
    else:
        improvement = None

    return {
        "Dataset": group["Dataset"].iloc[0],
        "Pred_len": int(group["Pred_len"].iloc[0]),
        "Metric": metric,
        "Num_models": len(valid),

        "Best_model": top1["Model"],
        "Best_value": float(top1[metric]),

        "Second_best_model": (
            top2["Model"] if top2 is not None else None
        ),
        "Second_best_value": (
            float(top2[metric]) if top2 is not None else None
        ),

        "Third_best_model": (
            top3["Model"] if top3 is not None else None
        ),
        "Third_best_value": (
            float(top3[metric]) if top3 is not None else None
        ),

        "TVCFormer_value": tvc_value,
        "TVCFormer_rank": tvc_rank,

        "Best_baseline": best_baseline_name,
        "Best_baseline_value": best_baseline_value,
        "TVCFormer_improvement_percent": improvement,
    }


def main():
    print("=" * 70)
    print("TVCFormer Benchmark Analysis")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print(f"ERROR: Results file not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE, skipinitialspace=True)

    # Loại bỏ khoảng trắng thừa trong tên cột
    df.columns = df.columns.str.strip()

    required_columns = {
        "Dataset",
        "Pred_len",
        "Model",
        *METRICS,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print(f"ERROR: Missing columns: {sorted(missing_columns)}")
        return

    # làm sạch dl
    df["Dataset"] = df["Dataset"].astype(str).str.strip()
    df["Model"] = df["Model"].astype(str).str.strip()
    df["Pred_len"] = pd.to_numeric(
        df["Pred_len"],
        errors="coerce",
    )

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    # Loại các dòng thiếu thông tin định danh trước khi phân nhóm
    df = df.dropna(subset=["Dataset", "Pred_len", "Model"])
    df["Pred_len"] = df["Pred_len"].astype(int)

    summary_rows = []

    # So sánh các mô hình theo từng bộ dữ liệu và độ dài dự báo
    for (dataset, pred_len), group in df.groupby(
        ["Dataset", "Pred_len"],
        sort=True,
    ):
        for metric in METRICS:
            result = analyze_metric(group, metric)

            if result is not None:
                summary_rows.append(result)

    if not summary_rows:
        print("ERROR: No valid results to analyze.")
        return

    summary = pd.DataFrame(summary_rows)


    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print()
    print(f"Input rows: {len(df)}")
    print(f"Models: {df['Model'].nunique()}")
    print(f"Datasets: {df['Dataset'].nunique()}")
    print(f"Summary rows: {len(summary)}")
    print(f"Saved to: {OUTPUT_FILE}")

    # Display the top three models for each dataset and prediction length.
    for dataset in sorted(summary["Dataset"].unique()):
        print()
        print("-" * 70)
        print(f"Dataset: {dataset}")
        print("-" * 70)

        dataset_summary = summary[summary["Dataset"] == dataset]

        for _, row in dataset_summary.sort_values(
            ["Pred_len", "Metric"]
        ).iterrows():
            print(
                f"\nPred_len={row['Pred_len']} | "
                f"Metric={row['Metric']}"
            )
            print(
                f"  Top 1: {row['Best_model']} "
                f"({row['Best_value']:.6f})"
            )

            if pd.notna(row["Second_best_model"]):
                print(
                    f"  Top 2: {row['Second_best_model']} "
                    f"({row['Second_best_value']:.6f})"
                )

            if pd.notna(row["Third_best_model"]):
                print(
                    f"  Top 3: {row['Third_best_model']} "
                    f"({row['Third_best_value']:.6f})"
                )

            if pd.notna(row["TVCFormer_rank"]):
                print(
                    f"  TVCFormer rank: "
                    f"{int(row['TVCFormer_rank'])}/{int(row['Num_models'])}"
                )

            if pd.notna(row["TVCFormer_improvement_percent"]):
                print(
                    f"  TVCFormer improvement vs best baseline: "
                    f"{row['TVCFormer_improvement_percent']:.2f}%"
                )

    print()
    print("=" * 70)
    print("Analysis completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
