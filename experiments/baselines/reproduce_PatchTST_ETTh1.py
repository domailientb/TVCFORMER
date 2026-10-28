# tái thực nghiệm mô hình PatchTST trên bộ dữ liệu ETTh1
from pathlib import Path
import csv
import re
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCHTST_DIR = (
    PROJECT_ROOT
    / "baselines"
    / "PatchTST_official"
    / "PatchTST_supervised"
)
DATASET_DIR = PROJECT_ROOT / "experiments" / "datasets" / "ETT-small"
DATASET_FILE = DATASET_DIR / "ETTh1.csv"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "PatchTST"
RESULT_FILE = RESULTS_DIR / "PatchTST_ETTh1_results.csv"
COMPARISON_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "ETTh1_baselines_comparison.csv"
)

SEQ_LEN = 336
PRED_LENGTHS = [96, 192, 336, 720]
RANDOM_SEED = 2021


def extract_metrics(text):
    """Trích xuất MSE và MAE từ log PatchTST."""

    mse_values = re.findall(
        r"\bmse\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )

    mae_values = re.findall(
        r"\bmae\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )

    if not mse_values or not mae_values:
        raise ValueError(
            "Không tìm thấy MSE/MAE trong log. "
            "Hãy kiểm tra phần output của run_longExp.py."
        )

    # Lấy giá trị MSE và MAE xuất hiện cuối cùng trong log
    return float(mse_values[-1]), float(mae_values[-1])

def save_result(pred_len, mse, mae):
    """Lưu kết quả của từng pred_len vào một CSV duy nhất."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Model",
        "Dataset",
        "Seq_len",
        "Pred_len",
        "MSE",
        "MAE",
    ]

    # Giữ lại kết quả khác, loại kết quả cũ trùng cấu hình hiện tại
    existing_rows = []
    if RESULT_FILE.exists() and RESULT_FILE.stat().st_size > 0:
        with RESULT_FILE.open(
            "r", newline="", encoding="utf-8-sig"
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                if not (
                    row.get("Model") == "PatchTST"
                    and row.get("Dataset") == "ETTh1"
                    and row.get("Seq_len") == str(SEQ_LEN)
                    and row.get("Pred_len") == str(pred_len)
                ):
                    existing_rows.append(row)

    # Thêm kết quả mới cho độ dài dự báo hiện tại
    existing_rows.append(
        {
            "Model": "PatchTST",
            "Dataset": "ETTh1",
            "Seq_len": SEQ_LEN,
            "Pred_len": pred_len,
            "MSE": f"{mse:.10f}",
            "MAE": f"{mae:.10f}",
        }
    )
    existing_rows.sort(key=lambda row: int(row["Pred_len"]))

    with RESULT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)

def main():
    print("=" * 65)
    print("TÁI THỰC NGHIỆM PATCHTST - ETTh1")
    print("=" * 65)
    print(f"Thư mục dự án: {PROJECT_ROOT}")
    print(f"Mã nguồn PatchTST: {PATCHTST_DIR}")
    print(f"Dữ liệu: {DATASET_FILE}")
    print(f"Thư mục kết quả: {RESULTS_DIR}")
    print()

    run_file = PATCHTST_DIR / "run_longExp.py"

    # Kiểm tra mã nguồn và dữ liệu trước khi chạy
    if not run_file.exists():
        print("LỖI: Không tìm thấy run_longExp.py:")
        print(run_file)
        print(
            "\nHãy sửa PATCHTST_DIR cho đúng với vị trí "
            "mã nguồn PatchTST trên máy của bạn."
        )
        return

    if not DATASET_FILE.exists():
        print("LỖI: Không tìm thấy file ETTh1.csv:")
        print(DATASET_FILE)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Chạy lần lượt bốn cấu hình dự báo
    for pred_len in PRED_LENGTHS:
        model_id = f"PatchTST_ETTh1_{SEQ_LEN}_{pred_len}"

        command = [
            sys.executable,
            str(run_file),
            "--random_seed", str(RANDOM_SEED),
            "--is_training", "1",
            "--root_path", str(DATASET_DIR) + "\\",
            "--data_path", "ETTh1.csv",
            "--model_id", model_id,
            "--model", "PatchTST",
            "--data", "ETTh1",
            "--features", "M",
            "--seq_len", str(SEQ_LEN),
            "--label_len", "48",
            "--pred_len", str(pred_len),
            "--enc_in", "7",
            "--dec_in", "7",
            "--c_out", "7",
            "--patch_len", "16",
            "--stride", "8",
            "--e_layers", "3",
            "--n_heads", "16",
            "--d_model", "128",
            "--d_ff", "256",
            "--dropout", "0.2",
            "--fc_dropout", "0.2",
            "--head_dropout", "0",
            "--des", "Exp",
            "--train_epochs", "100",
            "--patience", "10",
            "--lradj", "TST",
            "--pct_start", "0.2",
            "--itr", "1",
            "--batch_size", "16",
            "--learning_rate", "0.0001",
            "--num_workers", "0",
        ]

        print("=" * 65)
        print(f"Bắt đầu PatchTST - ETTh1 - pred_len={pred_len}")
        print("=" * 65)

        log_file = RESULTS_DIR / f"PatchTST_ETTh1_{pred_len}.log"

        # Lưu đầu ra và thông báo lỗi của quá trình chạy vào file log
        try:
            with log_file.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    command,
                    cwd=PATCHTST_DIR,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

        except OSError as exc:
            print(f"Không thể khởi động PatchTST: {exc}")
            return

        # Đọc log để lấy chỉ số đánh giá sau khi chạy
        log_text = log_file.read_text(
            encoding="utf-8",
            errors="replace",
        )

        print(log_text[-5000:])

        if process.returncode != 0:
            print(f"LỖI: PatchTST thất bại với pred_len={pred_len}.")
            print(f"Log đã lưu tại: {log_file}")
            return

        try:
            mse, mae = extract_metrics(log_text)
        except ValueError as exc:
            print(exc)
            print(f"Log đã lưu tại: {log_file}")
            return

        # Chỉ lưu kết quả khi chạy thành công và trích xuất được MSE, MAE
        save_result(pred_len, mse, mae)

        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"Đã lưu log: {log_file}")
        print(f"Đã cập nhật CSV: {RESULT_FILE}")
        print()

    print("=" * 65)
    print("HOÀN THÀNH CÁC LẦN CHẠY PATCHTST")
    print("=" * 65)
    print(f"CSV kết quả: {RESULT_FILE}")
    print(f"File so sánh tổng hợp: {COMPARISON_FILE}")


if __name__ == "__main__":
    main()