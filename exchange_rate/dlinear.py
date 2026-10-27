# Chạy thực nghiệm DLinear trên Exchange Rate với bốn độ dài dự báo 
from pathlib import Path
import subprocess
import sys
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = PROJECT_ROOT / "baselines" / "TimesNet_official"
DATASET_FILE = PROJECT_ROOT / "experiments" / "datasets" / "exchange_rate.csv"
RESULT_DIR = PROJECT_ROOT / "exchange_rate" / "results" / "DLinear"

SEQ_LEN = 336
PRED_LENGTHS = [96, 192, 336, 720]

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("DLinear - Exchange Rate")
    print("=" * 60)
    print(f"Thư mục baseline: {BASELINE_DIR}")
    print(f"Dữ liệu: {DATASET_FILE}")
    print(f"Thư mục kết quả: {RESULT_DIR}")

    # Kiểm tra mã nguồn và dữ liệu trước khi chạy
    if not BASELINE_DIR.exists():
        print("LỖI: Không tìm thấy thư mục baseline.")
        return

    run_file = BASELINE_DIR / "run.py"
    if not run_file.exists():
        run_file = BASELINE_DIR / "run_longExp.py"

    if not run_file.exists():
        print("LỖI: Không tìm thấy run.py hoặc run_longExp.py.")
        return

    if not DATASET_FILE.exists():
        print("LỖI: Không tìm thấy exchange_rate.csv.")
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Chạy DLinear lần lượt với từng độ dài dự báo
    for pred_len in PRED_LENGTHS:
        model_id = f"DLinear_Exchange_Rate_{SEQ_LEN}_{pred_len}"

        command = [
            sys.executable,
            str(run_file),
            "--task_name", "long_term_forecast",
            "--is_training", "1",
            "--model_id", model_id,
            "--model", "DLinear",
            "--data", "custom",
            "--root_path", str(DATASET_FILE.parent),
            "--data_path", DATASET_FILE.name,
            "--features", "M",
            "--target", "OT",
            "--freq", "d",
            "--seq_len", str(SEQ_LEN),
            "--label_len", "48",
            "--pred_len", str(pred_len),
            "--enc_in", "8",
            "--dec_in", "8",
            "--c_out", "8",
            "--batch_size", "32",
            "--train_epochs", "100",
            "--patience", "5",
            "--num_workers", "0",
            "--des", "DLinear_Exchange_Rate",
            "--itr", "1",
        ]

        print("\n" + "=" * 60)
        print(f"Bắt đầu DLinear - pred_len={pred_len}")
        print("=" * 60)

        log_file = RESULT_DIR / f"DLinear_pred{pred_len}.log"

        # Ghi đầu ra và thông báo lỗi vào log để tiện kiểm tra
        with open(log_file, "w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=BASELINE_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode != 0:
            print(f"LỖI khi chạy pred_len={pred_len}.")
            print(f"Kiểm tra log: {log_file}")
            return

        print(f"Đã chạy xong pred_len={pred_len}.")
        print(f"Log: {log_file}")
    print("\n" + "=" * 60)
    print("ĐÃ CHẠY XONG CÁC PRED_LEN")
    print(f"Kết quả và log: {RESULT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()