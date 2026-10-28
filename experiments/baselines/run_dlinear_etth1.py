# tái thực nghiệm mô hình DLinear trên bộ dữ liệu ETTh1
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DLINEAR_DIR = PROJECT_ROOT / "baselines" / "DLinear_official"
DATA_DIR = PROJECT_ROOT / "experiments" / "datasets" / "ETT-small"
DATA_FILE = DATA_DIR / "ETTh1.csv"

PRED_LENGTHS = [96, 192, 336, 720]

def main():
    run_file = DLINEAR_DIR / "run_longExp.py"

    # Kiểm tra mã nguồn và dữ liệu trước khi chạy
    if not run_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mã DLinear chính thức: {run_file}"
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu ETTh1: {DATA_FILE}"
        )

    print("=" * 65)
    print("DLinear reproduction - ETTh1")
    print("=" * 65)
    print(f"Official code: {DLINEAR_DIR}")
    print(f"Dataset: {DATA_FILE}")
    print(f"Prediction lengths: {PRED_LENGTHS}")

    # Chạy DLinear lần lượt với từng độ dài dự báo
    for pred_len in PRED_LENGTHS:
        model_id = f"DLinear_ETTh1_336_{pred_len}"

        command = [
            sys.executable,
            "run_longExp.py",
            "--is_training", "1",
            "--model_id", model_id,
            "--model", "DLinear",
            "--data", "ETTh1",
            "--root_path", str(DATA_DIR) + "/",
            "--data_path", "ETTh1.csv",
            "--features", "M",
            "--seq_len", "336",
            "--pred_len", str(pred_len),
            "--enc_in", "7",
            "--itr", "1",
            "--batch_size", "32",
            "--learning_rate", "0.005",
            "--train_epochs", "10",
            "--patience", "3",
            "--lradj", "type1",
            "--num_workers", "0",
            "--des", "Exp",
        ]

        print("\n" + "=" * 65)
        print(f"Running DLinear: pred_len={pred_len}")
        print("=" * 65)

        # Dừng chương trình nếu một lần chạy thất bại
        subprocess.run(
            command,
            cwd=DLINEAR_DIR,
            check=True,
        )

    print("\nĐã chạy xong cả bốn prediction lengths.")

if __name__ == "__main__":
    main()