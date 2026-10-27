# Chạy thực nghiệm TVCFormer trên bộ dữ liệu Exchange Rate với bốn độ dài dự báo.
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_DIR = PROJECT_ROOT / "official"
DATASET_DIR = PROJECT_ROOT / "experiments" / "datasets"
DATASET_FILE = DATASET_DIR / "exchange_rate.csv"
CONFIG_FILE = PROJECT_ROOT / "exchange_rate" / "exchange_rate.yaml"

SEQ_LEN = 336
PRED_LENGTHS = [96, 192, 336, 720]

def main():
    print("=" * 60)
    print("TVCFormer Reproduction - Exchange Rate")
    print("=" * 60)

    # Kiểm tra dữ liệu, mã nguồn và cấu hình trước khi chạy
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset: {DATASET_FILE}"
        )

    run_file = OFFICIAL_DIR / "run_TVCFormer.py"
    if not run_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mã nguồn TVCFormer: {run_file}"
        )

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy config: {CONFIG_FILE}"
        )

    # Chạy TVCFormer lần lượt với từng độ dài dự báo
    for pred_len in PRED_LENGTHS:
        model_id = f"TVCFormer_ExchangeRate_{SEQ_LEN}_{pred_len}"

        command = [
            sys.executable,
            "run_TVCFormer.py",
            "--config_yaml", str(CONFIG_FILE),
            "--is_training", "1",
            "--model_id", model_id,
            "--model", "TVCFormer",
            "--data", "custom",
            "--root_path", str(DATASET_DIR),
            "--data_path", DATASET_FILE.name,
            "--target", "OT",
            "--enc_in", "8",
            "--seq_len", str(SEQ_LEN),
            "--pred_len", str(pred_len),
        ]

        print("\n" + "=" * 60)
        print(f"Exchange Rate | pred_len={pred_len}")
        print("=" * 60)

        # Dừng chương trình nếu lần chạy hiện tại thất bại
        subprocess.run(
            command,
            cwd=OFFICIAL_DIR,
            check=True,
        )

    print("\nĐã chạy xong cả bốn cấu hình TVCFormer.")


if __name__ == "__main__":
    main()