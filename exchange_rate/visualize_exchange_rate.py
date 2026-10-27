# trực quan hóa kết quả dự báo của TVCFormer trên bộ dữ liệu Exchange Rate
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_DIR = PROJECT_ROOT / "official"
DATASET_DIR = PROJECT_ROOT / "experiments" / "datasets"
DATASET_FILE = DATASET_DIR / "exchange_rate.csv"
CONFIG_FILE = PROJECT_ROOT / "exchange_rate" / "exchange_rate.yaml"
SOURCE_DIR = OFFICIAL_DIR / "visualize_results"
RESULTS_DIR = PROJECT_ROOT / "exchange_rate" / "results" / "visualize_results"

SEQ_LEN = 336
PRED_LENGTHS = [96, 192, 336, 720]
VISUAL_SAMPLES_NUM = 15

def main():
    # Kiểm tra mã nguồn, dữ liệu và cấu hình trước khi chạy
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset: {DATASET_FILE}"
        )

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy config: {CONFIG_FILE}"
        )

    run_file = OFFICIAL_DIR / "run_TVCFormer.py"

    if not run_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy run_TVCFormer.py trong {OFFICIAL_DIR}"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Trực quan hóa kết quả cho từng độ dài dự báo
    for pred_len in PRED_LENGTHS:
        model_id = f"TVCFormer_ExchangeRate_{SEQ_LEN}_{pred_len}"

        command = [
            sys.executable,
            "run_TVCFormer.py",
            "--config_yaml", str(CONFIG_FILE),
            "--is_training", "0",
            "--model_id", model_id,
            "--model", "TVCFormer",
            "--data", "custom",
            "--root_path", str(DATASET_DIR),
            "--data_path", DATASET_FILE.name,
            "--target", "OT",
            "--enc_in", "8",
            "--seq_len", str(SEQ_LEN),
            "--pred_len", str(pred_len),
            "--visualize", "1",
            "--visual_samples_num", str(VISUAL_SAMPLES_NUM),
        ]

        print("=" * 60)
        print(f"Trực quan hóa Exchange Rate - pred_len={pred_len}")
        print("=" * 60)
        subprocess.run(
            command,
            cwd=OFFICIAL_DIR,
            check=True,
        )

        # Xác định thư mục kết quả của cấu hình hiện tại
        setting = (
            f"{model_id}_dm32_nh8_el2_dl1_df128_"
            f"gch4_gck3_tn7_kr0.7_0"
        )
        source = SOURCE_DIR / setting
        destination = RESULTS_DIR / setting
        if not source.exists():
            raise FileNotFoundError(
                f"Không tìm thấy kết quả trực quan hóa: {source}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Thay thế kết quả cũ của cùng cấu hình bằng kết quả mới
        if destination.exists():
            shutil.rmtree(destination)

        shutil.move(str(source), str(destination))
        print(f"Đã lưu kết quả tại: {destination}")

    print("=" * 60)
    print("Đã hoàn tất trực quan hóa Exchange Rate.")
    print(f"Thư mục kết quả: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()