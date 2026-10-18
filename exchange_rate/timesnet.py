
from pathlib import Path
import csv
import re
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TIMESNET_DIR = PROJECT_ROOT / "baselines" / "TimesNet_official"
DATASET_DIR = PROJECT_ROOT / "experiments" / "datasets"
DATASET_FILE = DATASET_DIR / "exchange_rate.csv"

RESULTS_DIR = PROJECT_ROOT / "exchange_rate" / "results" / "TimesNet"
RESULTS_CSV = RESULTS_DIR / "results_timesnet.csv"

SEQ_LEN = 336
LABEL_LEN = 48
PRED_LENGTHS = [720]

# Cấu hình TimesNet phù hợp để bắt đầu thực nghiệm trên GPU 4 GB.
BATCH_SIZE = 32
TRAIN_EPOCHS = 100
PATIENCE = 5


def save_metrics(pred_len, mse, mae):
    """Lưu hoặc cập nhật kết quả MSE, MAE vào file CSV."""
    rows = []

    if RESULTS_CSV.exists():
        with RESULTS_CSV.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

    fieldnames = ["Model", "Seq_len", "Pred_len", "MSE", "MAE"]

    # Nếu chạy lại cùng pred_len thì cập nhật kết quả cũ.
    rows = [
        row for row in rows
        if not (
            row.get("Model") == "TimesNet"
            and row.get("Seq_len") == str(SEQ_LEN)
            and row.get("Pred_len") == str(pred_len)
        )
    ]

    rows.append({
        "Model": "TimesNet",
        "Seq_len": SEQ_LEN,
        "Pred_len": pred_len,
        "MSE": mse,
        "MAE": mae,
    })

    rows.sort(key=lambda row: int(row["Pred_len"]))

    with RESULTS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã lưu MSE, MAE vào: {RESULTS_CSV}")


def extract_metrics(log_text):
    """Lấy MSE và MAE từ dòng kết quả cuối cùng trong log."""
    matches = re.findall(
        r"mse\s*:\s*([0-9.eE+-]+)\s*,\s*mae\s*:\s*([0-9.eE+-]+)",
        log_text,
        flags=re.IGNORECASE,
    )

    if not matches:
        return None, None

    mse, mae = matches[-1]
    return float(mse), float(mae)


def main():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset: {DATASET_FILE}"
        )

    if not (TIMESNET_DIR / "run.py").exists():
        raise FileNotFoundError(
            f"Không tìm thấy run.py trong {TIMESNET_DIR}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for pred_len in PRED_LENGTHS:
        model_id = f"TimesNet_ExchangeRate_{SEQ_LEN}_{pred_len}"

        command = [
            sys.executable,
            "run.py",
            "--task_name", "long_term_forecast",
            "--is_training", "1",
            "--model_id", model_id,
            "--model", "TimesNet",
            "--data", "custom",
            "--root_path", str(DATASET_DIR),
            "--data_path", DATASET_FILE.name,
            "--features", "M",
            "--target", "OT",
            "--freq", "d",
            "--seq_len", str(SEQ_LEN),
            "--label_len", str(LABEL_LEN),
            "--pred_len", str(pred_len),
            "--enc_in", "8",
            "--dec_in", "8",
            "--c_out", "8",
            "--e_layers", "2",
            "--d_layers", "1",
            "--d_model", "32",
            "--d_ff", "32",
            "--top_k", "5",
            "--num_kernels", "6",
            "--batch_size", str(BATCH_SIZE),
            "--train_epochs", str(TRAIN_EPOCHS),
            "--patience", str(PATIENCE),
            "--num_workers", "0",
            "--itr", "1",
        ]

        print("=" * 60)
        print(f"TimesNet - Exchange Rate - pred_len={pred_len}")
        print("=" * 60)

        # Lưu log để có thể trích xuất MSE, MAE sau khi chạy xong.
        log_file = RESULTS_DIR / f"log_pred_len_{pred_len}.txt"

        with log_file.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=TIMESNET_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        print(f"Log đã lưu tại: {log_file}")

        if process.returncode != 0:
            print(
                f"Lần chạy pred_len={pred_len} thất bại "
                f"(mã lỗi {process.returncode})."
            )
            print("Không ghi MSE, MAE cho lần chạy này.")
            print("Xem log để kiểm tra lỗi.")
            continue

        log_text = log_file.read_text(encoding="utf-8", errors="replace")
        mse, mae = extract_metrics(log_text)

        if mse is None or mae is None:
            print(
                f"Không tìm thấy MSE, MAE trong log pred_len={pred_len}."
            )
            print("Kiểm tra định dạng dòng kết quả trong file log.")
            continue

        save_metrics(pred_len, mse, mae)

        # Gom các kết quả test của cấu hình vừa chạy.
        test_dir = TIMESNET_DIR / "test_results"

        if test_dir.exists():
            matched_dirs = [
                path for path in test_dir.iterdir()
                if path.is_dir() and path.name.startswith(model_id)
            ]

            for source in matched_dirs:
                destination = RESULTS_DIR / source.name

                if destination.exists():
                    shutil.rmtree(destination)

                shutil.copytree(source, destination)
                print(f"Đã sao chép kết quả test: {destination}")

        print(
            f"Hoàn thành pred_len={pred_len}: "
            f"MSE={mse:.6f}, MAE={mae:.6f}"
        )

    print("=" * 60)
    print(f"Thư mục kết quả: {RESULTS_DIR}")
    print(f"File CSV: {RESULTS_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
