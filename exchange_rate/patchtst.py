
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = PROJECT_ROOT / "baselines" / "TimesNet_official"
DATASET_FILE = PROJECT_ROOT / "experiments" / "datasets" / "exchange_rate.csv"
RESULT_DIR = PROJECT_ROOT / "exchange_rate" / "results" / "PatchTST"

SEQ_LEN = 336
PRED_LENGTHS = [96, 192, 336, 720]


def main():
    print("=" * 60)
    print("PatchTST - Exchange Rate")
    print("=" * 60)

    if not BASELINE_DIR.exists():
        print(f"LOI: Khong tim thay baseline: {BASELINE_DIR}")
        return

    run_file = BASELINE_DIR / "run.py"
    if not run_file.exists():
        run_file = BASELINE_DIR / "run_longExp.py"

    if not run_file.exists():
        print("LOI: Khong tim thay run.py hoac run_longExp.py.")
        return

    if not DATASET_FILE.exists():
        print(f"LOI: Khong tim thay du lieu: {DATASET_FILE}")
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for pred_len in PRED_LENGTHS:
        model_id = f"PatchTST_Exchange_Rate_{SEQ_LEN}_{pred_len}"

        command = [
            sys.executable,
            str(run_file),
            "--task_name", "long_term_forecast",
            "--is_training", "1",
            "--model_id", model_id,
            "--model", "PatchTST",
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
            "--des", "PatchTST_Exchange_Rate",
            "--itr", "1",
        ]

        log_file = RESULT_DIR / f"PatchTST_pred{pred_len}.log"

        print("\n" + "=" * 60)
        print(f"Bat dau PatchTST - pred_len={pred_len}")
        print("=" * 60)

        with open(log_file, "w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=BASELINE_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        if result.returncode != 0:
            print(f"LOI khi chay pred_len={pred_len}.")
            print(f"Kiem tra log: {log_file}")
            return

        print(f"Da chay xong pred_len={pred_len}.")
        print(f"Log: {log_file}")

    print("\nDa chay xong ca 4 cau hinh PatchTST.")
    print(f"Log nam tai: {RESULT_DIR}")


if __name__ == "__main__":
    main()
