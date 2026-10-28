# Thực no Electricity
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OFFICIAL_DIR = PROJECT_ROOT / "official"

DATASET_DIR = PROJECT_ROOT / "experiments" / "datasets"
DATASET_FILE = DATASET_DIR / "electricity.csv"
CONFIG_FILE = PROJECT_ROOT / "experiments" / "configs" / "electricity.yaml"

SEQ_LEN = 336

PRED_LENGTHS = [96]


def main():
    print("=" * 65)
    print("TVCFormer Exploratory Experiment - Electricity")
    print("=" * 65)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Official code: {OFFICIAL_DIR}")
    print(f"Dataset      : {DATASET_FILE}")
    print(f"Config       : {CONFIG_FILE}")
    print(f"Sequence len : {SEQ_LEN}")
    print(f"Pred lengths : {PRED_LENGTHS}")
    print("AMP           : Disabled")
    print("=" * 65)

    required_files = [
        ("Official code", OFFICIAL_DIR / "run_TVCFormer.py"),
        ("Dataset", DATASET_FILE),
        ("Config", CONFIG_FILE),
    ]

    for name, path in required_files:
        if not path.is_file():
            print(f"ERROR: {name} not found: {path}")
            return 1

    for pred_len in PRED_LENGTHS:
        model_id = f"TVCFormer_Electricity_{SEQ_LEN}_{pred_len}"

        command = [
            sys.executable,
            "run_TVCFormer.py",
            "--config_yaml", str(CONFIG_FILE),
            "--is_training", "1",
            "--model_id", model_id,
            "--model", "TVCFormer",
            "--data", "custom",
            "--seq_len", str(SEQ_LEN),
            "--pred_len", str(pred_len),
            "--root_path", str(DATASET_DIR),
            "--data_path", DATASET_FILE.name,
            "--target", "OT",
            "--use_amp", "False",
        ]

        print()
        print("=" * 65)
        print(f"Starting Electricity: pred_len={pred_len}")
        print(f"Model ID: {model_id}")
        print("=" * 65)

        try:
            subprocess.run(command, cwd=OFFICIAL_DIR, check=True)

    # xly khi huấn luyện thất bại
        except subprocess.CalledProcessError as error:
            print(
                f"ERROR: Training failed for pred_len={pred_len}, "
                f"exit code={error.returncode}"
            )
            return error.returncode

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            return 130

    print()
    print("=" * 65)
    print("Electricity exploratory experiment completed.")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
