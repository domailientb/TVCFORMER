# Thực no ETTm2
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OFFICIAL_DIR = PROJECT_ROOT / "official"

DATASET_DIR = PROJECT_ROOT / "experiments" / "datasets" / "ETT-small"
DATASET_FILE = DATASET_DIR / "ETTm2.csv"
CONFIG_FILE = PROJECT_ROOT / "experiments" / "configs" / "ETTm2.yaml"


def main():
    print("=" * 60)
    print("TVCFormer Reproduction - ETTm2")
    print("=" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Official code: {OFFICIAL_DIR}")
    print(f"Dataset: {DATASET_FILE}")
    print(f"Config: {CONFIG_FILE}")
    print()

    if not DATASET_FILE.exists():
        print("ERROR: ETTm2.csv was not found.")
        return

    if not CONFIG_FILE.exists():
        print("ERROR: ETTm2.yaml was not found.")
        return

    pred_lengths = [96, 192, 336, 720]

    for pred_len in pred_lengths:
        model_id = f"TVCFormer_ETTm2_336_{pred_len}"

        command = [
            sys.executable,
            "run_TVCFormer.py",
            "--config_yaml",
            str(CONFIG_FILE),
            "--is_training",
            "1",
            "--model_id",
            model_id,
            "--model",
            "TVCFormer",
            "--data",
            "ETTm2",
            "--seq_len",
            "336",
            "--pred_len",
            str(pred_len),
            "--root_path",
            str(DATASET_DIR),
            "--data_path",
            "ETTm2.csv",
        ]

        print("=" * 60)
        print(f"Starting TVCFormer - ETTm2 - pred_len={pred_len}")
        print("=" * 60)
        print()

        subprocess.run(command, cwd=OFFICIAL_DIR, check=True)


if __name__ == "__main__":
    main()
