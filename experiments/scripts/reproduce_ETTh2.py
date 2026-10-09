from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OFFICIAL_DIR = PROJECT_ROOT / "official"

DATASET_DIR = PROJECT_ROOT / "experiments" / "datasets" / "ETT-small"
DATASET_FILE = DATASET_DIR / "ETTh2.csv"
CONFIG_FILE = OFFICIAL_DIR / "configs" / "ETTh2.yaml"


def main():
    print("=" * 60)
    print("TVCFormer Reproduction - ETTh2")
    print("=" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Official code: {OFFICIAL_DIR}")
    print(f"Dataset: {DATASET_FILE}")
    print(f"Config: {CONFIG_FILE}")
    print()

    if not DATASET_FILE.exists():
        print("ERROR: ETTh2.csv was not found.")
        return

    if not CONFIG_FILE.exists():
        print("ERROR: ETTh2.yaml was not found.")
        return

    pred_lengths = [96, 192, 336, 720]

    for pred_len in pred_lengths:
        model_id = f"TVCFormer_ETTh2_336_{pred_len}"

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
            "ETTh2",
            "--seq_len",
            "336",
            "--pred_len",
            str(pred_len),
            "--root_path",
            str(DATASET_DIR),
            "--data_path",
            "ETTh2.csv",
        ]

        print("=" * 60)
        print(f"Starting TVCFormer - ETTh2 - pred_len={pred_len}")
        print("=" * 60)
        print()

        subprocess.run(command, cwd=OFFICIAL_DIR, check=True)


if __name__ == "__main__":
    main()