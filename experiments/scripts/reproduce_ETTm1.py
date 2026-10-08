from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OFFICIAL_DIR = PROJECT_ROOT / "official"

DATASET_DIR = PROJECT_ROOT / "experiments" / "datasets" / "ETT-small"
DATASET_FILE = DATASET_DIR / "ETTm1.csv"
CONFIG_FILE = PROJECT_ROOT / "experiments" / "configs" / "ETTm1.yaml"


def main():
    print("=" * 60)
    print("TVCFormer Reproduction - ETTm1")
    print("=" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Official code: {OFFICIAL_DIR}")
    print(f"Dataset: {DATASET_FILE}")
    print(f"Config: {CONFIG_FILE}")
    print()

    if not DATASET_FILE.exists():
        print("ERROR: ETTm1.csv was not found.")
        return

    if not CONFIG_FILE.exists():
        print("ERROR: ETTm1.yaml was not found.")
        return

    command = [
        sys.executable,
        "run_TVCFormer.py",
        "--config_yaml",
        str(CONFIG_FILE),
        "--is_training",
        "1",
        "--model_id",
        "TVCFormer_ETTm1_336_192",
        "--model",
        "TVCFormer",
        "--data",
        "ETTm1",
        "--seq_len",
        "336",
        "--pred_len",
        "192",
        "--root_path",
        str(DATASET_DIR),
        "--data_path",
        "ETTm1.csv",
    ]

    print("Starting TVCFormer...")
    print()

    subprocess.run(command, cwd=OFFICIAL_DIR, check=True)


if __name__ == "__main__":
    main()