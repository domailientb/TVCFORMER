# # Thực no Weather
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OFFICIAL_DIR = PROJECT_ROOT / "official"

DATASET_DIR = PROJECT_ROOT / "experiments" / "datasets"
DATASET_FILE = DATASET_DIR / "weather.csv"
CONFIG_FILE = PROJECT_ROOT / "experiments" / "configs" / "weather.yaml"


def main():
    print("=" * 60)
    print("TVCFormer Reproduction - Weather")
    print("=" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Official code: {OFFICIAL_DIR}")
    print(f"Dataset: {DATASET_FILE}")
    print(f"Config: {CONFIG_FILE}")
    print()

    if not DATASET_FILE.exists():
        print("ERROR: weather.csv was not found.")
        return

    if not CONFIG_FILE.exists():
        print("ERROR: weather.yaml was not found.")
        return

    pred_lengths = [96, 192, 336, 720]

    for pred_len in pred_lengths:
        model_id = f"TVCFormer_weather_336_{pred_len}"

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
            "custom",
            "--seq_len",
            "336",
            "--pred_len",
            str(pred_len),
            "--root_path",
            str(DATASET_DIR),
            "--data_path",
            "weather.csv",
        ]

        print("=" * 60)
        print(f"Starting TVCFormer - Weather - pred_len={pred_len}")
        print("=" * 60)
        print()

        subprocess.run(command, cwd=OFFICIAL_DIR, check=True)


if __name__ == "__main__":
    main()
