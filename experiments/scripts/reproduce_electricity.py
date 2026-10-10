import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = PROJECT_ROOT / "official"
DATASET_DIR = PROJECT_ROOT / "experiments" / "datasets"

PRED_LENS = [96, 192, 336, 720]
SEQ_LEN = 336


def main():
    data_path = DATASET_DIR / "electricity.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset: {data_path}\n"
            "Hay kiem tra ten file va dat dataset vao experiments/datasets."
        )

    for pred_len in PRED_LENS:
        command = [
            sys.executable,
            str(OFFICIAL_DIR / "run_TVCFormer.py"),
            "--config_yaml", str(OFFICIAL_DIR / "configs" / "electricity.yaml"),
            "--model", "TVCFormer",
            "--data", "custom",
            "--root_path", str(DATASET_DIR),
            "--data_path", data_path.name,
            "--seq_len", str(SEQ_LEN),
            "--pred_len", str(pred_len),
        ]

        print(f"\n===== Electricity | pred_len={pred_len} =====", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
