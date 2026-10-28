from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_DIR = PROJECT_ROOT / "official"

DATASET_CONFIGS = {
    "ETTh1": {
        "dataset_dir": "ETT-small",
        "data_file": "ETTh1.csv",
        "config_file": "ETTh1.yaml",
    },
    "ETTh2": {
        "dataset_dir": "ETT-small",
        "data_file": "ETTh2.csv",
        "config_file": "ETTh2.yaml",
    },
}


def main():
    if len(sys.argv) != 2:
        print("Usage: python experiments/reproduce_tvcformer.py <dataset>")
        print("Available datasets:")
        for dataset in DATASET_CONFIGS:
            print(f"  - {dataset}")
        return

    dataset = sys.argv[1]

    if dataset not in DATASET_CONFIGS:
        print(f"ERROR: Unknown dataset '{dataset}'")
        print("Available datasets:")
        for name in DATASET_CONFIGS:
            print(f"  - {name}")
        return

    config = DATASET_CONFIGS[dataset]

    dataset_dir = PROJECT_ROOT / "experiments" / "datasets" / config["dataset_dir"]
    dataset_file = dataset_dir / config["data_file"]
    config_file = PROJECT_ROOT / "experiments" / "configs" / config["config_file"]

    print("=" * 60)
    print(f"TVCFormer Reproduction - {dataset}")
    print("=" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Official code: {OFFICIAL_DIR}")
    print(f"Dataset: {dataset_file}")
    print(f"Config: {config_file}")
    print()

    if not dataset_file.exists():
        print(f"ERROR: {config['data_file']} was not found.")
        return

    if not config_file.exists():
        print(f"ERROR: {config['config_file']} was not found.")
        return

    command = [
        sys.executable,
        "run_TVCFormer.py",
        "--config_yaml",
        str(config_file),
        "--is_training",
        "1",
        "--model_id",
        f"TVCFormer_{dataset}_336_192",
        "--model",
        "TVCFormer",
        "--data",
        dataset,
        "--seq_len",
        "336",
        "--pred_len",
        "192",
        "--root_path",
        str(dataset_dir),
        "--data_path",
        config["data_file"],
    ]

    print("Starting TVCFormer...")
    print()

    subprocess.run(command, cwd=OFFICIAL_DIR, check=True)


if __name__ == "__main__":
    main()