from pathlib import Path
import os
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TIMESNET_DIR = PROJECT_ROOT / "baselines" / "TimesNet_official"

DATASET_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "datasets"
    / "ETT-small"
)

DATASET_FILE = DATASET_DIR / "ETTh1.csv"

RESULT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "TimesNet"
    / "ETTh1"
)


SEQ_LEN = 96
LABEL_LEN = 48

PRED_LENGTHS = [192, 336, 720]

RANDOM_SEED = 2021

TRAIN_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
PATIENCE = 3
NUM_WORKERS = 0


def main():
    print("=" * 65)
    print("TimesNet reproduction - ETTh1")
    print("=" * 65)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"TimesNet code: {TIMESNET_DIR}")
    print(f"Dataset      : {DATASET_FILE}")
    print(f"Sequence len : {SEQ_LEN}")
    print(f"Prediction   : {PRED_LENGTHS}")
    print(f"Random seed  : {RANDOM_SEED}")
    print()


    if not TIMESNET_DIR.exists():
        print(f"ERROR: TimesNet directory not found: {TIMESNET_DIR}")
        return

    if not (TIMESNET_DIR / "run.py").exists():
        print(f"ERROR: run.py not found in {TIMESNET_DIR}")
        return

    if not DATASET_FILE.exists():
        print(f"ERROR: Dataset not found: {DATASET_FILE}")
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    for pred_len in PRED_LENGTHS:
        model_id = f"ETTh1_{SEQ_LEN}_{pred_len}"

        log_file = (
            RESULT_DIR
            / f"TimesNet_ETTh1_{SEQ_LEN}_{pred_len}_seed{RANDOM_SEED}.log"
        )

        command = [
            sys.executable,
            "run.py",
            "--task_name", "long_term_forecast",
            "--is_training", "1",
            "--root_path", str(DATASET_DIR),
            "--data_path", "ETTh1.csv",
            "--model_id", model_id,
            "--model", "TimesNet",
            "--data", "ETTh1",
            "--features", "M",
            "--seq_len", str(SEQ_LEN),
            "--label_len", str(LABEL_LEN),
            "--pred_len", str(pred_len),
            "--e_layers", "2",
            "--d_layers", "1",
            "--factor", "3",
            "--enc_in", "7",
            "--dec_in", "7",
            "--c_out", "7",
            "--d_model", "16",
            "--d_ff", "32",
            "--top_k", "5",
            "--train_epochs", str(TRAIN_EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--learning_rate", str(LEARNING_RATE),
            "--patience", str(PATIENCE),
            "--num_workers", str(NUM_WORKERS),
            "--seed", str(RANDOM_SEED),
            "--des", "Exp",
            "--itr", "1",
        ]

        print("=" * 65)
        print(
            f"Starting TimesNet: "
            f"seq_len={SEQ_LEN}, pred_len={pred_len}"
        )
        print(f"Log file: {log_file}")
        print("=" * 65)
        print()

        try:
            with open(log_file, "w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=TIMESNET_DIR,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env=env,
                )

            print(f"SUCCESS: pred_len={pred_len}")
            print(f"Log saved to: {log_file}")
            print()

        except subprocess.CalledProcessError as error:
            print(f"ERROR: pred_len={pred_len} failed.")
            print(f"Exit code: {error.returncode}")
            print(f"Check log: {log_file}")
            print()

            if log_file.exists():
                with open(
                    log_file,
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as log:
                    lines = log.readlines()

                print("Last 40 log lines:")
                print("".join(lines[-40:]))

            return

    print("=" * 65)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Results directory: {RESULT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
