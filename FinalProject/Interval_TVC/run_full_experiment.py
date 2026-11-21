import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT / "FinalProject" / "Interval_TVC"
PYTHON = REPOSITORY_ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER = PROJECT_ROOT / "train_interval.py"

DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
SEEDS = (2021, 2023, 2025)
PRED_LENS = (96, 192, 336, 720)
EXPERIMENTS = ("E0", "E1")


def run_is_complete(dataset: str, experiment: str, pred_len: int, seed: int) -> bool:
    run_dir = (
        PROJECT_ROOT
        / "results"
        / "full_3seeds_v1"
        / experiment
        / "{}_h{}_s{}".format(dataset, pred_len, seed)
    )
    if not (run_dir / "metrics.json").is_file():
        return False
    if not (run_dir / "test_forecasts.npz").is_file():
        return False
    if experiment == "E1":
        if not (run_dir / "calibration.json").is_file():
            return False
        calibrated_dir = (
            PROJECT_ROOT
            / "results"
            / "full_3seeds_v1"
            / "E1-C"
            / "{}_h{}_s{}".format(dataset, pred_len, seed)
        )
        if not (calibrated_dir / "metrics.json").is_file():
            return False
        if not (calibrated_dir / "test_forecasts.npz").is_file():
            return False
    return True


def format_status(dataset: str, experiment: str, pred_len: int, seed: int) -> str:
    status = "completed" if run_is_complete(dataset, experiment, pred_len, seed) else "missing"
    return "{} seed={} horizon={} {}={}".format(dataset, seed, pred_len, experiment, status)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential three-seed full experiment runner for Interval TVCFormer"
    )
    parser.add_argument("--dataset", choices=tuple(DATASETS), action="append")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--pred_lens", type=int, nargs="+", default=list(PRED_LENS))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    selected_datasets = args.dataset or list(DATASETS)
    if args.smoke_test:
        args.seeds = [args.seeds[0]]
        args.pred_lens = [args.pred_lens[0]]
        for dataset in selected_datasets:
            config = PROJECT_ROOT / "configs" / "interval.yaml"
            for experiment in EXPERIMENTS:
                command = [
                    str(PYTHON),
                    "-u",
                    str(RUNNER),
                    "--config",
                    str(config),
                    "--dataset",
                    dataset,
                    "--experiment",
                    experiment,
                    "--pred_len",
                    str(args.pred_lens[0]),
                    "--seed",
                    str(args.seeds[0]),
                    "--smoke_test",
                ]
                if args.batch_size is not None:
                    command += ["--batch_size", str(args.batch_size)]
                print(
                    "SMOKE Dataset={} seed={} horizon={} experiment={}".format(
                        dataset, args.seeds[0], args.pred_lens[0], experiment
                    )
                )
                result = subprocess.run(
                    command,
                    cwd=str(REPOSITORY_ROOT),
                )
                if result.returncode != 0:
                    raise SystemExit(result.returncode)
        print("ALL_SMOKE_TESTS_DONE")
        return

    log_root = PROJECT_ROOT / "logs" / "full_3seeds_v1"
    log_root.mkdir(parents=True, exist_ok=True)
    overall_started = time.perf_counter()

    for dataset in selected_datasets:
        dataset_log_dir = log_root / dataset
        dataset_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = dataset_log_dir / "interval_3seeds_full.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                "\n=== Dataset {} started UTC monotonic={} ===\n".format(
                    dataset, time.strftime("%Y-%m-%d %H:%M:%S")
                )
            )
            dataset_started = time.perf_counter()
            for seed in args.seeds:
                for pred_len in args.pred_lens:
                    for experiment in EXPERIMENTS:
                        if run_is_complete(dataset, experiment, pred_len, seed):
                            line = format_status(dataset, experiment, pred_len, seed)
                            print("SKIP " + line)
                            log.write("SKIP " + line + "\n")
                            log.flush()
                            continue

                        config = PROJECT_ROOT / "configs" / "interval.yaml"
                        command = [
                            str(PYTHON),
                            "-u",
                            str(RUNNER),
                            "--config",
                            str(config),
                            "--dataset",
                            dataset,
                            "--experiment",
                            experiment,
                            "--pred_len",
                            str(pred_len),
                            "--seed",
                            str(seed),
                        ]
                        if args.epochs is not None:
                            command += ["--epochs", str(args.epochs)]
                        if args.batch_size is not None:
                            command += ["--batch_size", str(args.batch_size)]
                        if args.smoke_test:
                            command.append("--smoke_test")

                        line = "START {}".format(
                            format_status(dataset, experiment, pred_len, seed).replace(" missing", "")
                        )
                        print(line)
                        log.write(line + "\n")
                        log.flush()
                        started = time.perf_counter()
                        process = subprocess.Popen(
                            command,
                            cwd=str(REPOSITORY_ROOT),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                        )
                        assert process.stdout is not None
                        for output_line in process.stdout:
                            print(output_line, end="")
                            log.write(output_line)
                            log.flush()
                        exit_code = process.wait()
                        elapsed = time.perf_counter() - started
                        completed = exit_code == 0 and run_is_complete(
                            dataset, experiment, pred_len, seed
                        )
                        if not completed:
                            message = (
                                "FAILED Dataset={} seed={} horizon={} experiment={} "
                                "exit_code={} elapsed_seconds={:.1f}\n"
                            ).format(dataset, seed, pred_len, experiment, exit_code, elapsed)
                            print(message, file=sys.stderr)
                            log.write(message)
                            log.flush()
                            raise SystemExit(exit_code or 1)
                        message = (
                            "DONE Dataset={} seed={} horizon={} experiment={} "
                            "elapsed_seconds={:.1f}\n"
                        ).format(dataset, seed, pred_len, experiment, elapsed)
                        print(message, end="")
                        log.write(message)
                        log.flush()
            elapsed = time.perf_counter() - dataset_started
            log.write(
                "=== Dataset {} finished elapsed_seconds={:.1f} ===\n".format(
                    dataset, elapsed
                )
            )

    if not args.smoke_test:
        aggregate = PROJECT_ROOT / "aggregate_results.py"
        subprocess.run(
            [str(PYTHON), "-u", str(aggregate)],
            cwd=str(REPOSITORY_ROOT),
            check=True,
        )
    print(
        "FULL_RUNNER_DONE elapsed_seconds={:.1f}".format(
            time.perf_counter() - overall_started
        )
    )


if __name__ == "__main__":
    main()
