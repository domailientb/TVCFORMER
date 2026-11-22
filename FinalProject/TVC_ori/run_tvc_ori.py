"""Run the original TVCFormer entry point with isolated logs and artifacts."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import uuid

import yaml

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
OFFICIAL_ENTRY = ROOT / "official" / "run_TVCFormer.py"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
HORIZONS = (96, 192, 336, 720)
DEFAULT_SEEDS = (2023, 2025)


def load_config(dataset):
    path = PROJECT / "configs" / (dataset + ".yaml")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if dataset == "Weather":
        data_name = "weather"
        data_root = ROOT / "experiments" / "datasets"
    else:
        data_name = dataset
        data_root = ROOT / "experiments" / "datasets" / "ETT-small"
    data_file = data_root / config["data_path"]
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    return path, config, data_name, data_root, data_file


def command_for(dataset, seed, horizon, run_id, config_path, config, data_name, data_root):
    checkpoint_root = PROJECT / "checkpoints" / dataset / ("seed{}".format(seed)) / run_id
    model_id = "TVC_ori_{}_336_{}_seed{}_{}".format(dataset, horizon, seed, run_id)
    command = [
        sys.executable, "-u", str(OFFICIAL_ENTRY),
        "--config_yaml", str(config_path),
        "--is_training", "1",
        "--model", "TVCFormer",
        "--model_id", model_id,
        "--data", data_name,
        "--root_path", str(data_root),
        "--data_path", config["data_path"],
        "--target", "OT",
        "--seq_len", "336",
        "--pred_len", str(horizon),
        "--random_seed", str(seed),
        "--checkpoints", str(checkpoint_root),
        "--use_gpu", "True",
        "--gpu", "0",
    ]
    return command, model_id, checkpoint_root


def run_dataset_seed(dataset, seed, horizons, dry_run=False):
    config_path, config, data_name, data_root, data_file = load_config(dataset)
    logs_dir = PROJECT / "logs"
    results_dir = PROJECT / "results"
    logs_dir.mkdir(parents=True, exist_ok=True) if not dry_run else None
    results_dir.mkdir(parents=True, exist_ok=True) if not dry_run else None
    log_path = logs_dir / "{}_seed{}.log".format(dataset, seed)
    failures = 0

    for horizon in horizons:
        run_id = "{}_{}".format(
            datetime.now().strftime("%Y%m%dT%H%M%S%f"), uuid.uuid4().hex[:8])
        command, model_id, checkpoint_root = command_for(
            dataset, seed, horizon, run_id, config_path, config, data_name, data_root)
        print("DATASET={} SEED={} PRED_LEN={} CONFIG={}".format(
            dataset, seed, horizon, config_path))
        print("MODEL_ID={}".format(model_id))
        print("CHECKPOINT_ROOT={}".format(checkpoint_root))
        print("LOG={}".format(log_path))
        print("COMMAND={}".format(subprocess.list2cmdline(command)))
        if dry_run:
            continue

        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write("\n========== DATASET={} SEED={} PRED_LEN={} START={} ==========\n".format(
                dataset, seed, horizon, datetime.now().isoformat(timespec="seconds")))
            log.write("config_file={}\nmodel_id={}\ncheckpoint_root={}\n".format(
                config_path, model_id, checkpoint_root))
            log.write("command={}\n".format(subprocess.list2cmdline(command)))
            log.flush()
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = str(seed)
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            try:
                completed = subprocess.run(
                    command, cwd=str(PROJECT), env=env, stdout=log,
                    stderr=subprocess.STDOUT, check=False,
                )
                setting = "{}_dm{}_nh{}_el{}_dl{}_df{}_gch{}_gck{}_tn{}_kr{}_0".format(
                    model_id, config["d_model"], config["n_heads"], config["e_layers"],
                    config["d_layers"], config["d_ff"], config["graph_conv_hidden_size"],
                    config["graph_conv_kernel_size"], config["time_node_num"],
                    config["subgraph_keep_rate"])
                result_dir = PROJECT / "results" / setting
                checkpoint_file = checkpoint_root / setting / "checkpoint.pth"
                if completed.returncode == 0 and (result_dir / "pred.npy").is_file() and checkpoint_file.is_file():
                    log.write("RUN_COMPLETED dataset={} seed={} pred_len={} setting={} result_dir={} checkpoint={} end={}\n".format(
                        dataset, seed, horizon, setting, result_dir, checkpoint_file,
                        datetime.now().isoformat(timespec="seconds")))
                else:
                    failures += 1
                    log.write("RUN_FAILED dataset={} seed={} pred_len={} exit_code={} result_exists={} checkpoint_exists={} end={}\n".format(
                        dataset, seed, horizon, completed.returncode,
                        (result_dir / "pred.npy").is_file(), checkpoint_file.is_file(),
                        datetime.now().isoformat(timespec="seconds")))
            except Exception as error:
                failures += 1
                log.write("RUN_FAILED dataset={} seed={} pred_len={} exception={}: {} end={}\n".format(
                    dataset, seed, horizon, type(error).__name__, error,
                    datetime.now().isoformat(timespec="seconds")))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=DEFAULT_SEEDS,
                        default=list(DEFAULT_SEEDS))
    parser.add_argument("--pred_lens", nargs="+", type=int,
                        choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--dry-run", action="store_true",
                        help="validate configuration and print commands without launching training")
    args = parser.parse_args()
    if not OFFICIAL_ENTRY.is_file():
        parser.error("official TVCFormer entry point is missing: {}".format(OFFICIAL_ENTRY))
    failures = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            failures += run_dataset_seed(dataset, seed, args.pred_lens, args.dry_run)
    print("DRY_RUN={} FAILED_HORIZONS={}".format(args.dry_run, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
