"""Mean-replacement feature ablation for official TVCFormer on ETTh1.

The baseline is trained once per horizon. For each ablation, one input
channel is replaced by that channel's mean on the train split, in the
standardized input space. We then evaluate the same checkpoint on the same
test windows; the target labels are never changed.
"""
import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import dgl
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TR_DIR = ROOT / "FinalProject/TR_TVC"
sys.path[:0] = [str(ROOT / "official"), str(TR_DIR / "models"), str(TR_DIR)]

from exp_tr_tvc import Exp_TR_TVC  # noqa: E402
from utils.metrics import metric  # noqa: E402


FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
DEFAULT_HORIZONS = [96, 192, 336, 720]
CONFIG_PATH = HERE / "configs" / "ETTh1.yaml"
RESULTS_DIR = HERE / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="ETTh1 mean-replacement ablation for TVCFormer")
    parser.add_argument("--pred_lens", default="96,192,336,720")
    parser.add_argument("--random_seed", type=int, default=2021)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--use_gpu", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_args(pred_len, cli):
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f) or {}
    if "traget" in config and "target" not in config:
        config["target"] = config.pop("traget")

    args = {
        "random_seed": cli.random_seed,
        "data": "ETTh1",
        "root_path": str(ROOT / "experiments/datasets/ETT-small"),
        "data_path": "ETTh1.csv",
        "target": "OT",
        "scale": True,
        "seq_len": 336,
        "label_len": 0,
        "pred_len": int(pred_len),
        "model": "TVCFormer",
        "model_id": "ETTh1_TVCFormer_ablation_pl{}_s{}".format(pred_len, cli.random_seed),
        "is_training": 1,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "patch_len": 12,
        "stride": 6,
        "padding_patch": "start",
        "time_node_num": 7,
        "gnn_r_degree": 1,
        "gnn_hidden_size": 32,
        "graph_conv_hidden_size": 4,
        "graph_conv_kernel_size": 3,
        "subgraph_keep_rate": 0.7,
        "graph_keep_self_cycles": False,
        "e_layers": 2,
        "d_layers": 1,
        "d_model": 32,
        "n_heads": 8,
        "d_ff": 128,
        "dropout": 0.5,
        "fc_dropout": 0.05,
        "head_dropout": 0.0,
        "individual": False,
        "revins": True,
        "affine": False,
        "subtract_last": False,
        "eps": 1e-5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "train_epochs": 100,
        "patience": 5,
        "pct_start": 0.3,
        "lradj": "type3",
        "weight_decay": 0.0,
        "num_workers": 0,
        "use_gpu": bool(cli.use_gpu and torch.cuda.is_available()),
        "gpu": 0,
        "use_multi_gpu": False,
        "devices": "0",
        "use_amp": False,
        "test_flop": False,
        "decomposition": False,
        "decomposition_kernel_size": 25,
        "checkpoints": str(HERE / "checkpoints"),
        # Exp_TR_TVC is used as a shared validation-only TVC trainer. These
        # weights are ignored for the ordinary TVCFormer model.
        "trend_loss_weight": 0.0,
        "residual_loss_weight": 0.0,
    }
    args.update(config)
    # Pin paths and protocol after loading the baseline YAML.
    args.update({
        "data": "ETTh1", "root_path": str(ROOT / "experiments/datasets/ETT-small"),
        "data_path": "ETTh1.csv", "target": "OT", "seq_len": 336,
        "label_len": 0, "pred_len": int(pred_len), "model": "TVCFormer",
        "model_id": "ETTh1_TVCFormer_ablation_pl{}_s{}".format(pred_len, cli.random_seed),
        "random_seed": int(cli.random_seed), "enc_in": 7, "dec_in": 7, "c_out": 7,
        "checkpoints": str(HERE / "checkpoints"),
        "use_gpu": bool(cli.use_gpu and torch.cuda.is_available()),
        "use_multi_gpu": False,
    })
    if cli.epochs is not None:
        args["train_epochs"] = int(cli.epochs)
    if cli.batch_size is not None:
        args["batch_size"] = int(cli.batch_size)
    return SimpleNamespace(**args)


def score_predictions(predictions, targets):
    mae, mse, rmse, _, _, rse, _ = metric(predictions, targets)
    return {"mse": float(mse), "mae": float(mae), "rmse": float(rmse), "rse": float(rse)}


def evaluate_ablation(exp, test_loader, train_mean, pred_len, setting):
    variants = [("Baseline", None)] + [
        ("Ablation {}".format(index + 1), feature) for index, feature in enumerate(FEATURES)
    ]
    rows = []
    exp.model.eval()
    with torch.no_grad():
        for name, masked_feature in variants:
            prediction_batches, target_batches = [], []
            for batch_x, batch_y, _, _ in test_loader:
                x = batch_x.float().to(exp.device)
                target = batch_y[:, -pred_len:, :].float().to(exp.device)
                if masked_feature is not None:
                    feature_index = FEATURES.index(masked_feature)
                    x = x.clone()
                    x[:, :, feature_index] = train_mean[feature_index]
                prediction_batches.append(exp.model(x).cpu().numpy())
                target_batches.append(target.cpu().numpy())
            predictions = np.concatenate(prediction_batches, axis=0)
            targets = np.concatenate(target_batches, axis=0)
            values = score_predictions(predictions, targets)
            rows.append({
                "dataset": "ETTh1", "pred_len": pred_len, "seed": exp.args.random_seed,
                "experiment": name, "masked_variable": masked_feature or "None",
                "replacement_value_scaled": "" if masked_feature is None else float(train_mean[FEATURES.index(masked_feature)]),
                **values, "influence_mse": 0.0, "influence_mae": 0.0,
                "checkpoint_setting": setting,
            })
            print("{} | mask={} | MSE={:.6f} MAE={:.6f}".format(
                name, masked_feature or "None", values["mse"], values["mae"]))

    baseline = rows[0]
    for row in rows[1:]:
        row["influence_mse"] = row["mse"] - baseline["mse"]
        row["influence_mae"] = row["mae"] - baseline["mae"]
    return rows


def write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "ETTh1_mean_replacement_ablation.csv"
    fields = [
        "dataset", "pred_len", "seed", "experiment", "masked_variable",
        "replacement_value_scaled", "mse", "mae", "rmse", "rse",
        "influence_mse", "influence_mae", "checkpoint_setting",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    cli = parse_args()
    pred_lens = [int(value.strip()) for value in cli.pred_lens.split(",") if value.strip()]
    results = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "ETTh1_mean_replacement_ablation.log"

    with open(log_path, "a", encoding="utf-8") as log:
        class Tee:
            def write(self, value):
                sys.__stdout__.write(value)
                log.write(value)
                sys.__stdout__.flush()
                log.flush()

            def flush(self):
                sys.__stdout__.flush()
                log.flush()

        old_stdout = sys.stdout
        sys.stdout = Tee()
        try:
            print("ETTh1 mean-replacement ablation", datetime.now())
            print("horizons:", pred_lens, "seed:", cli.random_seed)
            for pred_len in pred_lens:
                args = build_args(pred_len, cli)
                seed_everything(args.random_seed)
                exp = Exp_TR_TVC(args)
                setting = "ETTh1_TVCFormer_ablation_pl{}_seed{}".format(pred_len, cli.random_seed)
                print("\nTraining full-feature baseline:", setting)
                exp.train(setting)

                # Dataset_ETTh fits the StandardScaler on the official train
                # segment. Its transformed train rows have mean zero; use the
                # observed train-split mean to preserve exact numeric behavior.
                train_data, _ = exp._get_data(flag="train")
                train_mean = torch.as_tensor(
                    np.asarray(train_data.data_x).mean(axis=0),
                    dtype=torch.float32,
                    device=exp.device,
                )
                _, test_loader = exp._get_data(flag="test")
                results.extend(evaluate_ablation(exp, test_loader, train_mean, pred_len, setting))
                write_csv(results)
                del exp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print("\nCSV:", write_csv(results))
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    main()
