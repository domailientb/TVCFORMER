import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data import Subset


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = REPOSITORY_ROOT / "official"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(OFFICIAL_ROOT) in sys.path:
    sys.path.remove(str(OFFICIAL_ROOT))
sys.path.insert(0, str(OFFICIAL_ROOT))

from data_provider.data_loader import (  # noqa: E402
    Dataset_Custom,
    Dataset_ETTh,
    Dataset_ETTm,
)
from utils.tools import adjust_learning_rate  # noqa: E402

from FinalProject.Interval_TVC.losses.quantile_loss import (  # noqa: E402
    QUANTILE_LEVELS,
    quantile_loss,
)
from FinalProject.Interval_TVC.calibration.conformal import (  # noqa: E402
    CALIBRATION_METHODS,
    apply_cqr_calibration,
    fit_cqr_calibration,
    split_validation_indices,
)
from FinalProject.Interval_TVC.metrics.interval_metrics import (  # noqa: E402
    compute_forecast_metrics,
    inverse_scale,
)
from FinalProject.Interval_TVC.models.interval_tvcformer import (  # noqa: E402
    IntervalTVCFormer,
)
from FinalProject.Interval_TVC.official_bridge import TVCFormer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETTh1 point/quantile TVCFormer experiment")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "ETTh1.yaml"),
    )
    parser.add_argument("--experiment", choices=("E0", "E1"), required=True)
    parser.add_argument("--pred_len", type=int, choices=(96, 192, 336, 720))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="evaluate an existing best checkpoint without training",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument(
        "--dataset",
        choices=("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"),
    )
    return parser.parse_args()


def load_config(path: str, dataset: str = None) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")
    if "datasets" in config:
        if dataset is None:
            raise ValueError("a dataset name is required for the combined interval config")
        datasets = config["datasets"]
        if not isinstance(datasets, dict) or dataset not in datasets:
            raise ValueError("dataset not found in combined interval config: {}".format(dataset))
        config = dict(datasets[dataset])
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import dgl
    except ImportError:
        return
    dgl.random.seed(seed)


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def build_args(config: Dict[str, object], device: torch.device) -> SimpleNamespace:
    values = dict(config)
    values["device"] = str(device)
    values["use_gpu"] = device.type == "cuda"
    values["model"] = "TVCFormer"
    values["d_layers"] = 1
    values["pct_start"] = 0.3
    values["fc_dropout"] = float(values.get("fc_dropout", 0.0))
    values["head_dropout"] = float(values.get("head_dropout", 0.0))
    return SimpleNamespace(**values)


def make_dataset_and_loader(
    args: SimpleNamespace, split: str, shuffle: bool
) -> Tuple[object, DataLoader]:
    dataset_class = {
        "ETTh": Dataset_ETTh,
        "ETTm": Dataset_ETTm,
        "Custom": Dataset_Custom,
    }.get(args.dataset_loader)
    if dataset_class is None:
        raise ValueError("unsupported dataset_loader: {}".format(args.dataset_loader))
    dataset = dataset_class(
        root_path=str(resolve_path(args.root_path)),
        flag=split,
        size=[args.seq_len, args.label_len, args.pred_len],
        data_path=args.data_path,
        target=args.target,
        scale=args.scale,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=shuffle,
        pin_memory=args.device.startswith("cuda"),
    )
    print("{} {}".format(split, len(dataset)))
    return dataset, loader


def build_model(experiment: str, args: SimpleNamespace) -> torch.nn.Module:
    if experiment == "E0":
        return TVCFormer.Model(args)
    return IntervalTVCFormer(args)


def loss_for_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    experiment: str,
) -> torch.Tensor:
    output = model(x)
    if experiment == "E1":
        return quantile_loss(output, y)
    return torch.nn.functional.mse_loss(output, y)


def evaluate_loss(
    model: torch.nn.Module,
    loader: Iterable,
    args: SimpleNamespace,
    experiment: str,
    device: torch.device,
) -> float:
    model.eval()
    loss_total = 0.0
    sample_total = 0
    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            targets = batch_y[:, -args.pred_len :, :].float().to(device, non_blocking=True)
            loss = loss_for_batch(model, batch_x, targets, experiment)
            loss_total += float(loss.item()) * batch_x.shape[0]
            sample_total += batch_x.shape[0]
    if sample_total == 0:
        raise ValueError("evaluation loader produced no batches")
    return loss_total / sample_total


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    args: SimpleNamespace,
    experiment: str,
    device: torch.device,
) -> float:
    model.train()
    loss_total = 0.0
    sample_total = 0
    for batch_x, batch_y, _, _ in loader:
        batch_x = batch_x.float().to(device, non_blocking=True)
        targets = batch_y[:, -args.pred_len :, :].float().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for_batch(model, batch_x, targets, experiment)
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss became non-finite")
        loss.backward()
        optimizer.step()
        loss_total += float(loss.detach().item()) * batch_x.shape[0]
        sample_total += batch_x.shape[0]
    if sample_total == 0:
        raise ValueError("training loader produced no batches")
    return loss_total / sample_total


def collect_forecasts(
    model: torch.nn.Module,
    loader: Iterable,
    args: SimpleNamespace,
    experiment: str,
    device: torch.device,
    storage_dir: Path = None,
    storage_name: str = "forecasts",
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    if storage_dir is not None:
        storage_dir.mkdir(parents=True, exist_ok=True)
        sample_count = len(loader.dataset)
        prediction_shape = (
            (sample_count, args.pred_len, args.enc_in, 3)
            if experiment == "E1"
            else (sample_count, args.pred_len, args.enc_in)
        )
        predictions = np.lib.format.open_memmap(
            storage_dir / (storage_name + "_predictions.npy"),
            mode="w+", dtype=np.float32, shape=prediction_shape,
        )
        targets = np.lib.format.open_memmap(
            storage_dir / (storage_name + "_targets.npy"),
            mode="w+", dtype=np.float32,
            shape=(sample_count, args.pred_len, args.enc_in),
        )
        offset = 0
        with torch.no_grad():
            for batch_x, batch_y, _, _ in loader:
                batch_x = batch_x.float().to(device, non_blocking=True)
                output = model(batch_x)
                count = batch_x.shape[0]
                predictions[offset : offset + count] = output.detach().cpu().numpy()
                targets[offset : offset + count] = batch_y[:, -args.pred_len :, :].float().numpy()
                offset += count
        if offset != sample_count:
            raise ValueError("forecast count mismatch: wrote {}, expected {}".format(offset, sample_count))
        predictions.flush()
        targets.flush()
        return predictions, targets
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            target = batch_y[:, -args.pred_len :, :].float()
            output = model(batch_x)
            predictions.append(output.detach().cpu().numpy())
            targets.append(target.numpy())
    if not predictions:
        raise ValueError("test loader produced no batches")
    return np.concatenate(predictions), np.concatenate(targets)


def inverse_scale_memmap(
    values: np.ndarray, scaler: object, path: Path, chunk_samples: int = 64
) -> np.ndarray:
    shape = values.shape
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, shape[0], chunk_samples):
        stop = min(start + chunk_samples, shape[0])
        output[start:stop] = inverse_scale(values[start:stop], scaler)
    output.flush()
    return output


def calibrate_memmap_chunks(
    lower: np.ndarray,
    upper: np.ndarray,
    median: np.ndarray,
    calibrator: Dict[str, object],
    lower_path: Path,
    upper_path: Path,
    chunk_samples: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    calibrated_lower = np.lib.format.open_memmap(
        lower_path, mode="w+", dtype=np.float32, shape=lower.shape
    )
    calibrated_upper = np.lib.format.open_memmap(
        upper_path, mode="w+", dtype=np.float32, shape=upper.shape
    )
    for start in range(0, lower.shape[0], chunk_samples):
        stop = min(start + chunk_samples, lower.shape[0])
        lo, hi = apply_cqr_calibration(
            lower[start:stop], upper[start:stop], median[start:stop], calibrator
        )
        calibrated_lower[start:stop] = lo
        calibrated_upper[start:stop] = hi
    calibrated_lower.flush()
    calibrated_upper.flush()
    return calibrated_lower, calibrated_upper


def _save_forecasts_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary_path = path.with_name(path.stem + ".tmp.npz")
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_smoke_test(
    args: SimpleNamespace,
    experiment: str,
    train_loader: DataLoader,
    device: torch.device,
) -> None:
    model = build_model(experiment, args).to(device)
    batch_x, batch_y, _, _ = next(iter(train_loader))
    batch_x = batch_x[:2].float().to(device)
    target = batch_y[:2, -args.pred_len :, :].float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_for_batch(model, batch_x, target, experiment)
    if not torch.isfinite(loss):
        raise FloatingPointError("smoke-test loss is non-finite")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(grad).all().item() for grad in gradients):
        raise RuntimeError("smoke test did not produce finite gradients")
    if experiment == "E1":
        model.eval()
        with torch.no_grad():
            output = model(batch_x)
        if output.shape != (2, args.pred_len, args.enc_in, 3):
            raise AssertionError("quantile forecast shape is incorrect: {}".format(output.shape))
        if not torch.all(output[..., 0] <= output[..., 1]):
            raise AssertionError("lower quantile exceeds median")
        if not torch.all(output[..., 1] <= output[..., 2]):
            raise AssertionError("median exceeds upper quantile")
    else:
        output = model(batch_x)
        if output.shape != (2, args.pred_len, args.enc_in):
            raise AssertionError("baseline forecast shape is incorrect: {}".format(output.shape))
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint = Path(temporary_directory) / "smoke_checkpoint.pth"
        torch.save(model.state_dict(), str(checkpoint))
        model.load_state_dict(torch.load(str(checkpoint), map_location=device))
    print(
        "Smoke test passed: experiment={}, input={}, output={}, loss={:.6f}, checkpoint=round-trip, device={}".format(
            experiment, tuple(batch_x.shape), tuple(output.shape), float(loss.item()), device
        )
    )
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def save_interval_plot(
    targets: np.ndarray,
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    channel: int,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = sorted(set((0, len(targets) // 2, len(targets) - 1)))
    figure, axes = plt.subplots(len(selected), 1, figsize=(12, 3 * len(selected)))
    axes = np.atleast_1d(axes)
    horizon = np.arange(targets.shape[1])
    for axis, index in zip(axes, selected):
        axis.plot(horizon, targets[index, :, channel], label="actual", linewidth=1.5)
        axis.plot(horizon, median[index, :, channel], label="median", linewidth=1.5)
        axis.fill_between(
            horizon,
            lower[index, :, channel],
            upper[index, :, channel],
            alpha=0.25,
            label="95% nominal interval",
        )
        axis.set_title("Test forecast window {}".format(index))
        axis.set_xlabel("Forecast step")
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(str(path), dpi=150)
    plt.close(figure)


def train_and_evaluate(
    args: SimpleNamespace,
    experiment: str,
    datasets: Dict[str, Dataset_ETTh],
    loaders: Dict[str, DataLoader],
    device: torch.device,
) -> None:
    seed_everything(args.random_seed)
    model = build_model(experiment, args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    checkpoint_dir = (
        resolve_path(args.checkpoint_dir)
        / experiment
        / "{}_h{}_s{}".format(args.data, args.pred_len, args.random_seed)
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "best_model.pth"

    best_loss = math.inf
    patience_count = 0
    epoch_times = []
    if args.eval_only:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "evaluation-only mode requires an existing checkpoint: {}".format(
                    checkpoint
                )
            )
        model.load_state_dict(torch.load(str(checkpoint), map_location=device))
        best_loss = evaluate_loss(model, loaders["val"], args, experiment, device)
    else:
        for epoch in range(1, args.train_epochs + 1):
            start = time.perf_counter()
            train_loss = train_one_epoch(
                model, loaders["train"], optimizer, args, experiment, device
            )
            validation_loss = evaluate_loss(
                model, loaders["val"], args, experiment, device
            )
            elapsed = time.perf_counter() - start
            epoch_times.append(elapsed)
            print(
                "Epoch {}/{} train_loss={:.6f} validation_loss={:.6f} elapsed={:.1f}s".format(
                    epoch, args.train_epochs, train_loss, validation_loss, elapsed
                )
            )

            if validation_loss < best_loss:
                best_loss = validation_loss
                patience_count = 0
                torch.save(model.state_dict(), str(checkpoint))
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    print("Early stopping")
                    break
            adjust_learning_rate(optimizer, None, epoch, args, printout=False)

    if not checkpoint.is_file():
        raise FileNotFoundError("best checkpoint was not saved: {}".format(checkpoint))
    if not args.eval_only:
        model.load_state_dict(torch.load(str(checkpoint), map_location=device))
    scaler = datasets["val"].scaler
    eval_work_dir = Path(tempfile.mkdtemp(prefix="interval_eval_"))
    calibrators = {}
    if experiment == "E1":
        calibration_predictions, calibration_targets_scaled = collect_forecasts(
            model, loaders["calibration"], args, experiment, device,
            eval_work_dir, "calibration",
        )
        calibration_targets = inverse_scale_memmap(
            calibration_targets_scaled, scaler, eval_work_dir / "cal_targets_original.npy"
        )
        calibration_lower = inverse_scale_memmap(
            calibration_predictions[..., 0], scaler, eval_work_dir / "cal_lower.npy"
        )
        calibration_upper = inverse_scale_memmap(
            calibration_predictions[..., 2], scaler, eval_work_dir / "cal_upper.npy"
        )
        for method in CALIBRATION_METHODS:
            calibrators[method] = fit_cqr_calibration(
                calibration_lower,
                calibration_upper,
                calibration_targets,
                alpha=1.0 - args.nominal_coverage,
                method=method,
                min_calibration_windows=args.min_calibration_windows,
            )
        del (
            calibration_predictions,
            calibration_targets_scaled,
            calibration_targets,
            calibration_lower,
            calibration_upper,
        )

    predictions, scaled_targets = collect_forecasts(
        model, loaders["test"], args, experiment, device,
        eval_work_dir, "test",
    )
    targets_original = inverse_scale_memmap(
        scaled_targets, scaler, eval_work_dir / "test_targets_original.npy"
    )
    calibration_reports = {}
    if experiment == "E1":
        lower = inverse_scale_memmap(
            predictions[..., 0], scaler, eval_work_dir / "test_lower.npy"
        )
        median = inverse_scale_memmap(
            predictions[..., 1], scaler, eval_work_dir / "test_median.npy"
        )
        upper = inverse_scale_memmap(
            predictions[..., 2], scaler, eval_work_dir / "test_upper.npy"
        )
        metrics = compute_forecast_metrics(
            targets_original,
            median,
            lower=lower,
            upper=upper,
            nominal_coverage=args.nominal_coverage,
        )
        primary_calibrated_lower = None
        primary_calibrated_upper = None
        for method in CALIBRATION_METHODS:
            calibrated_lower, calibrated_upper = calibrate_memmap_chunks(
                lower, upper, median, calibrators[method],
                eval_work_dir / ("test_cal_lower_{}.npy".format(method)),
                eval_work_dir / ("test_cal_upper_{}.npy".format(method)),
            )
            calibration_reports[method] = {
                "calibration": calibrators[method],
                "metrics_original_units": compute_forecast_metrics(
                    targets_original,
                    median,
                    lower=calibrated_lower,
                    upper=calibrated_upper,
                    nominal_coverage=args.nominal_coverage,
                ),
            }
            if method == args.primary_calibration_method:
                primary_calibrated_lower = calibrated_lower
                primary_calibrated_upper = calibrated_upper
            else:
                del calibrated_lower, calibrated_upper
        if primary_calibrated_lower is None or primary_calibrated_upper is None:
            raise RuntimeError("primary calibration method did not produce bounds")
        quantile_values = {
            "lower": QUANTILE_LEVELS[0],
            "median": QUANTILE_LEVELS[1],
            "upper": QUANTILE_LEVELS[2],
        }
    else:
        median = inverse_scale_memmap(
            predictions, scaler, eval_work_dir / "test_median.npy"
        )
        metrics = compute_forecast_metrics(targets_original, median)
        lower = upper = None
        quantile_values = None

    result_dir = (
        resolve_path(args.result_dir)
        / experiment
        / "{}_h{}_s{}".format(args.data, args.pred_len, args.random_seed)
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": experiment,
        "dataset": args.data,
        "prediction_horizon": args.pred_len,
        "random_seed": args.random_seed,
        "best_validation_loss": float(best_loss),
        "epochs_completed": len(epoch_times),
        "mean_epoch_seconds": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "evaluation_only": bool(args.eval_only),
        "batch_size": args.batch_size,
        "metrics_original_units": metrics,
        "quantiles": quantile_values,
        "checkpoint": str(checkpoint),
    }
    if experiment == "E1":
        report["validation_calibration_split"] = loaders["split_info"]
        report["calibration_methods"] = {
            method: {
                "calibration": details["calibration"],
                "metrics_original_units": details["metrics_original_units"],
            }
            for method, details in calibration_reports.items()
        }
    with open(result_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    arrays = {
        "targets": targets_original,
        "median": median,
    }
    if lower is not None and upper is not None:
        arrays["lower"] = lower
        arrays["upper"] = upper
        primary_method = args.primary_calibration_method
        if primary_method not in calibrators:
            raise ValueError(
                "primary calibration method is not available: {}".format(primary_method)
            )
        arrays["calibrated_lower"] = primary_calibrated_lower
        arrays["calibrated_upper"] = primary_calibrated_upper
    _save_forecasts_npz(result_dir / "test_forecasts.npz", **arrays)
    if lower is not None and upper is not None:
        save_interval_plot(
            targets_original,
            lower,
            median,
            upper,
            channel=args.enc_in - 1,
            path=result_dir / "interval_examples_ot.png",
        )
        primary_method = args.primary_calibration_method
        calibrated_lower = primary_calibrated_lower
        calibrated_upper = primary_calibrated_upper
        save_interval_plot(
            targets_original,
            calibrated_lower,
            median,
            calibrated_upper,
            channel=args.enc_in - 1,
            path=result_dir / "interval_examples_ot_calibrated.png",
        )
        calibrated_report = {
            "experiment": "E1-C",
            "dataset": args.data,
            "prediction_horizon": args.pred_len,
            "random_seed": args.random_seed,
            "calibration_method": primary_method,
            "calibration": calibration_reports[primary_method]["calibration"],
            "validation_calibration_split": loaders["split_info"],
            "metrics_original_units": calibration_reports[primary_method][
                "metrics_original_units"
            ],
            "checkpoint": str(checkpoint),
        }
        calibrated_dir = (
            resolve_path(args.result_dir)
            / "E1-C"
            / "{}_h{}_s{}".format(args.data, args.pred_len, args.random_seed)
        )
        calibrated_dir.mkdir(parents=True, exist_ok=True)
        with open(calibrated_dir / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(calibrated_report, file, indent=2, allow_nan=False)
        _save_forecasts_npz(
            calibrated_dir / "test_forecasts.npz",
            targets=targets_original,
            median=median,
            lower=calibrated_lower,
            upper=calibrated_upper,
        )
        with open(result_dir / "calibration.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "calibration_split": loaders["split_info"],
                    "primary_method": primary_method,
                    "methods": {
                        method: details["calibration"]
                        for method, details in calibration_reports.items()
                    },
                },
                file,
                indent=2,
                allow_nan=False,
            )
    print(json.dumps(report, indent=2, allow_nan=False))
    memmaps = [predictions, scaled_targets, targets_original, median]
    if experiment == "E1":
        memmaps.extend(
            [
                lower,
                upper,
                primary_calibrated_lower,
                primary_calibrated_upper,
                calibrated_lower,
                calibrated_upper,
            ]
        )
    for array in memmaps:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()
    del memmaps, arrays
    shutil.rmtree(eval_work_dir, ignore_errors=True)


def main() -> None:
    cli = parse_args()
    config = load_config(cli.config, cli.dataset)
    config.setdefault("pred_len", 96)
    config.setdefault("random_seed", 2021)
    if cli.pred_len is not None:
        config["pred_len"] = cli.pred_len
    if cli.epochs is not None:
        config["train_epochs"] = cli.epochs
    if cli.batch_size is not None:
        config["batch_size"] = cli.batch_size
    if cli.seed is not None:
        config["random_seed"] = cli.seed
    config["eval_only"] = cli.eval_only

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = build_args(config, device)
    if args.pred_len not in (96, 192, 336, 720):
        raise ValueError("pred_len must be one of 96, 192, 336, or 720")
    seed_everything(args.random_seed)

    datasets = {}
    loaders = {}
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        datasets[split], split_loader = make_dataset_and_loader(args, split, shuffle)
        if split != "val":
            loaders[split] = split_loader

    monitor_indices, calibration_indices, split_info = split_validation_indices(
        len(datasets["val"]),
        args.pred_len,
        monitor_fraction=args.monitor_fraction,
        min_calibration_windows=args.min_calibration_windows,
    )
    loaders["val"] = DataLoader(
        Subset(datasets["val"], monitor_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.device.startswith("cuda"),
    )
    loaders["calibration"] = DataLoader(
        Subset(datasets["val"], calibration_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.device.startswith("cuda"),
    )
    loaders["split_info"] = split_info
    print(
        "validation monitor={} gap={} calibration={}".format(
            split_info["monitor_windows"],
            split_info["excluded_gap_windows"],
            split_info["calibration_windows"],
        )
    )

    if cli.smoke_test:
        run_smoke_test(args, cli.experiment, loaders["train"], device)
        return
    train_and_evaluate(args, cli.experiment, datasets, loaders, device)


if __name__ == "__main__":
    main()
