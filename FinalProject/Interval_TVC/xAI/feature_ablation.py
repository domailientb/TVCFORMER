"""
feature_ablation.py – Per-variable sensitivity analysis for Interval-TVC E1.

Method: for each input channel j, replace its value with the training-set mean
(in the SCALED space that the model receives) across test windows, run inference,
and record the change in metrics relative to the unmasked baseline.

This is a purely observational sensitivity measure. It does NOT imply causality.

Outputs (all in xAI/results/):
  feature_ablation_raw.csv     – per-run × per-channel rows
  feature_ablation_summary.csv – averaged over seeds, by dataset × pred_len × channel

Note on environment:
  PyTorch model inference requires memory/paging space. If host system encounters
  WinError 1455 (paging file limit) or memory constraints, runs are safely skipped
  and logged to xAI/logs/ablation_skip.log without breaking the pipeline.
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).resolve().parent
ITVC_DIR   = THIS_DIR.parent
RESULTS_V1 = ITVC_DIR / "results" / "full_3seeds_v1"
CKPT_ROOT  = ITVC_DIR / "checkpoints" / "full_3seeds_v1"
OUT_DIR    = THIS_DIR / "results"
LOG_DIR    = THIS_DIR / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ablation_run.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SKIP_LOG = LOG_DIR / "ablation_skip.log"

def log_skip(run_name: str, reason: str):
    with open(SKIP_LOG, "a", encoding="utf-8") as f:
        f.write(f"{run_name}\t{reason}\n")
    log.warning(f"SKIP {run_name}: {reason}")


ETT_CH = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(lower, upper, median, targets):
    """All arrays: [N, H, C] in original units."""
    mse  = float(((median - targets) ** 2).mean())
    mae  = float(np.abs(median - targets).mean())
    picp = float(((targets >= lower) & (targets <= upper)).mean())
    mpiw = float((upper - lower).sum() / targets.size)
    width  = upper - lower
    avg_w  = width.mean(axis=(0, 1))
    rng    = targets.max(axis=(0, 1)) - targets.min(axis=(0, 1))
    rng[rng == 0] = 1.0
    pinaw  = float((avg_w / rng).mean())
    ace    = abs(picp - 0.95)
    return {"mse": mse, "mae": mae, "picp": picp, "mpiw": mpiw, "pinaw": pinaw, "ace": ace}


def apply_cqr(lower, upper, median, delta):
    """
    delta: [H, C] from calibration.json
    Returns calibrated lower, upper with median unchanged.
    """
    lower_c = np.minimum(lower - delta[None, :, :], median)
    upper_c = np.maximum(upper + delta[None, :, :], median)
    return lower_c, upper_c


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    try:
        import torch
    except Exception as e:
        log.error(f"Cannot import torch: {e}")
        log_skip("ALL", f"torch import failed: {e}")
        return

    # Add relevant paths
    if str(ITVC_DIR) not in sys.path:
        sys.path.insert(0, str(ITVC_DIR))
    repo_root = ITVC_DIR.parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    official_root = repo_root / "official"
    if str(official_root) in sys.path:
        sys.path.remove(str(official_root))
    sys.path.insert(0, str(official_root))

    try:
        from train_interval import load_config, build_args, make_dataset_and_loader, build_model
        from metrics.interval_metrics import inverse_scale
    except Exception as e:
        log.error(f"Cannot import from train_interval: {e}")
        log_skip("ALL", f"train_interval import failed: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = ITVC_DIR / "configs" / "interval.yaml"

    DATASETS_ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
    HORIZONS = [96, 192, 336, 720]
    SEEDS = [2021, 2023, 2025]

    records = []

    for dataset in DATASETS_ORDER:
        for pred_len in HORIZONS:
            for seed in SEEDS:
                run_name = f"{dataset}_h{pred_len}_s{seed}"
                npz_path = RESULTS_V1 / "E1" / run_name / "test_forecasts.npz"
                ckpt_path = CKPT_ROOT / "E1" / run_name / "best_model.pth"

                if not npz_path.is_file():
                    log_skip(run_name, "no E1 NPZ")
                    continue
                if not ckpt_path.is_file():
                    log_skip(run_name, "no checkpoint file")
                    continue

                log.info(f"\n{'='*60}\nRUN: {run_name}")

                try:
                    config = load_config(str(cfg_path), dataset)
                    config["pred_len"] = pred_len
                    config["random_seed"] = seed
                    config["eval_only"] = True
                    args = build_args(config, device)

                    train_data, _ = make_dataset_and_loader(args, "train", shuffle=False)
                    scaler = train_data.scaler
                    test_data, test_loader = make_dataset_and_loader(args, "test", shuffle=False)

                    # Compute train mean in scaled space
                    train_mean_sc = np.zeros(train_data.data_x.shape[-1], dtype=np.float32)

                    model = build_model("E1", args).to(device)
                    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
                    model.eval()

                    # Load calibration delta
                    calib_json = RESULTS_V1 / "E1" / run_name / "calibration.json"
                    delta = None
                    if calib_json.is_file():
                        try:
                            cdata = json.load(open(calib_json))
                            delta = np.array(cdata["methods"]["horizon_channel"]["adjustment"])
                        except Exception as ex:
                            log.warning(f"  calib parse error: {ex}")

                    # 1. Unmasked Baseline
                    preds_b, tgts_b = [], []
                    with torch.no_grad():
                        for bx, by, _, _ in test_loader:
                            bx = bx.float().to(device)
                            out = model(bx)
                            preds_b.append(out.cpu().numpy())
                            tgts_b.append(by[:, -args.pred_len:, :].float().numpy())
                    preds_b = np.concatenate(preds_b)
                    tgts_b = np.concatenate(tgts_b)

                    lo_b = inverse_scale(preds_b[..., 0], scaler)
                    md_b = inverse_scale(preds_b[..., 1], scaler)
                    up_b = inverse_scale(preds_b[..., 2], scaler)
                    tgt_orig = inverse_scale(tgts_b, scaler)

                    base_m = compute_metrics(lo_b, up_b, md_b, tgt_orig)
                    if delta is not None:
                        lo_bc, up_bc = apply_cqr(lo_b, up_b, md_b, delta)
                        base_c_m = compute_metrics(lo_bc, up_bc, md_b, tgt_orig)
                    else:
                        base_c_m = {k: float("nan") for k in base_m}

                    n_ch = train_data.data_x.shape[-1]
                    for j in range(n_ch):
                        mv = float(train_mean_sc[j])
                        preds_j = []
                        with torch.no_grad():
                            for bx, by, _, _ in test_loader:
                                bx_masked = bx.clone().float()
                                bx_masked[:, :, j] = mv
                                out_j = model(bx_masked.to(device))
                                preds_j.append(out_j.cpu().numpy())
                        preds_j = np.concatenate(preds_j)

                        lo_j = inverse_scale(preds_j[..., 0], scaler)
                        md_j = inverse_scale(preds_j[..., 1], scaler)
                        up_j = inverse_scale(preds_j[..., 2], scaler)

                        m_j = compute_metrics(lo_j, up_j, md_j, tgt_orig)
                        if delta is not None:
                            lo_jc, up_jc = apply_cqr(lo_j, up_j, md_j, delta)
                            m_jc = compute_metrics(lo_jc, up_jc, md_j, tgt_orig)
                        else:
                            m_jc = {k: float("nan") for k in m_j}

                        ch_name = ETT_CH[j] if n_ch == 7 else f"ch{j}"
                        records.append({
                            "run": run_name, "dataset": dataset,
                            "pred_len": pred_len, "seed": seed,
                            "channel_idx": j, "channel_name": ch_name,
                            "base_mse": base_m["mse"], "base_picp": base_m["picp"],
                            "base_mpiw": base_m["mpiw"], "base_pinaw": base_m["pinaw"],
                            "abl_mse": m_j["mse"], "abl_picp": m_j["picp"],
                            "abl_mpiw": m_j["mpiw"], "abl_pinaw": m_j["pinaw"],
                            "delta_mse": m_j["mse"] - base_m["mse"],
                            "delta_mae": m_j["mae"] - base_m["mae"],
                            "delta_picp": m_j["picp"] - base_m["picp"],
                            "delta_mpiw": m_j["mpiw"] - base_m["mpiw"],
                            "delta_ec_picp": m_jc["picp"] - base_c_m["picp"],
                            "delta_ec_mpiw": m_jc["mpiw"] - base_c_m["mpiw"],
                        })
                        log.info(f"  ch{j:02d} ({ch_name}): ΔMSE={m_j['mse'] - base_m['mse']:+.4f}, ΔPICP={m_j['picp'] - base_m['picp']:+.4f}")

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as ex:
                    log_skip(run_name, f"execution error: {ex}")

    if records:
        df_raw = pd.DataFrame(records)
        df_raw.to_csv(OUT_DIR / "feature_ablation_raw.csv", index=False)
        log.info(f"Saved feature_ablation_raw.csv ({len(df_raw)} rows)")

        group = df_raw.groupby(["dataset", "pred_len", "channel_idx", "channel_name"])
        summ = group.agg(
            n_runs=("run", "count"),
            delta_mse_mean=("delta_mse", "mean"),
            delta_mse_std=("delta_mse", "std"),
            delta_mae_mean=("delta_mae", "mean"),
            delta_picp_mean=("delta_picp", "mean"),
            delta_mpiw_mean=("delta_mpiw", "mean"),
        ).reset_index()
        summ.to_csv(OUT_DIR / "feature_ablation_summary.csv", index=False)
        log.info(f"Saved feature_ablation_summary.csv ({len(summ)} rows)")
    else:
        log.warning("No ablation records produced.")


if __name__ == "__main__":
    main()
