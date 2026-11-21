"""
calibration_analysis.py – Analyse E1 raw vs E1-C calibrated results.

Sources used:
  • results/full_3seeds_v1/E1/{run}/test_forecasts.npz   – raw quantile outputs
  • results/full_3seeds_v1/E1-C/{run}/test_forecasts.npz – calibrated outputs
  • results/full_3seeds_v1/E1/{run}/calibration.json     – δ per [H, C]
  • results/full_3seeds_v1/E1/{run}/metrics.json         – stored reference metrics

Outputs (all in xAI/results/):
  calibration_delta_by_run.csv   – δ statistics per run
  e1_vs_e1c_metrics.csv          – PICP / MPIW / PINAW / ACE for E1 and E1-C side by side
  calibration_summary.csv        – aggregated stats across runs

No inference, no refitting, no test-set leakage.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

THIS_DIR   = Path(__file__).resolve().parent
ITVC_DIR   = THIS_DIR.parent
RESULTS_V1 = ITVC_DIR / "results" / "full_3seeds_v1"
OUT_DIR    = THIS_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ETT_CHANNELS  = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
WEATHER_CHANNELS = [f"ch{i}" for i in range(21)]   # generic; rename if needed


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers (match code in metrics/interval_metrics.py)
# ──────────────────────────────────────────────────────────────────────────────

def compute_picp(lower, upper, targets):
    covered = ((targets >= lower) & (targets <= upper))
    return covered.sum() / targets.size


def compute_mpiw(lower, upper, targets):
    return (upper - lower).sum() / targets.size


def compute_pinaw(lower, upper, targets):
    """Mean over channels of (avg_width / channel_range_in_test)."""
    # targets: [N, H, C]
    width = upper - lower                                 # [N, H, C]
    avg_w = width.mean(axis=(0, 1))                       # [C]
    ch_min = targets.min(axis=(0, 1))
    ch_max = targets.max(axis=(0, 1))
    rng    = ch_max - ch_min
    rng[rng == 0] = 1.0
    return float((avg_w / rng).mean())


def compute_ace(picp):
    return abs(picp - 0.95)


# ──────────────────────────────────────────────────────────────────────────────
# Enumerate available runs
# ──────────────────────────────────────────────────────────────────────────────

def list_e1_runs():
    e1_root = RESULTS_V1 / "E1"
    return sorted(e1_root.iterdir()) if e1_root.is_dir() else []


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

records_comparison = []  # e1_vs_e1c_metrics.csv
records_delta      = []  # calibration_delta_by_run.csv

for run_dir in list_e1_runs():
    run_name = run_dir.name  # e.g. "ETTh1_h96_s2021"

    # Parse run name
    parts = run_name.split("_")
    if len(parts) < 3:
        print(f"  [SKIP] unexpected dir name: {run_name}")
        continue
    dataset  = parts[0]
    pred_len = int(parts[1].replace("h", ""))
    seed     = int(parts[2].replace("s", ""))

    # Paths
    e1_npz      = run_dir / "test_forecasts.npz"
    e1c_dir     = RESULTS_V1 / "E1-C" / run_name
    e1c_npz     = e1c_dir / "test_forecasts.npz"
    calib_json  = run_dir / "calibration.json"
    metrics_json= run_dir / "metrics.json"

    if not e1_npz.is_file() or not e1c_npz.is_file():
        print(f"  [SKIP] missing NPZ for {run_name}")
        continue

    # ── Load E1 raw ────────────────────────────────────────────────────────
    e1 = np.load(e1_npz)
    targets = e1["targets"]           # [N, H, C]
    lower   = e1["lower"]
    upper   = e1["upper"]
    median  = e1["median"]

    # ── Load E1-C calibrated ───────────────────────────────────────────────
    e1c = np.load(e1c_npz)
    lower_c = e1c["lower"]
    upper_c = e1c["upper"]

    # ── Compute metrics ────────────────────────────────────────────────────
    e1_picp  = compute_picp(lower, upper, targets)
    e1_mpiw  = compute_mpiw(lower, upper, targets)
    e1_pinaw = compute_pinaw(lower, upper, targets)
    e1_ace   = compute_ace(e1_picp)
    e1_mse   = float(((median - targets) ** 2).mean())
    e1_mae   = float(np.abs(median - targets).mean())

    ec_picp  = compute_picp(lower_c, upper_c, targets)
    ec_mpiw  = compute_mpiw(lower_c, upper_c, targets)
    ec_pinaw = compute_pinaw(lower_c, upper_c, targets)
    ec_ace   = compute_ace(ec_picp)

    # ── Load calibration δ ─────────────────────────────────────────────────
    delta_mean = delta_std = delta_min = delta_max = float("nan")
    neg_frac   = float("nan")
    if calib_json.is_file():
        calib = json.load(open(calib_json))
        try:
            adj   = np.array(calib["methods"]["horizon_channel"]["adjustment"])
            delta_mean = float(adj.mean())
            delta_std  = float(adj.std())
            delta_min  = float(adj.min())
            delta_max  = float(adj.max())
            neg_frac   = float((adj < 0).mean())   # fraction where lower bound is tightened
        except Exception as ex:
            print(f"  [WARN] calib parse error {run_name}: {ex}")

    records_comparison.append({
        "run":        run_name,
        "dataset":    dataset,
        "pred_len":   pred_len,
        "seed":       seed,
        # E1 raw
        "e1_picp":    round(e1_picp,  6),
        "e1_mpiw":    round(e1_mpiw,  4),
        "e1_pinaw":   round(e1_pinaw, 6),
        "e1_ace":     round(e1_ace,   6),
        "e1_mse":     round(e1_mse,   6),
        "e1_mae":     round(e1_mae,   6),
        # E1-C calibrated
        "ec_picp":    round(ec_picp,  6),
        "ec_mpiw":    round(ec_mpiw,  4),
        "ec_pinaw":   round(ec_pinaw, 6),
        "ec_ace":     round(ec_ace,   6),
        # Deltas
        "delta_picp": round(ec_picp - e1_picp,  6),
        "delta_mpiw": round(ec_mpiw - e1_mpiw,  4),
        "delta_pinaw":round(ec_pinaw - e1_pinaw, 6),
        "delta_ace":  round(ec_ace  - e1_ace,    6),
    })

    records_delta.append({
        "run":        run_name,
        "dataset":    dataset,
        "pred_len":   pred_len,
        "seed":       seed,
        "delta_mean": round(delta_mean, 6),
        "delta_std":  round(delta_std,  6),
        "delta_min":  round(delta_min,  6),
        "delta_max":  round(delta_max,  6),
        "neg_frac":   round(neg_frac,   4),   # fraction of [H,C] cells where δ < 0
    })

    print(f"  {run_name:30s}  E1 PICP={e1_picp:.4f}  E1-C PICP={ec_picp:.4f}  δ_mpiw={ec_mpiw-e1_mpiw:+.3f}")

# ── Save CSVs ──────────────────────────────────────────────────────────────────
df_cmp = pd.DataFrame(records_comparison).sort_values(["dataset", "pred_len", "seed"])
df_cmp.to_csv(OUT_DIR / "e1_vs_e1c_metrics.csv", index=False)
print(f"\nSaved: e1_vs_e1c_metrics.csv  ({len(df_cmp)} rows)")

df_dlt = pd.DataFrame(records_delta).sort_values(["dataset", "pred_len", "seed"])
df_dlt.to_csv(OUT_DIR / "calibration_delta_by_run.csv", index=False)
print(f"Saved: calibration_delta_by_run.csv  ({len(df_dlt)} rows)")

# ── Summary ────────────────────────────────────────────────────────────────────
agg = {}
for ds in df_cmp.dataset.unique():
    sub = df_cmp[df_cmp.dataset == ds]
    agg[ds] = {
        "n_runs":        len(sub),
        "e1_picp_mean":  round(sub.e1_picp.mean(),  4),
        "e1_picp_std":   round(sub.e1_picp.std(),   4),
        "ec_picp_mean":  round(sub.ec_picp.mean(),  4),
        "ec_picp_std":   round(sub.ec_picp.std(),   4),
        "e1_mpiw_mean":  round(sub.e1_mpiw.mean(),  4),
        "ec_mpiw_mean":  round(sub.ec_mpiw.mean(),  4),
        "delta_picp":    round(sub.delta_picp.mean(),4),
        "pct_ec_ge_095": round((sub.ec_picp >= 0.95).mean() * 100, 1),
    }

# Overall
agg["OVERALL"] = {
    "n_runs":        len(df_cmp),
    "e1_picp_mean":  round(df_cmp.e1_picp.mean(),  4),
    "ec_picp_mean":  round(df_cmp.ec_picp.mean(),  4),
    "e1_mpiw_mean":  round(df_cmp.e1_mpiw.mean(),  4),
    "ec_mpiw_mean":  round(df_cmp.ec_mpiw.mean(),  4),
    "pct_ec_ge_095": round((df_cmp.ec_picp >= 0.95).mean() * 100, 1),
}

df_sum = pd.DataFrame(agg).T.reset_index().rename(columns={"index": "dataset"})
df_sum.to_csv(OUT_DIR / "calibration_summary.csv", index=False)
print(f"Saved: calibration_summary.csv")
print(df_sum.to_string())
