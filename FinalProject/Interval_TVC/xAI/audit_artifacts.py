"""
audit_artifacts.py – Scan all Interval-TVC result slots and produce
  xAI/results/xai_run_inventory.csv   : one row per (variant, dataset, pred_len, seed)
  xAI/results/xai_audit.json          : summary counts

Run from repo root:
  python FinalProject/Interval_TVC/xAI/audit_artifacts.py
"""
import json, os, sys
from pathlib import Path
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).resolve().parent
ITVC_DIR   = THIS_DIR.parent
RESULTS_V1 = ITVC_DIR / "results" / "full_3seeds_v1"
CKPT_ROOT  = ITVC_DIR / "checkpoints" / "full_3seeds_v1"
OUT_DIR    = THIS_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Inventory config ───────────────────────────────────────────────────────────
DATASETS   = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
HORIZONS   = [96, 192, 336, 720]
SEEDS      = [2021, 2023, 2025]
VARIANTS   = ["E0", "E1", "E1-C"]

# Known missing: Weather H720 all seeds; Weather H96/192/336 seeds 2023 & 2025
KNOWN_MISSING = set()
for seed in [2023, 2025]:
    for h in [96, 192, 336]:
        KNOWN_MISSING.add(("E1",   "Weather", h, seed))
        KNOWN_MISSING.add(("E1-C", "Weather", h, seed))
        KNOWN_MISSING.add(("E0",   "Weather", h, seed))
for seed in SEEDS:
    KNOWN_MISSING.add(("E1",   "Weather", 720, seed))
    KNOWN_MISSING.add(("E1-C", "Weather", 720, seed))
    KNOWN_MISSING.add(("E0",   "Weather", 720, seed))


def run_dir(variant: str, dataset: str, h: int, seed: int) -> Path:
    name = f"{dataset}_h{h}_s{seed}"
    return RESULTS_V1 / variant / name


def ckpt_dir(variant: str, dataset: str, h: int, seed: int) -> Path:
    name = f"{dataset}_h{h}_s{seed}"
    return CKPT_ROOT / variant / name


def audit_slot(variant, dataset, h, seed):
    rdir = run_dir(variant, dataset, h, seed)
    cdir = ckpt_dir(variant, dataset, h, seed)
    has_metrics      = (rdir / "metrics.json").is_file()
    has_npz          = (rdir / "test_forecasts.npz").is_file()
    has_calib_json   = (rdir / "calibration.json").is_file()
    ckpt_exists      = any(cdir.glob("*.pth")) if cdir.is_dir() else False

    expected = (variant, dataset, h, seed) not in KNOWN_MISSING
    if has_metrics and has_npz:
        status = "complete"
    elif expected:
        status = "missing"
    else:
        status = "expected_missing"

    return {
        "variant":          variant,
        "dataset":          dataset,
        "pred_len":         h,
        "seed":             seed,
        "has_metrics_json": has_metrics,
        "has_npz":          has_npz,
        "has_calib_json":   has_calib_json,
        "ckpt_exists":      ckpt_exists,
        "status":           status,
    }


rows = []
for v in VARIANTS:
    for ds in DATASETS:
        for h in HORIZONS:
            for s in SEEDS:
                rows.append(audit_slot(v, ds, h, s))

df = pd.DataFrame(rows)
inv_path = OUT_DIR / "xai_run_inventory.csv"
df.to_csv(inv_path, index=False)
print(f"Inventory written: {inv_path}")

# ── Summary ────────────────────────────────────────────────────────────────────
summary = {}
for v in VARIANTS:
    sub = df[df.variant == v]
    summary[v] = {
        "complete":          int((sub.status == "complete").sum()),
        "missing":           int((sub.status == "missing").sum()),
        "expected_missing":  int((sub.status == "expected_missing").sum()),
        "total_slots":       len(sub),
    }
summary["overall"] = {
    "total_rows": len(df),
    "complete":   int((df.status == "complete").sum()),
}

audit_path = OUT_DIR / "xai_audit.json"
with open(audit_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Audit JSON written: {audit_path}")
print(json.dumps(summary, indent=2))
