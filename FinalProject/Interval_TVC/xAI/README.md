# XAI Module – Interval-TVC

> **Scope**: Explainability analysis for the **Interval-TVC** improvements to TVCFormer:
> Quantile Head (E1) and CQR Calibration (E1-C). This module does NOT repeat the
> TVCFormer core XAI (TVC Graph, GraphSAGE); it focuses on **prediction interval quality**,
> **calibration mechanics**, and **input sensitivity of the quantile head**.

---

## 1. Project Architecture

```
FinalProject/Interval_TVC/
├── models/interval_tvcformer.py    ← Main model (QuantileFlattenHead + RevIN)
├── calibration/conformal.py        ← CQR calibration logic
├── train_interval.py               ← Training / evaluation entrypoint
├── metrics/interval_metrics.py     ← PICP, MPIW, PINAW, ACE definitions
├── configs/interval.yaml           ← All dataset configs (5 datasets)
├── results/full_3seeds_v1/         ← All experiment artifacts (DO NOT MODIFY)
│   ├── E0/{run}/                   ← Point-forecast baseline (no quantile head)
│   ├── E1/{run}/                   ← Quantile head, raw (no calibration)
│   └── E1-C/{run}/                 ← E1 + CQR calibration applied
├── checkpoints/full_3seeds_v1/     ← Model weights (.pth files)
└── xAI/                            ← ← ← THIS MODULE
    ├── README.md                   (this file)
    ├── run_xai.py                  ← Pipeline orchestrator
    ├── audit_artifacts.py          ← Scan all 180 slots; write inventory CSV
    ├── calibration_analysis.py     ← E1 vs E1-C metrics; δ statistics
    ├── plot_xai.py                 ← All figures
    ├── feature_ablation.py         ← Per-channel sensitivity (requires inference)
    ├── requirements_notes.txt      ← Dependency notes
    ├── results/                    ← Generated CSVs and JSON
    ├── figures/                    ← Generated PNG figures
    └── logs/                       ← Run logs and skip records
```

---

## 2. E0 / E1 / E1-C Differences

| Variant | Head        | Calibration | Output per sample          |
|---------|-------------|-------------|----------------------------|
| E0      | FlattenHead (point) | None | median prediction only  |
| E1      | QuantileFlattenHead | None | lower (q0.025), median (q0.5), upper (q0.975) |
| E1-C    | QuantileFlattenHead | CQR (applied post-hoc from stored δ) | calibrated lower/upper; **median unchanged** |

**Key fact**: E1-C is NOT a separately trained model. It applies a stored adjustment δ
(fitted on the validation calibration split) to E1's raw quantile outputs at test time.
No new training or test-set leakage occurs.

### QuantileFlattenHead (`models/interval_tvcformer.py`)
```python
# Output: [batch, horizon, channel, 3]
#   dim=-1 indices: 0=lower, 1=median, 2=upper
# Ordering enforced via softplus gaps:
#   lower_gap  = softplus(raw_lower_gap)   → lower  = median - lower_gap
#   upper_gap  = softplus(raw_upper_gap)   → upper  = median + upper_gap
# This guarantees lower < median < upper structurally.
```

### CQR Calibration (`calibration/conformal.py`)
- **Score**: `s = max(lower - y, y - upper)` per sample × horizon × channel
- **Method**: `horizon_channel` → δ shape `[pred_len, C]`
- **Adjustment**: `calibrated_lower = min(lower − δ, median)`, `calibrated_upper = max(upper + δ, median)`
- **δ saved to**: `results/full_3seeds_v1/E1/{run}/calibration.json`
  → key path: `methods → horizon_channel → adjustment`

---

## 3. Data Pipeline

### Input tensor shape
- `x`: `[batch, seq_len=336, channels]` (StandardScaler on training data)
- Inside model: `x.permute(0,2,1)` → `[batch, channels, seq_len]` → TVCFormer backbone
- RevIN applied before TVCFormer, denormed after QuantileFlattenHead
- **Model output is in original units** (after RevIN denorm)

### Channels per dataset
| Dataset | Channels (C) | Channel names |
|---------|-------------|---------------|
| ETTh1, ETTh2, ETTm1, ETTm2 | 7 | HUFL, HULL, MUFL, MULL, LUFL, LULL, OT |
| Weather | 21 | ch0 … ch20 (OT = last channel) |

Target channel: **OT** (last channel, index 6 for ETT, index 20 for Weather)

### Train/Val/Test splits
- **ETT datasets**: 12/4/4 months (chronological)
- **Weather / others**: 60% / 20% / 20%
- Val is further split: 40% monitor | gap=pred_len | rest for calibration
- Calibration δ fitted on val-calibration split ONLY

### Scaler
- `StandardScaler` fitted on training split only
- For feature ablation: train-mean replacement value = `scaler.transform(train_mean)[channel]`
  (this equals ~0 by definition of StandardScaler, but explicit computation is used)

---

## 4. Artifact Structure per Run

```
results/full_3seeds_v1/E1/{dataset}_h{H}_s{seed}/
├── metrics.json          ← stored metrics (reference; do not recompute from here)
│   └── metrics_original_units → {mse, mae, interval → {picp, mpiw, pinaw, ace, ...}}
├── test_forecasts.npz    ← keys: targets, median, lower, upper, calibrated_lower, calibrated_upper
│                            shapes all [N, H, C] in ORIGINAL units
└── calibration.json      ← δ adjustment
    └── methods → horizon_channel → adjustment: list[list], shape [H, C]

results/full_3seeds_v1/E1-C/{dataset}_h{H}_s{seed}/
└── test_forecasts.npz    ← keys: targets, median, lower, upper
                             lower/upper are the calibrated bounds
```

---

## 5. Run Inventory

**Available runs** (confirmed by `audit_artifacts.py`):

| Dataset | Horizons | Seeds | n runs |
|---------|----------|-------|--------|
| ETTh1   | 96/192/336/720 | 2021/2023/2025 | 12 |
| ETTh2   | 96/192/336/720 | 2021/2023/2025 | 12 |
| ETTm1   | 96/192/336/720 | 2021/2023/2025 | 12 |
| ETTm2   | 96/192/336/720 | 2021/2023/2025 | 12 |
| Weather | 96/192/336     | **2021 only**  | 3  |
| Weather | 720            | —              | 0 (MISSING) |
| **Total** | | | **51** |

> [!WARNING]
> Weather has only n=1 seed per horizon for H∈{96,192,336}. Standard deviation for
> Weather is undefined (n=1). Do NOT extrapolate to H=720 or additional seeds.
> Tables must report n alongside each Weather statistic.

---

## 6. Metric Definitions (from `metrics/interval_metrics.py`)

| Metric | Formula | Notes |
|--------|---------|-------|
| MSE    | `mean((median − target)²)` over all N×H×C | Point forecast error |
| MAE    | `mean(|median − target|)` over all N×H×C | |
| PICP   | `mean(lower ≤ target ≤ upper)` over all N×H×C | Marginal coverage |
| MPIW   | `sum(upper − lower) / (N×H×C)` | In original units |
| PINAW  | `mean_C[(avg_width_C / range_C_in_test)]` | Normalized; range from test set |
| ACE    | `|PICP − 0.95|` | Absolute calibration error |

> [!NOTE]
> CRPS is **not computed** in this study: the model outputs only 3 quantiles
> (q0.025, q0.5, q0.975) and does not describe the full predictive distribution.
> Winkler Score can be computed from stored NPZ files if needed post-hoc.

---

## 7. Running the XAI Pipeline

```powershell
# From repo root (d:\TVCFORMER)

# Step 1: Audit artifacts
d:\.venv\Scripts\python.exe FinalProject/Interval_TVC/xAI/audit_artifacts.py

# Step 2: Calibration analysis (no inference needed)
d:\.venv\Scripts\python.exe FinalProject/Interval_TVC/xAI/calibration_analysis.py

# Step 3: Plots (no inference needed)
d:\.venv\Scripts\python.exe FinalProject/Interval_TVC/xAI/plot_xai.py

# Step 4: Feature ablation (requires DGL; set env var)
$env:DGLBACKEND="pytorch"
d:\.venv\Scripts\python.exe FinalProject/Interval_TVC/xAI/feature_ablation.py

# Or all steps via orchestrator
d:\.venv\Scripts\python.exe FinalProject/Interval_TVC/xAI/run_xai.py --skip-ablation
```

> [!IMPORTANT]
> Feature ablation runs the model for every test window × every channel × every run.
> On CPU, expected runtime: ~3–5 min per ETT run × 48 ETT runs = several hours.
> Run with `--skip-ablation` first to get all non-inference outputs immediately.

---

## 8. Output Files

| File | Description |
|------|-------------|
| `results/xai_run_inventory.csv` | All 180 slots with status |
| `results/xai_audit.json` | Summary counts by variant |
| `results/e1_vs_e1c_metrics.csv` | PICP/MPIW/PINAW/ACE for E1 and E1-C, all 51 runs |
| `results/calibration_delta_by_run.csv` | δ statistics per run |
| `results/calibration_summary.csv` | Per-dataset aggregation |
| `results/feature_ablation_raw.csv` | Per-run × per-channel ablation metrics |
| `results/feature_ablation_summary.csv` | Averaged over seeds |
| `figures/fig1_picp_dumbbell.png` | E1 vs E1-C PICP per run |
| `figures/fig2_mpiw_comparison.png` | MPIW by dataset |
| `figures/fig3_picp_mpiw_scatter.png` | PICP–MPIW tradeoff with arrows |
| `figures/fig4_delta_heatmap_*.png` | CQR δ heatmaps per dataset |
| `figures/fig5_interval_before_after.png` | Before/after calibration on OT |
| `figures/fig6_delta_distribution.png` | Distribution of all δ values |
