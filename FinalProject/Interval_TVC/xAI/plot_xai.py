"""
plot_xai.py – Generate all XAI figures for Interval-TVC.

Requires:
  xAI/results/e1_vs_e1c_metrics.csv          (from calibration_analysis.py)
  xAI/results/calibration_delta_by_run.csv   (from calibration_analysis.py)
  E1 / E1-C test_forecasts.npz               (for before/after plot)
  E1 calibration.json                        (for δ heatmaps)

Outputs (all in xAI/figures/):
  fig1_picp_dumbbell.png          – E1 vs E1-C PICP per run, 0.95 reference
  fig2_mpiw_comparison.png        – E1 vs E1-C MPIW per dataset
  fig3_picp_mpiw_scatter.png      – PICP–MPIW tradeoff with arrows E1→E1-C
  fig4_delta_heatmap_ETTh1.png    – δ heatmap [horizon, channel] for ETTh1
  fig5_interval_before_after.png  – prediction interval overlay, OT channel
  fig6_delta_distribution.png     – distribution of δ across all runs
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

THIS_DIR   = Path(__file__).resolve().parent
ITVC_DIR   = THIS_DIR.parent
RESULTS_V1 = ITVC_DIR / "results" / "full_3seeds_v1"
OUT_FIG    = THIS_DIR / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)
RES_DIR    = THIS_DIR / "results"

PALETTE = {
    "E1":  "#4C72B0",   # blue
    "E1C": "#DD8452",   # orange
    "ref": "#2CA02C",   # green reference line
}

ETT_CH = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


# ── Load comparison CSV ────────────────────────────────────────────────────────
df = pd.read_csv(RES_DIR / "e1_vs_e1c_metrics.csv")
df_dlt = pd.read_csv(RES_DIR / "calibration_delta_by_run.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1: Dumbbell – PICP per run, E1 vs E1-C
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(20, 9), sharey=False)
datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]

for ax, ds in zip(axes, datasets):
    sub = df[df.dataset == ds].sort_values(["pred_len", "seed"])
    y_pos = np.arange(len(sub))
    ax.axvline(0.95, color=PALETTE["ref"], lw=1.5, ls="--", label="0.95 target")
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.plot([row.e1_picp, row.ec_picp], [i, i], color="#aaa", lw=1.0, zorder=1)
    ax.scatter(sub.e1_picp,  y_pos, color=PALETTE["E1"],  s=50, zorder=2, label="E1")
    ax.scatter(sub.ec_picp,  y_pos, color=PALETTE["E1C"], s=50, marker="D", zorder=3, label="E1-C")
    labels = [f"h{r.pred_len}_s{r.seed}" for _, r in sub.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("PICP", fontsize=9)
    ax.set_title(ds, fontweight="bold")
    ax.set_xlim(0.72, 1.01)
    ax.grid(axis="x", alpha=0.3)

axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("PICP: E1 (raw quantile) vs E1-C (calibrated) per run\n"
             "Dashed line = nominal coverage 0.95", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_FIG / "fig1_picp_dumbbell.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_picp_dumbbell.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2: MPIW by dataset – grouped bar
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))
ds_list = df.dataset.unique()
x = np.arange(len(ds_list))
w = 0.35
e1_means  = [df[df.dataset==d].e1_mpiw.mean()  for d in ds_list]
ec_means  = [df[df.dataset==d].ec_mpiw.mean()  for d in ds_list]
e1_stds   = [df[df.dataset==d].e1_mpiw.std()   for d in ds_list]
ec_stds   = [df[df.dataset==d].ec_mpiw.std()   for d in ds_list]

ax.bar(x - w/2, e1_means, w, yerr=e1_stds, color=PALETTE["E1"],  label="E1",   alpha=0.85, capsize=4)
ax.bar(x + w/2, ec_means, w, yerr=ec_stds, color=PALETTE["E1C"], label="E1-C", alpha=0.85, capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(ds_list, fontsize=10)
ax.set_ylabel("MPIW (original units)", fontsize=10)
ax.set_title("Mean Prediction Interval Width: E1 vs E1-C by dataset\n(mean ± std across runs; note: Weather n=1 seed per horizon)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_FIG / "fig2_mpiw_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_mpiw_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3: PICP–MPIW scatter with arrows E1→E1-C (colour = dataset)
# ═══════════════════════════════════════════════════════════════════════════════
ds_colors = {
    "ETTh1": "#1f77b4", "ETTh2": "#ff7f0e", "ETTm1": "#2ca02c",
    "ETTm2": "#d62728", "Weather": "#9467bd",
}
fig, ax = plt.subplots(figsize=(9, 7))
ax.axvline(0.95, color=PALETTE["ref"], lw=1.5, ls="--", alpha=0.6, label="PICP=0.95")

for _, row in df.iterrows():
    c = ds_colors.get(row.dataset, "gray")
    ax.annotate("",
        xy=(row.ec_picp, row.ec_mpiw),
        xytext=(row.e1_picp, row.e1_mpiw),
        arrowprops=dict(arrowstyle="->", color=c, lw=0.9, alpha=0.6))
    ax.scatter(row.e1_picp,  row.e1_mpiw,  color=c, s=35, marker="o", alpha=0.7)
    ax.scatter(row.ec_picp,  row.ec_mpiw,  color=c, s=35, marker="D", alpha=0.9)

legend_handles = [mpatches.Patch(color=v, label=k) for k, v in ds_colors.items()]
legend_handles += [
    plt.scatter([], [], marker="o", color="gray", label="E1 (tail)"),
    plt.scatter([], [], marker="D", color="gray", label="E1-C (head)"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
ax.set_xlabel("PICP", fontsize=11)
ax.set_ylabel("MPIW (original units)", fontsize=11)
ax.set_title("PICP–MPIW tradeoff: arrows show E1→E1-C shift\n"
             "(higher PICP = better coverage; lower MPIW = narrower intervals)", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_FIG / "fig3_picp_mpiw_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_picp_mpiw_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4: δ heatmaps per dataset (average over seeds)
# ═══════════════════════════════════════════════════════════════════════════════
for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
    runs = sorted((RESULTS_V1 / "E1").glob(f"{ds}_h*_s*"))
    all_adj = {}   # pred_len → list of [H, C] arrays
    for rd in runs:
        cj = rd / "calibration.json"
        if not cj.is_file():
            continue
        d = json.load(open(cj))
        adj = np.array(d["methods"]["horizon_channel"]["adjustment"])  # [H, C]
        pl  = int(rd.name.split("_")[1].replace("h", ""))
        all_adj.setdefault(pl, []).append(adj)

    if not all_adj:
        continue

    pred_lens = sorted(all_adj.keys())
    nrow = len(pred_lens)
    fig, axes = plt.subplots(1, nrow, figsize=(4*nrow, 5), squeeze=False)
    for col, pl in enumerate(pred_lens):
        mats = all_adj[pl]
        mat_mean = np.stack(mats, axis=0).mean(axis=0)   # [H, C]
        ax = axes[0][col]
        # Subsample horizon axis for readability: show every k-th step
        step = max(1, pl // 24)
        im = ax.imshow(mat_mean[::step, :], aspect="auto",
                       cmap="RdBu_r", vmin=-np.abs(mat_mean).max(), vmax=np.abs(mat_mean).max())
        ax.set_title(f"H={pl} (n={len(mats)} seeds)", fontsize=9)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Horizon step (sub-sampled)")
        ax.set_xticks(range(7))
        ax.set_xticklabels(ETT_CH, fontsize=7, rotation=45)
        plt.colorbar(im, ax=ax, fraction=0.046, label="δ")

    fig.suptitle(f"{ds}: CQR adjustment δ [horizon, channel]\n"
                 f"(mean over {len(runs)//len(pred_lens)} seeds; positive = interval widened)", fontsize=11)
    plt.tight_layout()
    fname = f"fig4_delta_heatmap_{ds}.png"
    fig.savefig(OUT_FIG / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5: Before/after calibration – ETTh1 H96 seed2021, OT channel (index 6)
# ═══════════════════════════════════════════════════════════════════════════════
primary_run = "ETTh1_h96_s2021"
e1_npz  = RESULTS_V1 / "E1"   / primary_run / "test_forecasts.npz"
e1c_npz = RESULTS_V1 / "E1-C" / primary_run / "test_forecasts.npz"

if e1_npz.is_file() and e1c_npz.is_file():
    e1  = np.load(e1_npz)
    e1c = np.load(e1c_npz)

    targets = e1["targets"]    # [N, 96, 7]
    lower   = e1["lower"]
    upper   = e1["upper"]
    median  = e1["median"]
    lower_c = e1c["lower"]
    upper_c = e1c["upper"]

    CH = 6   # OT channel
    H  = 96  # all horizon steps
    # Pick a contiguous slice of test samples to display as a time series
    # Flatten: N windows × H steps for OT
    n_show = 100   # show 100 windows
    start  = len(targets) // 2   # middle of test set
    t  = targets[start:start+n_show, :, CH].flatten()     # [n_show*H]
    lo = lower  [start:start+n_show, :, CH].flatten()
    up = upper  [start:start+n_show, :, CH].flatten()
    lo_c = lower_c[start:start+n_show, :, CH].flatten()
    up_c = upper_c[start:start+n_show, :, CH].flatten()
    md   = median [start:start+n_show, :, CH].flatten()
    xs   = np.arange(len(t))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: E1 raw
    ax = axes[0]
    ax.fill_between(xs, lo, up, alpha=0.3, color=PALETTE["E1"], label="90% interval (E1)")
    ax.plot(xs, md, color=PALETTE["E1"], lw=1.0, alpha=0.8, label="Median (E1)")
    ax.plot(xs, t,  color="black", lw=1.2, alpha=0.9, label="Actual")
    picp_e1 = float(((t >= lo) & (t <= up)).mean())
    ax.set_title(f"E1 raw (before calibration) — PICP={picp_e1:.3f} on shown slice", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylabel("OT value (original units)")
    ax.grid(alpha=0.25)

    # Bottom: E1-C calibrated
    ax = axes[1]
    ax.fill_between(xs, lo_c, up_c, alpha=0.3, color=PALETTE["E1C"], label="90% interval (E1-C)")
    ax.plot(xs, md, color=PALETTE["E1C"], lw=1.0, alpha=0.8, label="Median (E1-C=E1 median)")
    ax.plot(xs, t,  color="black", lw=1.2, alpha=0.9, label="Actual")
    picp_ec = float(((t >= lo_c) & (t <= up_c)).mean())
    ax.set_title(f"E1-C (after calibration) — PICP={picp_ec:.3f} on shown slice", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlabel(f"Time step (100 consecutive windows × H=96, OT channel)")
    ax.set_ylabel("OT value (original units)")
    ax.grid(alpha=0.25)

    fig.suptitle(f"ETTh1 H=96 seed=2021 | OT channel | middle 100 test windows\n"
                 f"Note: median prediction identical in both; only bounds differ", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_FIG / "fig5_interval_before_after.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig5_interval_before_after.png")
else:
    print(f"[SKIP] fig5 – NPZ not found for {primary_run}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6: Distribution of δ values across all runs
# ═══════════════════════════════════════════════════════════════════════════════
all_deltas = []
for rd in sorted((RESULTS_V1 / "E1").glob("*_h*_s*")):
    cj = rd / "calibration.json"
    if not cj.is_file():
        continue
    try:
        d   = json.load(open(cj))
        adj = np.array(d["methods"]["horizon_channel"]["adjustment"])
        all_deltas.append(adj.flatten())
    except Exception:
        pass

if all_deltas:
    vals = np.concatenate(all_deltas)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vals, bins=100, color=PALETTE["E1"], alpha=0.7, edgecolor="none")
    ax.axvline(0, color="red", lw=1.5, ls="--", label="δ = 0")
    ax.axvline(np.median(vals), color="green", lw=1.5, ls="--", label=f"median δ = {np.median(vals):.3f}")
    neg_pct = 100 * (vals < 0).mean()
    ax.set_xlabel("δ (CQR adjustment)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Distribution of CQR δ values across all 51 E1 runs\n"
                 f"({neg_pct:.1f}% negative → those horizon-channel cells have tightened lower bound)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_FIG / "fig6_delta_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig6_delta_distribution.png")

print("\nAll figures saved to:", OUT_FIG)
