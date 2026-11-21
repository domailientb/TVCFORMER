# Quantile Interval TVCFormer

This project adds a three-quantile forecast head in a new folder and does not
edit `official/`. E0 trains the official point-forecast model with MSE. E1
reuses the official TVCFormer graph construction, GraphSAGE, joint encoding,
Transformer, and input normalization, replacing only the final flatten
prediction head with a quantile head.

The official graph is **batch-dependent**: each forward pass selects signed
Pearson channel pairs from that batch, then passes them through the existing
`GraphGenerator`. E1 follows that same per-batch path. The graph, self-loop
handling, batch offsets, and channel-to-local-node mapping are unchanged.

## Interval parameterization and objective

The head predicts a median and two unconstrained gap parameters. In normalized
coordinates:

```text
lower = median - softplus(lower_gap)
upper = median + softplus(upper_gap)
```

The three forecast quantiles are `0.025`, `0.5`, and `0.975`, trained with the
equally weighted mean pinball loss. RevIN denormalizes each quantile with the
same per-instance statistics used by the official model.

The data loader uses the official ETTh1 chronological split and fits its
scaler on the training segment only. Validation is used for early stopping;
test windows are evaluated only after restoring the best validation
checkpoint. Reported MSE, MAE, coverage, and widths are in the original data
units.

## Run a smoke test

From the repository root in Git Bash:

```powershell
./FinalProject/Interval_TVC/scripts/ETTh1.sh
```

The full dataset scripts run 3 seeds × 4 horizons × E0/E1 and therefore should
only be started after the smoke matrix succeeds.

## Train E0 and E1

Run both experiments with the same seed and horizon for a matched comparison:

```powershell
.\.venv\Scripts\python.exe -u .\FinalProject\Interval_TVC\train_interval.py --config .\FinalProject\Interval_TVC\configs\interval.yaml --dataset ETTh1 --experiment E0 --pred_len 96 --seed 2021
.\.venv\Scripts\python.exe -u .\FinalProject\Interval_TVC\train_interval.py --config .\FinalProject\Interval_TVC\configs\interval.yaml --dataset ETTh1 --experiment E1 --pred_len 96 --seed 2021
```

The same commands accept prediction horizons 192, 336, and 720. For example:

```powershell
.\.venv\Scripts\python.exe -u .\FinalProject\Interval_TVC\train_interval.py --config .\FinalProject\Interval_TVC\configs\interval.yaml --dataset ETTh1 --experiment E0 --pred_len 192 --seed 2021
.\.venv\Scripts\python.exe -u .\FinalProject\Interval_TVC\train_interval.py --config .\FinalProject\Interval_TVC\configs\interval.yaml --dataset ETTh1 --experiment E1 --pred_len 192 --seed 2021
```

To run a shorter diagnostic training, pass `-Epochs 2`; to reduce VRAM use
`-BatchSize 16`. Both are overrides and must be applied equally to E0 and E1
for an apples-to-apples experiment. The default batch size is 32 for the
4-GB GPU setting.

The training runner uses the repository's `.venv` Python. Dependencies are
the same as official TVCFormer (PyTorch, DGL, NumPy, pandas, PyYAML,
scikit-learn, and matplotlib); it does not install packages or modify
dependency files.

## Metrics and artifacts

Per-run `metrics.json` and `test_forecasts.npz` are saved beneath
`results/E0/` or `results/E1/`; checkpoints are written beneath `checkpoints/`.
E1 also saves `interval_examples_ot.png` for deterministic first, middle, and
last test windows of the OT channel. No windows are selected based on their
forecast error.

The interval report includes:

- Median point MSE and MAE.
- PICP: fraction of test targets within the interval.
- MPIW: mean interval width in original units.
- PINAW: per-channel mean width divided by that channel's test-target range,
  averaged over channels.
- ACE: absolute difference between empirical PICP and the nominal coverage.

These are empirical test metrics, not a finite-sample coverage guarantee.
Overlapping time-series forecast windows are dependent.

After both E0 and E1 have been run for each desired horizon:

```powershell
.\.venv\Scripts\python.exe .\FinalProject\Interval_TVC\compare_results.py --horizons 96 192 336 720
.\.venv\Scripts\python.exe .\FinalProject\Interval_TVC\visualize_results.py --horizons 96 192 336 720
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest .\FinalProject\Interval_TVC\tests -q
```

No experiment is described as better than TVCFormer or other models until
matched runs have completed; this implementation alone makes no performance
claim.

## Full multi-dataset experiments

The official dataset-specific hyperparameters are stored in one combined
`configs/interval.yaml` file.

Run one dataset with the three-seed protocol:

```powershell
./FinalProject/Interval_TVC/scripts/ETTh1.sh
./FinalProject/Interval_TVC/scripts/ETTh2.sh
./FinalProject/Interval_TVC/scripts/ETTm1.sh
./FinalProject/Interval_TVC/scripts/ETTm2.sh
./FinalProject/Interval_TVC/scripts/Weather.sh
```

The runner uses seeds 2021, 2023, and 2025 and horizons 96, 192, 336, and
720. It runs one E0/E1 model at a time on the GTX 1650 and resumes completed
runs. A run is resumed only when both prediction artifacts and its required
metrics/calibration artifacts exist. New official runs are stored under
`results/full_3seeds_v1/` and `checkpoints/full_3seeds_v1/`; they do not
reuse or overwrite previous ETTh1 or calibration results.

Run the pre-training smoke matrix for all datasets:

```powershell
.\.venv\Scripts\python.exe .\FinalProject\Interval_TVC\run_full_experiment.py --smoke_test --batch_size 2
```

Aggregate completed runs:

```powershell
.\.venv\Scripts\python.exe .\FinalProject\Interval_TVC\aggregate_results.py
```
