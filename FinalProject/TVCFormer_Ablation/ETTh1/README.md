# ETTh1 feature ablation and TVC graph analysis

ETTh1 has seven numeric features: HUFL, HULL, MUFL, MULL, LUFL, LULL, and OT.
This folder contains the four horizon checkpoints, configs, scripts, results,
figures, and reports. The 336 and 720 checkpoints and analyses are included.

## Feature ablation

For each horizon the runner trains one full-feature TVCFormer, selects its
checkpoint by validation MSE, then evaluates the fixed checkpoint on the test
split with each input feature replaced by its standardized train-split mean.
It reports baseline and ablation MSE/MAE and their differences; it does not
retrain ablated models.

```powershell
.\.venv\Scripts\python.exe .\FinalProject\TVCFormer_Ablation\ETTh1\run_etth1_ablation.py
```

Results are written to `ETTh1/results/ETTh1_mean_replacement_ablation.csv`.

## TVC graph analysis

The saved checkpoints can be processed in inference-only mode:

```powershell
.\.venv\Scripts\python.exe .\FinalProject\TVCFormer_Ablation\ETTh1\extract_etth1_graphs.py
```

The script extracts 49x49 graphs and aggregates their variable-to-variable
blocks into 7x7 matrices. Matrices, heatmaps, weighted degrees, edge rankings,
stability metrics, and the report are under `ETTh1/results/graphs/`.
