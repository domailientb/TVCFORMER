# TVCFormer XAI experiments

Each dataset has an isolated folder with its own configs, scripts, checkpoints,
results, figures, and report.

| Dataset | Numeric features | Graph sizes | Horizons |
|---|---:|---:|---|
| ETTh1 | 7 | 49x49 and 7x7 | 96, 192, 336, 720 |
| ExchangeRate | 8 | 64x64 and 8x8 | 96, 192, 336, 720 |

ETTh1 is already complete. Its results and scripts are in `ETTh1/`. ExchangeRate
uses an official TVCFormer checkpoint configured with eight temporal nodes; this
is required for a genuine 64x64 graph. From the repository root, run:

```powershell
.\.venv\Scripts\python.exe .\FinalProject\TVCFormer_Ablation\Exchangerate\run_exchangerate_ablation.py
.\.venv\Scripts\python.exe .\FinalProject\TVCFormer_Ablation\Exchangerate\plot_exchangerate_xai.py
.\.venv\Scripts\python.exe .\FinalProject\TVCFormer_Ablation\Exchangerate\extract_exchangerate_graphs.py
```

For ETTh1, graph extraction is inference-only and uses the four saved
checkpoints:

```powershell
.\.venv\Scripts\python.exe .\FinalProject\TVCFormer_Ablation\ETTh1\extract_etth1_graphs.py
.\.venv\Scripts\python.exe .\FinalProject\TVCFormer_Ablation\ETTh1\compare_xai_methods.py
```

Feature ablation replaces one standardized input variable by its train-split
mean and evaluates the unchanged checkpoint. Positive ΔMSE/ΔMAE means the
replacement increased the forecast error. Graph weighted degree measures
connectivity, which is a different descriptive property from ablation
sensitivity.
