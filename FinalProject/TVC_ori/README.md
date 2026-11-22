# TVCFormer original runs

This folder reuses the unmodified `official/run_TVCFormer.py` entry point and
the official `Exp_Main`, model, data provider, and utility modules. It does not
import code from Clustering-TVCFormer or MultiMetric-TVCFormer and does not
patch files in `official/`. Dataset YAMLs here preserve the values in
`official/configs/`; dataset paths are adapted to this repository's available
data files.

Run from the repository root with the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -u .\FinalProject\TVC_ori\run_tvc_ori.py --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather --seeds 2023 2025
```

To run one dataset/seed or inspect the commands without training:

```powershell
.\.venv\Scripts\python.exe -u .\FinalProject\TVC_ori\run_tvc_ori.py --datasets ETTh1 --seeds 2023
.\.venv\Scripts\python.exe -u .\FinalProject\TVC_ori\run_tvc_ori.py --datasets ETTh1 --seeds 2025 --dry-run
```

The runner executes prediction lengths 96, 192, 336, and 720 sequentially.
Each dataset+seed appends all horizons and stdout/stderr to one file under
`logs/`. Checkpoints, prediction outputs, test visualizations, and `result.txt`
are kept under this project directory; each horizon gets a unique timestamped
model ID and checkpoint path so reruns do not replace prior artifacts.
Training/validation/test logic is the original official entry point's logic.

The official runner seeds Python `random`, NumPy, PyTorch, and DGL using the
requested seed. DataLoader worker count remains zero as specified in the
configs. No training command has been run as part of preparing this folder.
