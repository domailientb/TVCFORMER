# ATVF - Final Project branch

This branch adds an experimental **Adaptive Time-Variable Coupling TVCFormer (ATVF)** while keeping the original TVCFormer baseline available.

## What is added

- `layers/ATVF_GraphGenerator.py`: input-conditioned temporal edge scores, learnable channel-pair bias, top-k edge selection, and graph regularization.
- `models/ATVF.py`: wraps the original forecasting model and replaces only its graph generator.
- `exp/exp_main.py`: registers ATVF and adds its auxiliary graph loss during training.
- `configs/ATVF_ETTh1.yaml` and `configs/ATVF_ExchangeRate.yaml`: experiment configurations.
- `experiments/run_atvf.py`: runs the requested prediction horizons and seeds.
- `experiments/collect_atvf_results.py`: extracts MSE/MAE from the official `result.txt`.

The baseline model implementation is not replaced. ATVF is a research proposal and is **not claimed to outperform** TVCFormer or PatchTST until the experiments establish that.

## Environment

Use the compatible environment specified by `official/README.md`: Python 3.9, PyTorch 2.0, DGL 1.1, plus `pip install -r requirements.txt`. Install the matching PyTorch/DGL build for the machine's CUDA version if using GPU.

## Data

- ETTh1: `official/datasets/Ett_Series/ETTh1.csv`
- Exchange_rate: `official/datasets/Customs/exchange_rate.csv`

The Exchange_rate CSV should contain the standard eight numeric exchange-rate columns. A leading date column is supported and ignored for forecasting; a file containing only the eight numeric columns is also supported. Keep `enc_in: 8`.

## Run

From the `official/` directory:

```bash
python experiments/run_atvf.py --data ETTh1
python experiments/run_atvf.py --data exchange_rate
```

For a quick smoke run:

```bash
python experiments/run_atvf.py --data ETTh1 --seeds 2021 --pred_lens 96 --epochs 1
```

Then aggregate results:

```bash
python experiments/collect_atvf_results.py --input result.txt --output results/atvf_summary.csv
```

Compare this CSV with the existing TVCFormer and PatchTST result files. Use identical data splits, sequence length, horizons, and seeds for a fair comparison.

## Important

1. Check that the installed `torch`, `dgl`, and CUDA versions are compatible before starting long runs.
2. The default batch size is reduced to 32 to reduce memory pressure. Increase it only after confirming available memory.
3. The regularization coefficients and edge keep rate are starting hyperparameters; tune them on validation data, not the test set.
4. Do not present a performance improvement as a result until the runs have completed.
