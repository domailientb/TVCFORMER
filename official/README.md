# TVCFormer: A Transformer-Based Long-Term Multivariate Time Series Forecasting Method Using Time-Variable Coupling Correlation Graph

This is the official Python implementation of **TVCFormer**.

DOI: [10.1016/j.knosys.2025.113147](https://doi.org/10.1016/j.knosys.2025.113147)

## Requirements

1. Python 3.9, PyTorch 2.0, and DGL 1.1 (GPU versions recommended).
2. Install additional dependencies with:

   ```bash
   pip install -r requirements.txt
   ```

## Data

We use six public datasets in this work: **ETTh1**, **ETTh2**, **ETTm1**, **ETTm2**, **Weather**, and **Electricity**.
These datasets are consistent with the pre-processed versions provided in *\[Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting]*.

* Datasets can be downloaded from the [Autoformer GitHub repository](https://github.com/thuml/Autoformer).
* Place them into the `datasets/` folder. For example:

  ```
  datasets/Customs/electricity.csv
  datasets/Ett_Series/ETTh1.csv
  ```

## Usage

### 1. Setup

* Install the required dependencies (see [Requirements](#requirements)).
* Download datasets and place them into the correct folder (see [Data](#data)).

### 2. Run a Single Experiment

To conduct a single experiment, simply run the `run_TVCFormer.py` script.
For example, to train and evaluate a model with input historical series length $T_s = 336$ and forecast length $T_p = 192$ on the ETTh1 dataset:

```bash
python -u run_TVCFormer.py \
  --config_yaml configs/ETTh1.yaml \
  --is_training 1 \
  --model_id TVCFormer_ETTh1_336_192 \
  --model TVCFormer \
  --data ETTh1 \
  --seq_len 336 \
  --pred_len 192
```

* **Checkpoints**: Saved in
  `checkpoints/Ett_series/ETTh1/TVCFormer_ETTh1_336_192_.../checkpoint.pth`
* **Forecasting result data**: Saved in
  `results/TVCFormer_ETTh1_336_192_.../pred.npy`
* **Forecast visualizations**: Stored in
  `test_results/TVCFormer_ETTh1_336_192_.../`

  * Orange line = forecasted values
  * Blue line = ground truth

### 3. Visualization & Analysis

To further visualize results, set `--is_training 0`, enable `--visualize` and control the number of visualization samples using `--visual_samples_num`.
For example:

```bash
python -u run_TVCFormer.py \
  --config_yaml configs/ETTh1.yaml \
  --is_training 0 \
  --model_id TVCFormer_ETTh1_336_192 \
  --model TVCFormer \
  --data ETTh1 \
  --seq_len 336 \
  --pred_len 192 \
  --visualize 1 \
  --visual_samples_num 15
```

Generated outputs include:

* **Normalized input series** → `visualize_results/[model_id].../visual_data/inputs`
* **Correlation coefficient matrices** → `visualize_results/[model_id].../visual_data/corrs`
* **TVC graph adjacency matrices** → `visualize_results/[model_id].../visual_data/graphs`

Each file follows the naming pattern `[Index].npy` (data) and `[Index].png` (image).

### 4. Batch Experiments

To run multiple experiments in batch, execute the provided scripts:

```bash
sh scripts/ETTh1.sh
sh scripts/ETTh2.sh
sh scripts/ETTm1.sh
sh scripts/ETTm2.sh
sh scripts/electricity.sh
sh scripts/weather.sh
...
```

* The evaluation results are saved in `result.txt`
* Logs are saved in `logs/LongForecasting/[model_id].log`

---

## Parameters

Parameters are defined and managed using **argparse** and **YAML**.
The priority order of parameter assignment is:

**Command line arguments > YAML config file > Default values**

### Structure

1. **`run_TVCFormer.py`**

   * Defines all parameters in `parser`
   * `config_parser` specifies the `config_yaml` path
   * Reference YAML configuration files corresponding to `config_yaml` are provided in the `configs/` folder

2. **YAML Config File**

   * The YAML configuration file provides customized definitions for parameters such as dataset settings, training options, and model hyperparameters.

3. **Command Line Arguments**

   * Must specify core parameters:
     `config_yaml`, `is_training`, `model_id`, `model`, `data`, `seq_len`, `pred_len`
   * Can override any additional parameter.
     For example, setting `time_node_num=3`:

     ```bash
     python -u run_TVCFormer.py \
       --config_yaml configs/ETTh1.yaml \
       --is_training 1 \
       --time_node_num 3 \
       --model_id TVCFormer_ETTh1_336_192 \
       --model TVCFormer \
       --data ETTh1 \
       --seq_len 336 \
       --pred_len 192
     ```

## Citation

If you use **TVCFormer** in your work, please cite the following paper:

```bibtex
@article{liu2025tvc,
  title={TVC Former: A transformer-based long-term multivariate time series forecasting method using time-variable coupling correlation graph},
  author={Liu, Zhenyu and Feng, Yuan and Liu, Hui and Tang, Ruining and Yang, Bo and Zhang, Donghao and Jia, Weiqiang and Tan, Jianrong},
  journal={Knowledge-Based Systems},
  volume={314},
  pages={113147},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgements

We sincerely thank all readers and researchers for their interest in **TVCFormer**.
Your attention and feedback are greatly appreciated and will help us improve our work.
