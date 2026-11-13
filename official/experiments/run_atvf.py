"""Run ATVF on ETTh1 or Exchange_rate for several seeds and horizons.

Run from the official/ directory:
    python experiments/run_atvf.py --data ETTh1
    python experiments/run_atvf.py --data exchange_rate
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "run_TVCFormer.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["ETTh1", "exchange_rate"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023])
    parser.add_argument("--pred_lens", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = args.config or (
        "configs/ATVF_ETTh1.yaml" if args.data == "ETTh1"
        else "configs/ATVF_ExchangeRate.yaml"
    )

    for seed in args.seeds:
        for pred_len in args.pred_lens:
            model_id = f"ATVF_{args.data}_336_{pred_len}_seed{seed}"
            cmd = [
                sys.executable, RUNNER,
                "--config_yaml", config,
                "--is_training", "1",
                "--model_id", model_id,
                "--model", "ATVF",
                "--data", args.data,
                "--seq_len", "336",
                "--pred_len", str(pred_len),
                "--random_seed", str(seed),
            ]
            if args.epochs is not None:
                cmd += ["--train_epochs", str(args.epochs)]
            print("\n>>>", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
