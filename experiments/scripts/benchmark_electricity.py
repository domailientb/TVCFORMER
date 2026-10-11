
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = PROJECT_ROOT / "official"
CONFIG_FILE = PROJECT_ROOT / "experiments" / "configs" / "electricity.yaml"

sys.path.insert(0, str(OFFICIAL_DIR))

from models.TVCFormer import Model


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # These settings match the Electricity experiment.
    config.update({
        "seq_len": 336,
        "pred_len": 96,
        "enc_in": 321,
        "model": "TVCFormer",
        "use_gpu": True,
        "gpu": 0,
        "use_multi_gpu": False,
        "device": torch.device("cuda:0"),
        "use_amp": False,
    })

    # Fill defaults required by Model if absent from YAML.
    defaults = {
        "eps": 1e-5,
        "graph_keep_self_cycles": False,
        "fc_dropout": 0.0,
        "head_dropout": 0.0,
        "individual": False,
        "padding_patch": "end",
        "revins": True,
        "subtract_last": False,
        "decomposition": False,
        "decomposition_kernel_size": 25,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    args = SimpleNamespace(**config)
    device = torch.device("cuda:0")

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Input shape per sample: [336, 321]")
    print("Prediction length:", args.pred_len)
    print("AMP: disabled")
    print("NOTE: benchmark only; not a training result.")

    # Initialize once per batch size so each test uses the same model
    # architecture and the same config.
    for batch_size in (1, 2, 4):
        print(f"\n{'=' * 55}")
        print(f"Testing batch_size={batch_size}")
        torch.cuda.empty_cache()

        model = None
        x = None
        loss = None

        try:
            args.device = device
            model = Model(args).float().to(device)
            model.train()

            x = torch.randn(
                batch_size, args.seq_len, args.enc_in,
                dtype=torch.float32, device=device
            )

            # Warm-up: one forward/backward pass.
            model.zero_grad(set_to_none=True)
            output = model(x)
            loss = output.square().mean()
            loss.backward()
            torch.cuda.synchronize()

            # Measure 3 forward/backward passes.
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()

            for _ in range(3):
                model.zero_grad(set_to_none=True)
                output = model(x)
                loss = output.square().mean()
                loss.backward()

            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 3
            peak_gb = (
                torch.cuda.max_memory_allocated() / 1024**3
            )

            print(f"Average forward + backward: {elapsed:.3f} sec")
            print(f"Peak allocated VRAM: {peak_gb:.2f} GB")

        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory for this batch size.")
            torch.cuda.empty_cache()

        finally:
            del model, x, loss
            torch.cuda.empty_cache()

    print("\nBenchmark complete. No optimizer updates were performed.")


if __name__ == "__main__":
    main()
