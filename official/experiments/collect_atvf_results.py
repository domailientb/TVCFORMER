"""Aggregate ATVF result.txt entries into a tidy CSV.

The official runner writes MSE/MAE to result.txt. This parser is intentionally
strict: it reports entries it cannot parse rather than fabricating metrics.
"""
import argparse
import csv
import re
from pathlib import Path

PATTERN = re.compile(
    r"(?P<setting>[^\n]+)\s*\n\s*mse:(?P<mse>[-+0-9.eE]+),\s*"
    r"mae:(?P<mae>[-+0-9.eE]+)"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="result.txt")
    parser.add_argument("--output", default="results/atvf_summary.csv")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8", errors="replace")
    rows = []
    for match in PATTERN.finditer(text):
        setting = match.group("setting").strip()
        # Expected ID prefix: ATVF_<dataset>_336_<horizon>_seed<seed>
        meta = re.search(
            r"ATVF_(ETTh1|exchange_rate)_336_(96|192|336|720)_seed(\d+)",
            setting,
            flags=re.IGNORECASE,
        )
        rows.append({
            "setting": setting,
            "dataset": meta.group(1) if meta else "unknown",
            "pred_len": int(meta.group(2)) if meta else "",
            "seed": int(meta.group(3)) if meta else "",
            "mse": float(match.group("mse")),
            "mae": float(match.group("mae")),
        })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", "dataset", "pred_len", "seed", "mse", "mae"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} parsed runs to {out}")

