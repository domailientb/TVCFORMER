# lưu kq TVCFormer
from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_FILE = PROJECT_ROOT / "experiments" / "results" / "reproduction_results.csv"

# Mỗi dòng gồm: Pred_len, MSE, MAE, RSE
RESULTS = {
    "ETTh1": [
        (96, 0.36449319, 0.39083716, 0.57247978),
        (192, 0.39980584, 0.41383228, 0.60044920),
        (336, 0.42267010, 0.42998970, 0.62148023),
        (720, 0.43064615, 0.45328006, 0.63029712),
    ],
    "ETTh2": [
        (96, 0.27344292, 0.33489630, 0.41773763),
        (192, 0.33650035, 0.37778836, 0.46515137),
        (336, 0.32352257, 0.38003573, 0.45440143),
        (720, 0.37429574, 0.41818500, 0.49049509),
    ],
    "ETTm1": [
        (96, 0.28850044, 0.34578973, 0.52057844),
        (192, 0.33161572, 0.36587685, 0.54772425),
        (336, 0.36226815, 0.38391072, 0.57267922),
        (720, 0.41697350, 0.41518873, 0.61424464),
    ],
    "ETTm2": [
        (96, 0.16428503, 0.25412986, 0.32835582),
        (192, 0.22106625, 0.29301935, 0.37998787),
        (336, 0.27556863, 0.32850865, 0.42352492),
        (720, 0.36042770, 0.380589395, 0.482013344),
    ],
    "Weather": [
        (96, 0.15060541, 0.19990897, 0.51124650),
        (192, 0.19451171, 0.24292062, 0.58021462),
        (336, 0.24629523, 0.28291723, 0.65191370),
        (720, 0.31443340, 0.33348766, 0.73974472),
    ],
}


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Dataset",
            "Pred_len",
            "MSE",
            "MAE",
            "RSE",
            "Seq_len",
            "Device",
            "Early_stopping",
        ])

        for dataset, rows in RESULTS.items():
            for pred_len, mse, mae, rse in rows:
                writer.writerow([
                    dataset,
                    pred_len,
                    mse,
                    mae,
                    rse,
                    336,
                    "cuda:0",
                    "yes",
                ])

    print(f"Da tao file CSV: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
