# vẽ biểu đồ so sánh giá trị thực tế với giá trị dự báo
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "figures" / "predictions"

# đọc và ktra dl
def load_npy(file_path):
    """Đọc file .npy và kiểm tra dữ liệu."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    data = np.load(file_path, allow_pickle=False)
    
    # Chỉ chấp nhận dữ liệu số và không chứa NaN/Inf.
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(f"Data must be numeric: {file_path}")
    if not np.isfinite(data).all():
        raise ValueError(f"Data contains NaN or Inf: {file_path}")
    return data

# chọn mẫu và đặc trưg
def select_sample_and_feature(data, sample, feature, name):
    """Chọn một mẫu và một đặc trưng để trực quan hóa."""
    if data.ndim == 3:
        # Shape: (samples, prediction_length, features)
        if not 0 <= sample < data.shape[0]:
            raise IndexError(
                f"sample={sample} is invalid for {name}; "
                f"available samples: {data.shape[0]}"
            )
        if not 0 <= feature < data.shape[2]:
            raise IndexError(
                f"feature={feature} is invalid for {name}; "
                f"available features: {data.shape[2]}"
            )
        result = data[sample, :, feature]
        
    elif data.ndim == 2:
        # Shape: (samples, prediction_length)
        if not 0 <= sample < data.shape[0]:
            raise IndexError(
                f"sample={sample} is invalid for {name}; "
                f"available samples: {data.shape[0]}"
            )

        result = data[sample, :]

    elif data.ndim == 1:
        # Shape: (prediction_length,)
        result = data
    else:
        raise ValueError(f"Unsupported shape for {name}: {data.shape}")
    return np.asarray(result).reshape(-1)

# thiết lập tham số đầu vào 
def main():
    parser = argparse.ArgumentParser(description="Plot TVCFormer predictions against ground truth.")
    parser.add_argument("--true", required=True, help="Path to the ground-truth .npy file")
    parser.add_argument("--pred", required=True, help="Path to the prediction .npy file")
    parser.add_argument("--dataset", default="ETTh1", help="Dataset name, e.g. ETTh1 or ETTm1")
    parser.add_argument("--pred_len", type=int, required=True, help="Prediction length: 96, 192, 336, or 720")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to plot")
    parser.add_argument("--feature", type=int, default=0, help="Feature index to plot")
    args = parser.parse_args()
    
    # kiểm tra các tham số đầu vào
    if args.pred_len <= 0:
        parser.error("--pred_len must be greater than 0")
    if args.sample < 0 or args.feature < 0:
        parser.error("--sample and --feature must be non-negative")
    
    # tải và chuẩn bị dữ liệu
    true_data = load_npy(args.true)
    pred_data = load_npy(args.pred)
    print(f"Ground-truth shape: {true_data.shape}")
    print(f"Prediction shape:   {pred_data.shape}")
    y_true = select_sample_and_feature(true_data, args.sample, args.feature, "ground truth")
    y_pred = select_sample_and_feature(pred_data, args.sample, args.feature, "prediction")
    
    # Đảm bảo hai chuỗi có cùng độ dài và đúng pred_len
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: ground truth={len(y_true)}, prediction={len(y_pred)}")
    if len(y_true) != args.pred_len:
        raise ValueError(f"Expected {args.pred_len} prediction steps, but loaded {len(y_true)}.")
    
    # VẼ GROUND TRUTH VÀ PREDICTION
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(len(y_true)), y_true, label="Ground Truth", linewidth=2)
    ax.plot(np.arange(len(y_pred)), y_pred, label="TVCFormer Prediction", linewidth=2)
    ax.set_title(f"{args.dataset}: Ground Truth vs Prediction (pred_len={args.pred_len})")
    ax.set_xlabel("Prediction time step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_file = OUTPUT_DIR / f"{args.dataset}_pred_len_{args.pred_len}_comparison.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot: {output_file}")

if __name__ == "__main__":
    main()
