# trực quan hóa TVC Graph
from pathlib import Path
import argparse
import re
import shutil
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = PROJECT_ROOT / "official"
CONFIG_DIR = OFFICIAL_DIR / "configs"
CHECKPOINT_DIR = OFFICIAL_DIR / "checkpoints"
DATA_ROOT = PROJECT_ROOT / "experiments" / "datasets"
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "figures" / "graph"

# Ánh xạ tên dataset với file cấu hình YAML tương ứng
CONFIG_NAMES = {
    "ETTh1": "ETTh1.yaml",
    "ETTh2": "ETTh2.yaml",
    "ETTm1": "ETTm1.yaml",
    "ETTm2": "ETTm2.yaml",
    "weather": "weather.yaml",
    "electricity": "electricity.yaml",
}

# đọc tham số đầu vào 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize saved TVCFormer graph matrices."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(CONFIG_NAMES),
    )
    parser.add_argument(
        "--pred_len",
        required=True,
        type=int,
        choices=[96, 192, 336, 720],
    )
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--samples", type=int, default=3)
    return parser.parse_args()

# tìm dữ liệu và checkpoint
def find_dataset_file(dataset_name):  # dn hàm tìm file
    candidates = list(DATA_ROOT.rglob(f"{dataset_name}.csv"))
    candidates += list(
        (OFFICIAL_DIR / "datasets").rglob(f"{dataset_name}.csv")
    ) if (OFFICIAL_DIR / "datasets").is_dir() else []

    candidates = list({p.resolve() for p in candidates if p.is_file()})
    # Ưu tiên dataset nằm trong thư mục experiments/datasets
    preferred = [
        p for p in candidates
        if (PROJECT_ROOT / "experiments" / "datasets").resolve()
        in p.parents
    ]

    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        raise RuntimeError(
            f"Multiple dataset files found for {dataset_name}:\n"
            + "\n".join(map(str, preferred))
        )
    if len(candidates) == 1: # nếu 0 có file ưu tiên nhưg kq sd 1 file, thì sd file đó
        return candidates[0]
    
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find {dataset_name}.csv under experiments/datasets "
            "or official/datasets."
        )

    raise RuntimeError(  # lỗi khi nh file nhưg 0 thể lựa chọn duy nhất
        f"Multiple dataset files found for {dataset_name}:\n"
        + "\n".join(map(str, candidates))
    )

#tìm mô hình đã huấn luyện
def find_checkpoint(dataset_name, seq_len, pred_len):
    pattern = (
        f"TVCFormer_{dataset_name}_{seq_len}_{pred_len}_*"
    )
    matches = list(CHECKPOINT_DIR.rglob(pattern))
    matches = [
        p for p in matches
        if p.is_dir() and (p / "checkpoint.pth").is_file() #ktra trog file có trọg số checkpoint hay 0
    ]
    # tiếp tục khi tìm được đúng một checkpoint phù hợp
    if len(matches) != 1:
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint found for {dataset_name}, "
                f"seq_len={seq_len}, pred_len={pred_len}."
            )
        raise RuntimeError(
            "Multiple checkpoints found; cannot choose safely:\n"
            + "\n".join(map(str, matches))
        )

    return matches[0]

# khôi phục siêu tham số
def recover_hyperparameters(setting):
    values = {}
     # Đọc các siêu tham số được mã hóa trong tên thư mục checkpoint
    for part in setting.split("_"):
        if part.startswith("dm"):
            values["d_model"] = part[2:]
        elif part.startswith("nh"):
            values["n_heads"] = part[2:]
        elif part.startswith("el"):
            values["e_layers"] = part[2:]
        elif part.startswith("dl"):
            values["d_layers"] = part[2:]
        elif part.startswith("df"):
            values["d_ff"] = part[2:]
        elif part.startswith("gch"):
            values["graph_conv_hidden_size"] = part[3:]
        elif part.startswith("gck"):
            values["graph_conv_kernel_size"] = part[3:]
        elif part.startswith("tn"):
            values["time_node_num"] = part[2:]
        elif part.startswith("kr"):
            values["subgraph_keep_rate"] = part[2:]

    return values

#cbi các đg dẫn
def build_paths(args):
    dataset_file = find_dataset_file(args.dataset)
    config_file = CONFIG_DIR / CONFIG_NAMES[args.dataset]

    if not config_file.is_file():
        raise FileNotFoundError(f"Config not found:\n{config_file}")

    checkpoint_dir = find_checkpoint(
        args.dataset, args.seq_len, args.pred_len
    )
    setting = checkpoint_dir.name

    output_dir = OUTPUT_ROOT / f"{args.dataset}_pred{args.pred_len}"
    official_output = (
        OFFICIAL_DIR / "visualize_results" / setting / "visual_data"
    )

    return {
        "dataset_file": dataset_file,
        "config_file": config_file,
        "checkpoint_dir": checkpoint_dir,
        "setting": setting,
        "model_id": f"TVCFormer_{args.dataset}_{args.seq_len}_{args.pred_len}",
        "output_dir": output_dir,
        "graph_dir": output_dir / "graphs",
        "official_output": official_output,
    }

# chạy trực quan hóa chính thức
def run_official_visualization(args, paths):
    official_output = paths["official_output"]

    # Xóa kết quả trực quan hóa cũ của đúng checkpoint này
    if official_output.exists():
        shutil.rmtree(official_output)

    command = [
        sys.executable,
        "run_TVCFormer.py",
        "--config_yaml", str(paths["config_file"]),
        "--is_training", "0",
        "--visualize", "True",
        "--visual_samples_num", str(args.samples),
        "--model_id", paths["model_id"],
        "--model", "TVCFormer",
        "--data", args.dataset,
        "--root_path", str(paths["dataset_file"].parent),
        "--data_path", paths["dataset_file"].name,
        "--seq_len", str(args.seq_len),
        "--pred_len", str(args.pred_len),
    ]
    # Truyền lại các siêu tham số lấy từ tên checkpoint
    for key, value in recover_hyperparameters(
        paths["setting"]
    ).items():
        command.extend([f"--{key}", str(value)])

    print("Running official visualization:")
    print(subprocess.list2cmdline(command))
    print()
    # Chạy trong thư mục official để sử dụng đúng cấu trúc dự án
    subprocess.run(command, cwd=OFFICIAL_DIR, check=True)

    if not official_output.is_dir():
        raise FileNotFoundError(
            f"Official visualization output not found:\n{official_output}"
        )

# kq
def collect_outputs(paths):
    output_dir = paths["output_dir"]
    official_output = paths["official_output"]

    # Chỉ xóa thư mục kết quả của dataset và pred_len đang chạy
    if output_dir.exists():
        shutil.rmtree(output_dir)

    copied_files = 0
    #sao chép
    for source in official_output.rglob("*"):
        if not source.is_file():
            continue

        relative_path = source.relative_to(official_output)
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_files += 1
        
    if copied_files == 0:
        raise RuntimeError("Official visualization generated no files.")

    print(f"Copied {copied_files} files into:\n{output_dir}")

# vẽ ma trận TVC Graph
def plot_graph_matrix(graph_matrix, sample_index, output_dir):
    graph_matrix = np.asarray(graph_matrix).squeeze()
    
    # Ma trận đồ thị phải là ma trận vuông
    if (
        graph_matrix.ndim != 2
        or graph_matrix.shape[0] != graph_matrix.shape[1]
    ):
        raise ValueError(
            f"Expected a square graph matrix, got {graph_matrix.shape}"
        )
    # Không vẽ nếu ma trận chứa NaN hoặc vô cực
    if not np.isfinite(graph_matrix).all():
        raise ValueError(
            f"Graph matrix {sample_index} contains NaN or infinity."
        )

    node_count = graph_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(10, 9), dpi=180)

    vmin = min(0.0, float(np.min(graph_matrix)))
    vmax = float(np.max(graph_matrix))
    if vmax == vmin:
        vmax = vmin + 1.0
        
    # Mỗi ô biểu diễn trọng số liên kết giữa một cặp nút
    image = ax.imshow(
        graph_matrix,
        cmap="Blues",
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

     # Thêm đường lưới phân tách các ô của ma trận
    ax.set_xticks(np.arange(-0.5, node_count, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, node_count, 1), minor=True)
    ax.grid(which="minor", color="#9AAFC2", linewidth=0.3, alpha=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    if node_count == 49:
        group_size = 7
        centers = [
            group * group_size + (group_size - 1) / 2
            for group in range(7)
        ]
        labels = [f"V{i}" for i in range(1, 8)]

        for boundary in range(group_size, node_count, group_size):
            ax.axvline(boundary - 0.5, color="#34495E", linewidth=1.1)
            ax.axhline(boundary - 0.5, color="#34495E", linewidth=1.1)

        ax.set_xticks(centers)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks(centers)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Target variable")
        ax.set_ylabel("Source variable")
    else:
        ax.set_xlabel("Target node")
        ax.set_ylabel("Source node")

    ax.set_title(f"TVCFormer Graph A: {sample_index}", fontsize=14, pad=12)
    # thanh màu of gtri trọg số cạh
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label(
        "Edge weight"
    )
    fig.tight_layout()

    image_path = output_dir / f"{sample_index}.png"
    fig.savefig(
        image_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Updated graph image: {image_path}")

# đọc ma trận và vẽ
def redraw_graph_images(paths):
    graph_dir = paths["graph_dir"]
    if not graph_dir.is_dir():
        raise FileNotFoundError(f"Graph directory not found:\n{graph_dir}")

    matrix_files = sorted(
        graph_dir.glob("*.npy"),
        key=lambda p: (not p.stem.isdigit(),
                       int(p.stem) if p.stem.isdigit() else p.stem),
    )

    if not matrix_files:
        raise FileNotFoundError(f"No graph NPY files found in:\n{graph_dir}")

    for matrix_file in matrix_files:
        plot_graph_matrix(
            np.load(matrix_file),
            matrix_file.stem,
            graph_dir,
        )


def main():
    args = parse_args()
    paths = build_paths(args)

    print("=" * 65)
    print(
        f"TVCFormer graph visualization - {args.dataset}, "
        f"pred_len={args.pred_len}"
    )
    print("=" * 65)
    print(f"Project    : {PROJECT_ROOT}")
    print(f"Dataset    : {paths['dataset_file']}")
    print(f"Config     : {paths['config_file']}")
    print(f"Checkpoint : {paths['checkpoint_dir'] / 'checkpoint.pth'}")
    print(f"Output     : {paths['output_dir']}")
    print()

    run_official_visualization(args, paths)
    collect_outputs(paths)
    redraw_graph_images(paths)

    print()
    print("=" * 65)
    print("GRAPH VISUALIZATION COMPLETED")
    print("=" * 65)
    print(f"Results saved in:\n{paths['output_dir']}")


if __name__ == "__main__":
    main()
