
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = PROJECT_ROOT / "official"

PRED_LEN = 720
SEQ_LEN = 336
DATASET_NAME = "ETTh1"
SAMPLE_COUNT = 3

CONFIG_FILE = OFFICIAL_DIR / "configs" / "ETTh1.yaml"

DATASET_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "datasets"
    / "ETT-small"
)
DATASET_FILE = DATASET_DIR / "ETTh1.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "figures"
    / "graph"
    / f"{DATASET_NAME}_pred{PRED_LEN}"
)

GRAPH_DIR = OUTPUT_DIR / "graphs"

MODEL_ID = f"TVCFormer_{DATASET_NAME}_{SEQ_LEN}_{PRED_LEN}"

SETTING = (
    f"TVCFormer_{DATASET_NAME}_{SEQ_LEN}_{PRED_LEN}"
    "_dm32_nh8_el2_dl1_df128"
    "_gch4_gck3_tn7_kr0.7_0"
)

CHECKPOINT = (
    OFFICIAL_DIR
    / "checkpoints"
    / "Ett_series"
    / "ETTh1"
    / SETTING
    / "checkpoint.pth"
)

OFFICIAL_OUTPUT = (
    OFFICIAL_DIR
    / "visualize_results"
    / SETTING
    / "visual_data"
)

def check_paths():
    required_files = [
        OFFICIAL_DIR / "run_TVCFormer.py",
        CONFIG_FILE,
        DATASET_FILE,
        CHECKPOINT,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    print("=" * 65)
    print(
        f"TVCFormer graph visualization - "
        f"{DATASET_NAME}, pred_len={PRED_LEN}"
    )
    print("=" * 65)
    print(f"Project    : {PROJECT_ROOT}")
    print(f"Dataset    : {DATASET_FILE}")
    print(f"Config     : {CONFIG_FILE}")
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Output     : {OUTPUT_DIR}")
    print()


def run_official_visualization():
  
    if OFFICIAL_OUTPUT.exists():
        shutil.rmtree(OFFICIAL_OUTPUT)

    command = [
        sys.executable,
        "run_TVCFormer.py",
        "--config_yaml",
        str(CONFIG_FILE),
        "--is_training",
        "0",
        "--visualize",
        "True",
        "--visual_samples_num",
        str(SAMPLE_COUNT),
        "--model_id",
        MODEL_ID,
        "--model",
        "TVCFormer",
        "--data",
        DATASET_NAME,
        "--root_path",
        str(DATASET_DIR),
        "--data_path",
        DATASET_FILE.name,
        "--seq_len",
        str(SEQ_LEN),
        "--pred_len",
        str(PRED_LEN),
    ]

    print("Running official visualization:")
    print(subprocess.list2cmdline(command))
    print()

    subprocess.run(
        command,
        cwd=OFFICIAL_DIR,
        check=True,
    )

    if not OFFICIAL_OUTPUT.is_dir():
        raise FileNotFoundError(
            "Official visualization output was not created:\n"
            f"{OFFICIAL_OUTPUT}"
        )



def collect_outputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copied_files = 0

    for source in OFFICIAL_OUTPUT.rglob("*"):
        if not source.is_file():
            continue

        relative_path = source.relative_to(OFFICIAL_OUTPUT)
        destination = OUTPUT_DIR / relative_path

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(source, destination)
        copied_files += 1

    if copied_files == 0:
        raise RuntimeError(
            "No visualization files were generated."
        )

    print(f"Copied {copied_files} files into:")
    print(OUTPUT_DIR)


def plot_graph_matrix(graph_matrix, sample_index, output_dir):
    """
    Redraw a saved graph matrix.

    PNG files are saved directly into graphs/ and overwrite
    the corresponding existing PNG files. The graph matrix
    NPY files are retained unchanged.
    """
    graph_matrix = np.asarray(graph_matrix).squeeze()

    if (
        graph_matrix.ndim != 2
        or graph_matrix.shape[0] != graph_matrix.shape[1]
    ):
        raise ValueError(
            f"Expected a square graph matrix, "
            f"got shape {graph_matrix.shape}"
        )

    node_count = graph_matrix.shape[0]

    if not np.isfinite(graph_matrix).all():
        raise ValueError(
            f"Graph matrix {sample_index} contains NaN or infinity."
        )

    fig, ax = plt.subplots(
        figsize=(10, 9),
        dpi=180,
    )

    vmax = float(np.max(graph_matrix))
    vmin = min(0.0, float(np.min(graph_matrix)))

    if vmax == vmin:
        vmax = vmin + 1.0

    image = ax.imshow(
        graph_matrix,
        cmap="Blues",
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    # Draw thin lines between individual nodes.
    ax.set_xticks(
        np.arange(-0.5, node_count, 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, node_count, 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="#9AAFC2",
        linewidth=0.3,
        alpha=0.9,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    if node_count == 49:
        time_node_num = 7
        variable_count = 7

        # Strong separators between variable groups.
        for boundary in range(
            time_node_num,
            node_count,
            time_node_num,
        ):
            ax.axvline(
                boundary - 0.5,
                color="#34495E",
                linewidth=1.1,
            )
            ax.axhline(
                boundary - 0.5,
                color="#34495E",
                linewidth=1.1,
            )

        centers = [
            group * time_node_num
            + (time_node_num - 1) / 2
            for group in range(variable_count)
        ]

        labels = [
            f"V{i}"
            for i in range(1, variable_count + 1)
        ]

        ax.set_xticks(centers)
        ax.set_xticklabels(labels, fontsize=10)

        ax.set_yticks(centers)
        ax.set_yticklabels(labels, fontsize=10)

        ax.set_xlabel(
            "Target variable",
            fontsize=11,
            labelpad=10,
        )
        ax.set_ylabel(
            "Source variable",
            fontsize=11,
            labelpad=10,
        )
    else:
        ax.set_xlabel("Target node")
        ax.set_ylabel("Source node")

    ax.set_title(
        f"TVCFormer Graph A: {sample_index}",
        fontsize=14,
        pad=12,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("Edge weight", fontsize=10)

    fig.tight_layout()

    # Save directly to graphs/<sample_index>.png.
    image_path = output_dir / f"{sample_index}.png"

    fig.savefig(
        image_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Updated graph image: {image_path}")


def redraw_graph_images():
    if not GRAPH_DIR.is_dir():
        raise FileNotFoundError(
            f"Graph directory not found:\n{GRAPH_DIR}"
        )

    matrix_files = list(GRAPH_DIR.glob("*.npy"))

    def sort_key(path):
        if path.stem.isdigit():
            return (0, int(path.stem))
        return (1, path.stem)

    matrix_files.sort(key=sort_key)

    if not matrix_files:
        raise FileNotFoundError(
            f"No graph NPY files found in:\n{GRAPH_DIR}"
        )

    for matrix_file in matrix_files:
        graph_matrix = np.load(matrix_file)

        plot_graph_matrix(
            graph_matrix,
            matrix_file.stem,
            GRAPH_DIR,
        )


def main():
    check_paths()
    run_official_visualization()
    collect_outputs()
    redraw_graph_images()

    print()
    print("=" * 65)
    print("GRAPH VISUALIZATION COMPLETED")
    print("=" * 65)
    print(f"Results saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
