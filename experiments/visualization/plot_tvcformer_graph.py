
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = PROJECT_ROOT / "official"

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
    / "ETTh1_pred96"
)

GRAPH_DIR = OUTPUT_DIR / "graphs"

MODEL_ID = "TVCFormer_ETTh1_336_96"

SETTING = (
    "TVCFormer_ETTh1_336_96"
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


# ============================================================
# 2. Check required files
# ============================================================

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
    print("TVCFormer graph visualization - ETTh1, pred_len=96")
    print("=" * 65)
    print(f"Dataset    : {DATASET_FILE}")
    print(f"Config     : {CONFIG_FILE}")
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Graph dir  : {GRAPH_DIR}")
    print()


# ============================================================
# 3. Run official visualization
# ============================================================

def run_official_visualization():
    # Remove only the old visualization output for this setting.
    # Do not remove checkpoints or source code.
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
        "3",
        "--model_id",
        MODEL_ID,
        "--model",
        "TVCFormer",
        "--data",
        "ETTh1",
        "--root_path",
        str(DATASET_DIR),
        "--data_path",
        DATASET_FILE.name,
        "--seq_len",
        "336",
        "--pred_len",
        "96",
    ]

    print("Running official visualization...")
    print(subprocess.list2cmdline(command))
    print()

    subprocess.run(
        command,
        cwd=OFFICIAL_DIR,
        check=True,
    )

    if not OFFICIAL_OUTPUT.is_dir():
        raise FileNotFoundError(
            "Official visualization output not found:\n"
            f"{OFFICIAL_OUTPUT}"
        )


# ============================================================
# 4. Copy author outputs into experiments/
# ============================================================

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

    print(f"Copied {copied_files} visualization files.")


# ============================================================
# 5. Redraw graph images and overwrite existing PNG files
# ============================================================

def plot_graph_matrix(graph_matrix, sample_index, output_dir):
    """
    Draw a graph matrix with visible cell boundaries and
    stronger separators between the 7 variable groups.
    Save directly to graphs/<sample_index>.png.
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

    fig, ax = plt.subplots(
        figsize=(10, 9),
        dpi=180,
    )

    vmax = float(np.nanmax(graph_matrix))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    image = ax.imshow(
        graph_matrix,
        cmap="Blues",
        interpolation="nearest",
        aspect="equal",
        vmin=0,
        vmax=vmax,
    )

    # Fine grid: one cell per graph node.
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
        color="#8FA3B8",
        linewidth=0.3,
        alpha=0.9,
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # ETTh1: 7 variables, each represented by 7 time nodes.
    if node_count == 49:
        time_node_num = 7
        variable_count = 7

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
        f"Graph A: {sample_index}",
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

    # Same filename as the existing graph image: overwrite it.
    image_path = output_dir / f"{sample_index}.png"

    fig.savefig(
        image_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"Updated image: {image_path}")


def redraw_graph_images():
    if not GRAPH_DIR.is_dir():
        raise FileNotFoundError(
            f"Graph directory not found:\n{GRAPH_DIR}"
        )

    matrix_files = list(GRAPH_DIR.glob("*.npy"))

    def sort_key(path):
        return (
            0, int(path.stem)
        ) if path.stem.isdigit() else (1, path.stem)

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


# ============================================================
# 6. Main
# ============================================================

def main():
    check_paths()
    run_official_visualization()
    collect_outputs()
    redraw_graph_images()

    print()
    print("=" * 65)
    print("COMPLETED")
    print("=" * 65)
    print("Graph PNG images have been overwritten in:")
    print(GRAPH_DIR)


if __name__ == "__main__":
    main()
