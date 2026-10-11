
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Thư mục chứa dữ liệu CSV
DATASETS_ROOT = PROJECT_ROOT / "experiments" / "datasets"

# Thư mục lưu các biểu đồ
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "figures"

# Tên bộ dữ liệu và tên tệp CSV tương ứng
DATASETS = {
    "ETTh1": "ETTh1.csv",
    "ETTh2": "ETTh2.csv",
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
    "Electricity": "electricity.csv",
    "Weather": "weather.csv",
}


def find_dataset(filename):
    """Tìm tệp CSV trong thư mục dữ liệu của dự án."""
    matches = sorted(
        path
        for path in DATASETS_ROOT.rglob("*")
        if path.is_file() and path.name.lower() == filename.lower()
    )

    if not matches:
        return None

    if len(matches) > 1:
        print(f"Cảnh báo: Tìm thấy nhiều tệp tên {filename}:")
        for path in matches:
            print(f"  - {path}")
        print("Hãy kiểm tra để bảo đảm đang dùng đúng dữ liệu.")

    return matches[0]


def load_dataset(dataset_name, dataset_file):
    """Đọc dữ liệu và lấy các cột có kiểu số."""
    print(f"\nĐang đọc bộ dữ liệu: {dataset_name}")
    print(f"Tệp dữ liệu: {dataset_file}")

    df = pd.read_csv(dataset_file)

    if df.empty:
        raise ValueError("Tệp CSV không có dữ liệu.")

    numeric_df = df.select_dtypes(include="number")
    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.empty:
        raise ValueError("Không tìm thấy cột số hợp lệ.")

    print(f"Số dòng dữ liệu: {len(df):,}")
    print(f"Số biến dạng số: {len(numeric_df.columns)}")
    print(f"Các biến: {list(numeric_df.columns)}")

    return numeric_df


def plot_correlation(numeric_df, dataset_name, output_dir):
    """Vẽ ma trận tương quan Pearson giữa các biến."""
    correlation = numeric_df.corr(method="pearson")

    # Giới hạn kích thước hình để tránh tạo ảnh quá lớn
    figure_size = max(10, min(24, len(correlation.columns) * 0.35))

    plt.figure(figsize=(figure_size, figure_size))

    # Chỉ hiện hệ số trên từng ô khi số biến không quá nhiều
    show_values = len(correlation.columns) <= 15

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=show_values,
        fmt=".2f",
        square=True,
    )

    plt.title(f"{dataset_name} - Ma trận tương quan Pearson")
    plt.tight_layout()

    output_file = output_dir / "correlation_matrix.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Đã lưu biểu đồ: {output_file}")


def plot_distribution(numeric_df, dataset_name, output_dir):
    """Vẽ biểu đồ phân phối giá trị cho các biến."""
    columns = numeric_df.columns.tolist()

    # Chia thành nhiều trang nếu bộ dữ liệu có nhiều biến
    variables_per_page = 12

    for page_start in range(0, len(columns), variables_per_page):
        page_columns = columns[
            page_start:page_start + variables_per_page
        ]

        fig, axes = plt.subplots(
            len(page_columns),
            1,
            figsize=(11, 2.5 * len(page_columns)),
            squeeze=False,
        )

        for ax, column in zip(axes.flatten(), page_columns):
            values = numeric_df[column].dropna()

            ax.hist(values, bins=50, edgecolor="black")
            ax.set_title(f"Phân phối của biến {column}")
            ax.set_xlabel("Giá trị")
            ax.set_ylabel("Tần suất")
            ax.grid(True, alpha=0.2)

        page_number = page_start // variables_per_page + 1
        fig.suptitle(
            f"{dataset_name} - Phân phối dữ liệu "
            f"(Trang {page_number})"
        )
        fig.tight_layout()

        if len(columns) <= variables_per_page:
            filename = "distribution.png"
        else:
            filename = f"distribution_page_{page_number}.png"

        output_file = output_dir / filename
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Đã lưu biểu đồ: {output_file}")


def plot_time_series(numeric_df, dataset_name, output_dir):
    """Vẽ diễn biến theo thời gian của tối đa bốn biến đầu tiên."""
    columns = numeric_df.columns[:4]

    # Chỉ lấy 2.000 dòng đầu để biểu đồ dễ quan sát
    data = numeric_df.loc[:, columns].iloc[:2000]

    fig, axes = plt.subplots(
        len(columns),
        1,
        figsize=(14, 3 * len(columns)),
        sharex=True,
        squeeze=False,
    )

    for ax, column in zip(axes.flatten(), columns):
        ax.plot(data.index, data[column], linewidth=0.8)
        ax.set_title(f"Diễn biến của biến {column}")
        ax.set_ylabel("Giá trị")
        ax.grid(True, alpha=0.3)

    axes.flatten()[-1].set_xlabel("Thứ tự thời điểm")
    fig.suptitle(f"{dataset_name} - Biểu đồ chuỗi thời gian")
    fig.tight_layout()

    output_file = output_dir / "time_series.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {output_file}")


def process_dataset(dataset_name, filename):
    """Xử lý và vẽ biểu đồ cho một bộ dữ liệu."""
    dataset_file = find_dataset(filename)

    if dataset_file is None:
        print(
            f"\nBỎ QUA {dataset_name}: "
            f"Không tìm thấy tệp {filename}."
        )
        return False

    output_dir = OUTPUT_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_df = load_dataset(dataset_name, dataset_file)

    plot_correlation(numeric_df, dataset_name, output_dir)
    plot_distribution(numeric_df, dataset_name, output_dir)
    plot_time_series(numeric_df, dataset_name, output_dir)

    print(f"Hoàn thành bộ dữ liệu: {dataset_name}")
    return True


def main():
    """Chạy khám phá dữ liệu cho tất cả các bộ đã khai báo."""
    if not DATASETS_ROOT.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {DATASETS_ROOT}"
        )

    print("=" * 65)
    print("KHÁM PHÁ DỮ LIỆU TVCFORMER")
    print("=" * 65)

    completed = []
    failed = []

    for dataset_name, filename in DATASETS.items():
        try:
            success = process_dataset(dataset_name, filename)

            if success:
                completed.append(dataset_name)
            else:
                failed.append(dataset_name)

        except Exception as error:
            print(f"\nLỗi khi xử lý {dataset_name}: {error}")
            failed.append(dataset_name)

    print("\n" + "=" * 65)
    print("TỔNG KẾT")
    print("=" * 65)
    print(f"Các bộ hoàn thành: {completed}")
    print(f"Các bộ bị bỏ qua hoặc lỗi: {failed}")
    print(f"Thư mục lưu biểu đồ: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
