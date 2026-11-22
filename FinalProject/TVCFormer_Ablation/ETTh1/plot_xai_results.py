
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Đường dẫn dữ liệu và thư mục lưu biểu đồ
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

csv_path = RESULTS / "ETTh1_variable_influence.csv"
df = pd.read_csv(csv_path)

# Thứ tự các biến đầu vào
features = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
horizons = [96, 192, 336, 720]

# Đảm bảo đúng thứ tự biến và horizon
df["pred_len"] = df["pred_len"].astype(int)
df["variable"] = pd.Categorical(
    df["variable"], categories=features, ordered=True
)
df = df.sort_values(["pred_len", "variable"])

# Chuyển dữ liệu thành bảng:
# Hàng là biến, cột là horizon
pivot = df.pivot(
    index="variable",
    columns="pred_len",
    values="influence_mse"
).reindex(features)[horizons]

# BIỂU ĐỒ 1: So sánh mức ảnh hưởng giữa các biến
ax = pivot.plot(
    kind="bar",
    figsize=(11, 6),
    width=0.8
)

ax.set_title("Feature Ablation on ETTh1 - MSE Increase")
ax.set_xlabel("Input Variables")
ax.set_ylabel("Increase in MSE (Delta MSE)")
ax.legend(title="Prediction Horizon")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    RESULTS / "ETTh1_XAI_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# BIỂU ĐỒ 2: Mức ảnh hưởng thay đổi theo horizon
plt.figure(figsize=(11, 6))

for feature in features:
    plt.plot(
        horizons,
        pivot.loc[feature].values,
        marker="o",
        label=feature
    )

plt.title("Feature Influence across Prediction Horizons")
plt.xlabel("Prediction Horizon")
plt.ylabel("Increase in MSE (Delta MSE)")
plt.xticks(horizons)
plt.grid(linestyle="--", alpha=0.4)
plt.legend(title="Input Variables")
plt.tight_layout()

plt.savefig(
    RESULTS / "ETTh1_XAI_horizon_analysis.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("Da tao xong hai bieu do XAI!")
