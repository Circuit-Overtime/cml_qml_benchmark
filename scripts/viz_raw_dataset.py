# ============================================================
# Visualization 1: Raw Dataset Feature Distribution
# Shows correlation heatmap of clinical features (A1-A9)
# colored by ASD class, plus class balance bar chart.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/data.csv")
df["ASD_traits"] = LabelEncoder().fit_transform(df["ASD_traits"])

clinical_cols = [f"A{i}" for i in range(1, 10)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})

# --- Left: Correlation heatmap of clinical features ---
corr = df[clinical_cols + ["ASD_traits"]].corr()
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
    square=True, linewidths=0.5, ax=axes[0],
    cbar_kws={"shrink": 0.8}
)
axes[0].set_title("Feature Correlation Matrix (A1–A9 + ASD Label)", fontsize=11)

# --- Right: Class distribution ---
class_counts = df["ASD_traits"].value_counts().sort_index()
bars = axes[1].bar(
    ["Non-ASD (0)", "ASD (1)"],
    class_counts.values,
    color=["#4C72B0", "#DD8452"],
    edgecolor="black", linewidth=0.8
)
for bar, count in zip(bars, class_counts.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 str(count), ha="center", fontsize=11, fontweight="bold")
axes[1].set_title("Class Distribution", fontsize=11)
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("paper/figures/viz_raw_dataset.png", dpi=300, bbox_inches="tight")
print("Saved → paper/figures/viz_raw_dataset.png")
plt.close()
