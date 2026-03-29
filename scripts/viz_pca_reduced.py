# ============================================================
# Visualization 2: PCA-Reduced Feature Space
# Shows scatter plots of PCA components colored by ASD class
# and explained variance bar chart.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA

df = pd.read_csv("data/data.csv")
target_col = "ASD_traits"

categorical_cols = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "Who_completed_the_test"]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
df[target_col] = LabelEncoder().fit_transform(df[target_col])

X = df.drop(columns=[target_col])
y = df[target_col].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

colors = ["#4C72B0" if label == 0 else "#DD8452" for label in y]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Left: PC1 vs PC2 scatter ---
axes[0].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c="#4C72B0", alpha=0.4, s=10, label="Non-ASD")
axes[0].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c="#DD8452", alpha=0.4, s=10, label="ASD")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].set_title("PCA: PC1 vs PC2")
axes[0].legend(fontsize=9)

# --- Middle: PC3 vs PC4 scatter ---
axes[1].scatter(X_pca[y == 0, 2], X_pca[y == 0, 3], c="#4C72B0", alpha=0.4, s=10, label="Non-ASD")
axes[1].scatter(X_pca[y == 1, 2], X_pca[y == 1, 3], c="#DD8452", alpha=0.4, s=10, label="ASD")
axes[1].set_xlabel("PC3")
axes[1].set_ylabel("PC4")
axes[1].set_title("PCA: PC3 vs PC4")
axes[1].legend(fontsize=9)

# --- Right: Explained variance ---
var_ratio = pca.explained_variance_ratio_ * 100
cumulative = np.cumsum(var_ratio)
bars = axes[2].bar(
    ["PC1", "PC2", "PC3", "PC4"], var_ratio,
    color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    edgecolor="black", linewidth=0.8
)
axes[2].plot(["PC1", "PC2", "PC3", "PC4"], cumulative, "ko-", linewidth=2, label="Cumulative")
for bar, v in zip(bars, var_ratio):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
axes[2].set_ylabel("Explained Variance (%)")
axes[2].set_title("PCA Explained Variance")
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig("paper/figures/viz_pca_reduced.png", dpi=300, bbox_inches="tight")
print("Saved → paper/figures/viz_pca_reduced.png")
plt.close()
