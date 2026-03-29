# ============================================================
# Visualization 3: Quantum Encoding Pipeline
# Shows: PCA features → angle-encoded values → Bloch sphere
# coordinates, illustrating the classical-to-quantum mapping.
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

# ---- Angle encoding: RY(pi * x_i) maps x_i to Bloch sphere ----
# After RY(theta), qubit state: cos(theta/2)|0> + sin(theta/2)|1>
# With theta = pi * x_i:
#   P(|1>) = sin^2(pi * x_i / 2)
#   P(|0>) = cos^2(pi * x_i / 2)

theta = np.pi * X_pca  # shape: (n_samples, 4)
prob_1 = np.sin(theta / 2) ** 2  # probability of measuring |1>

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

pc_labels = ["PC1", "PC2", "PC3", "PC4"]

for idx, ax in enumerate(axes.flat):
    # PCA values (x-axis) vs encoded |1> probability (y-axis)
    ax.scatter(
        X_pca[y == 0, idx], prob_1[y == 0, idx],
        c="#4C72B0", alpha=0.3, s=8, label="Non-ASD"
    )
    ax.scatter(
        X_pca[y == 1, idx], prob_1[y == 1, idx],
        c="#DD8452", alpha=0.3, s=8, label="ASD"
    )

    # Reference curve: the ideal sin^2(pi*x/2) mapping
    x_ref = np.linspace(X_pca[:, idx].min(), X_pca[:, idx].max(), 200)
    y_ref = np.sin(np.pi * x_ref / 2) ** 2
    ax.plot(x_ref, y_ref, "k--", linewidth=1.5, alpha=0.6, label=r"$\sin^2(\pi x / 2)$")

    ax.set_xlabel(f"{pc_labels[idx]} (PCA value)")
    ax.set_ylabel(r"$P(|1\rangle)$ after $R_Y(\pi \cdot x_i)$")
    ax.set_title(f"Qubit {idx}: {pc_labels[idx]} → Angle Encoding")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

plt.suptitle(
    r"Quantum Angle Encoding: PCA Features $\rightarrow$ $R_Y(\pi \cdot x_i)$ $\rightarrow$ Qubit $|1\rangle$ Probability",
    fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("paper/figures/viz_quantum_encoding.png", dpi=300, bbox_inches="tight")
print("Saved → paper/figures/viz_quantum_encoding.png")
plt.close()
