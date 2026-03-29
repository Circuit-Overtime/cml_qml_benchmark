# ============================================================
# Quantum Support Vector Machine (QSVM) — Standalone
# Same preprocessing as QNB/VQC for consistent evaluation
# Outputs: Accuracy, Precision, Recall, F1, Confusion Matrix
# ============================================================

import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit import QuantumCircuit

# ============================================================
# 1. Load and preprocess dataset (identical to QNB/VQC)
# ============================================================
df = pd.read_csv("data/data.csv")
target_col = "ASD_traits"

categorical_cols = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "Who_completed_the_test"]

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

df[target_col] = LabelEncoder().fit_transform(df[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# PCA to 4 components
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)

X_train_pca = np.asarray(X_train_pca, dtype=float)
X_test_pca = np.asarray(X_test_pca, dtype=float)
y_train = np.asarray(y_train, dtype=float)
y_test = np.asarray(y_test, dtype=float)

# ============================================================
# 2. QSVM — ZZFeatureMap + Statevector Fidelity Kernel
# ============================================================
print("\nRunning QSVM (manual statevector fidelity kernel)...")

from qiskit import transpile
from qiskit_aer import AerSimulator

backend = AerSimulator(method="statevector")
print("Using qiskit_aer AerSimulator (statevector)")

feature_map = ZZFeatureMap(feature_dimension=4, reps=2)


def get_statevector(feature_array):
    """Encode a single sample into a quantum state via ZZFeatureMap."""
    param_dict = {p: float(val) for p, val in zip(feature_map.parameters, feature_array)}
    qc_num = feature_map.assign_parameters(param_dict)
    qc_num.save_statevector()
    tqc = transpile(qc_num, backend=backend)
    result = backend.run(tqc).result()
    sv = result.data()["statevector"]
    return np.asarray(sv)


# Compute statevectors for all train + test samples
X_all = np.vstack([X_train_pca, X_test_pca])
n_train = X_train_pca.shape[0]
n_all = X_all.shape[0]

print(f"Computing statevectors for {n_all} samples...")
statevectors = []
t0 = time.time()
for i, x in enumerate(X_all):
    sv = get_statevector(x)
    statevectors.append(sv)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{n_all} done...")
statevectors = np.vstack([sv.reshape(1, -1) for sv in statevectors])
print(f"Statevectors computed in {time.time() - t0:.1f}s")

# Build fidelity kernel matrix: K_ij = |<psi_i|psi_j>|^2
print("Building kernel matrix...")
K = np.zeros((n_all, n_all), dtype=float)
for i in range(n_all):
    vi = statevectors[i].view(np.complex128)
    for j in range(i, n_all):
        vj = statevectors[j].view(np.complex128)
        fid = np.abs(np.vdot(vi, vj)) ** 2
        K[i, j] = fid
        K[j, i] = fid

K_train = K[:n_train, :n_train]
K_test = K[n_train:, :n_train]

# ============================================================
# 3. Classical SVC with precomputed quantum kernel
# ============================================================
print("Training SVC with precomputed quantum kernel...")
svc = SVC(kernel="precomputed")
svc.fit(K_train, y_train)

y_pred = svc.predict(K_test)

# ============================================================
# 4. Evaluation
# ============================================================
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n==== QSVM (Quantum Support Vector Machine) ====")
print(f"Acc : {acc:.4f}  Prec: {pre:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print(f"  Total={cm.sum()} (should be 749)")
