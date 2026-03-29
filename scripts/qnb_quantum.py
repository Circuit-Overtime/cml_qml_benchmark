# ============================================================
# Quantum Naive Bayes (QNB) — Hybrid Quantum-Classical
# Uses quantum circuit as feature transformer + GaussianNB
# ============================================================

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import pennylane as qml
from pennylane import numpy as pnp

# ============================================================
# 1. Load and preprocess dataset (same as VQC/QSVM)
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

# PCA to 4 components (matching VQC/QSVM)
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)

X_train_pca = np.asarray(X_train_pca, dtype=float)
X_test_pca = np.asarray(X_test_pca, dtype=float)
y_train = np.asarray(y_train, dtype=float)
y_test = np.asarray(y_test, dtype=float)

# ============================================================
# 2. QNB Circuit — No trainable parameters
# ============================================================
n_qubits = 4

try:
    dev = qml.device("lightning.gpu", wires=n_qubits)
    print("Using PennyLane Lightning GPU")
except Exception:
    print("lightning.gpu not available, falling back to default.qubit")
    dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def qnb_circuit(x):
    x = pnp.array(x, dtype=float)

    # Stage 1: Angle Encoding — RY(pi * x_i)
    for i in range(n_qubits):
        qml.RY(np.pi * x[i], wires=i)

    # Stage 2: Entangling Layer — CNOT ring
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])

    # Stage 3: Phase Encoding — RZ(pi * x_i)
    for i in range(n_qubits):
        qml.RZ(np.pi * x[i], wires=i)

    # Stage 4: Measurement — PauliZ on ALL qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ============================================================
# 3. Quantum Feature Transformation
# ============================================================
print("\nTransforming training features through quantum circuit...")
X_train_q = np.array([qnb_circuit(x) for x in tqdm(X_train_pca)])

print("Transforming test features through quantum circuit...")
X_test_q = np.array([qnb_circuit(x) for x in tqdm(X_test_pca)])

# ============================================================
# 4. Classical GaussianNB on quantum-transformed features
# ============================================================
print("\nTraining GaussianNB on quantum features...")
gnb = GaussianNB()
gnb.fit(X_train_q, y_train)
y_pred = gnb.predict(X_test_q)

# ============================================================
# 5. Evaluation
# ============================================================
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n==== QNB (Quantum Naive Bayes) ====")
print(f"Acc : {acc:.4f}  Prec: {pre:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
