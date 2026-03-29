# ============================================================
# Variational Quantum Classifier (VQC) — Standalone
# Same preprocessing as QNB/QSVM for consistent evaluation
# Outputs: Accuracy, Precision, Recall, F1, Confusion Matrix
# ============================================================

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import pennylane as qml
from pennylane import numpy as pnp

# ============================================================
# 1. Load and preprocess dataset (identical to QNB/QSVM)
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
# 2. VQC Circuit
# ============================================================
n_qubits = 4

try:
    dev = qml.device("lightning.gpu", wires=n_qubits)
    print("Using PennyLane Lightning GPU (CUDA)")
except Exception:
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        print("Using PennyLane Lightning Qubit (C++ optimized)")
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
        print("Using default.qubit (slowest)")


@qml.qnode(dev)
def qcircuit(weights, x):
    x = pnp.array(x, dtype=float)

    # Stage 1: Angle Encoding — RY(pi * x_i)
    for i in range(n_qubits):
        qml.RY(np.pi * x[i], wires=i)

    # Stage 2: Variational Ansatz — Rot + CNOT ladder
    for i in range(n_qubits):
        qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)

    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    # Stage 3: Measurement — PauliZ on qubit 0
    return qml.expval(qml.PauliZ(0))


# ============================================================
# 3. Training
# ============================================================
weights = pnp.random.uniform(0, np.pi, (n_qubits, 3), requires_grad=True)
opt = qml.AdamOptimizer(0.1)

loss_history = []


def loss_fn(w):
    preds = pnp.array([qcircuit(w, xi) for xi in X_train_pca])
    preds = (preds + 1) / 2
    return pnp.mean((preds - y_train) ** 2)


EPOCHS = 50

print(f"\nTraining VQC ({EPOCHS} epochs)...")
for epoch in tqdm(range(EPOCHS)):
    weights = opt.step(loss_fn, weights)
    loss = loss_fn(weights)
    loss_history.append(float(loss))
    print(f"  Epoch {epoch+1}/{EPOCHS}  Loss: {loss:.4f}")

# ============================================================
# 4. Evaluation
# ============================================================
print("\nEvaluating on test set...")
vqc_preds = np.array([(qcircuit(weights, x) + 1) / 2 > 0.5 for x in tqdm(X_test_pca)]).astype(int)

acc = accuracy_score(y_test, vqc_preds)
pre = precision_score(y_test, vqc_preds)
rec = recall_score(y_test, vqc_preds)
f1 = f1_score(y_test, vqc_preds)

print(f"\n==== VQC (Variational Quantum Classifier) ====")
print(f"Acc : {acc:.4f}  Prec: {pre:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, vqc_preds, digits=4))

cm = confusion_matrix(y_test, vqc_preds)
print(f"Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print(f"  Total={cm.sum()} (should be 749)")
