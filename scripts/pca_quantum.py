# ============================================================
# Quantum ML on PCA-Reduced Dataset (n_components=4)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------------------------------------------------------------------
# PENNYLANE - GPU VQC
# --------------------------------------------------------------------
import pennylane as qml
from pennylane import numpy as pnp

try:
    dev_vqc = qml.device("lightning.gpu", wires=4)
    print("⚡ Using PennyLane Lightning GPU")
except:
    print("⚠ lightning.gpu not available → falling back to default.qubit")
    dev_vqc = qml.device("default.qubit", wires=4)

# --------------------------------------------------------------------
# QISKIT (Updated API)
# --------------------------------------------------------------------
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit.primitives import BackendSampler
from qiskit_aer import AerSimulator


# ============================================================
# 1. Load and preprocess dataset
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

# PCA
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)

# ensure pure numpy
X_train_pca = np.asarray(X_train_pca, dtype=float)
X_test_pca  = np.asarray(X_test_pca, dtype=float)

y_train = np.asarray(y_train, dtype=float)
y_test  = np.asarray(y_test, dtype=float)


# ============================================================
# Helper
# ============================================================
def evaluate(name, y_true, y_pred, table):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)

    print(f"\n==== {name} ====")
    print(f"Acc : {acc:.4f}  Prec: {pre:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")

    table.append([name, acc, pre, rec, f1])
    return table


results = []

# ============================================================
# 2. VQC – GPU (with tqdm)
# ============================================================
n_qubits = 4

@qml.qnode(dev_vqc)
def qcircuit(weights, x):
    x = pnp.array(x, dtype=float)
    for i in range(n_qubits):
        qml.RY(np.pi * x[i], wires=i)

    for i in range(n_qubits):
        qml.Rot(weights[i,0], weights[i,1], weights[i,2], wires=i)

    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

    return qml.expval(qml.PauliZ(0))


weights = pnp.random.uniform(0, np.pi, (n_qubits, 3), requires_grad=True)
opt = qml.AdamOptimizer(0.1)

loss_history = []

def loss_fn(w):
    preds = pnp.array([qcircuit(w, xi) for xi in X_train_pca])
    preds = (preds + 1) / 2
    return pnp.mean((preds - y_train) ** 2)

EPOCHS = 10

print("\n🚀 Training VQC...")
for epoch in tqdm(range(EPOCHS)):
    weights = opt.step(loss_fn, weights)
    loss = loss_fn(weights)
    loss_history.append(float(loss))


# Predictions
vqc_preds = [(qcircuit(weights, x) + 1)/2 > 0.5 for x in X_test_pca]
vqc_preds = np.array(vqc_preds).astype(int)

results = evaluate("VQC (GPU)", y_test, vqc_preds, results)


# Save VQC loss curve
plt.figure(figsize=(7,5))
plt.plot(loss_history, linewidth=2)
plt.title("VQC Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("vqc_loss_curve.png", dpi=200)
plt.close()

print("📁 Saved → vqc_loss_curve.png")


# =========================
# Robust QSVM via manual kernel (statevector fidelity)
# =========================
from qiskit import Aer, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from sklearn.svm import SVC
import numpy as np
import time
import os

print("\n🚀 Running robust QSVM (manual statevector kernel)...")

backend = Aer.get_backend("aer_simulator_statevector")  # Uses AerSimulator statevector
# if using qiskit_aer.AerSimulator:
# from qiskit_aer import AerSimulator
# backend = AerSimulator(method="statevector")

feature_map = ZZFeatureMap(feature_dimension=4, reps=2)  # keep reps small for speed

# function to produce statevector for a single sample (feature array of length 4)
def get_statevector(feature_array):
    qc = feature_map.bind_parameters(feature_array) if hasattr(feature_map, "bind_parameters") else feature_map
    # In the common case feature_map is a ParameterizedCircuit; we need to assign parameters to produce a concrete circuit.
    # Build a fresh circuit with parameters set:
    xv = feature_array
    qc_copy = QuantumCircuit(feature_map.num_qubits)
    # Instead of trying to programmatically bind, build a numeric version:
    # The simplest: evaluate feature_map with parameters substituted (if it's ParameterizedCircuit)
    try:
        param_dict = {p: float(val) for p, val in zip(feature_map.parameters, xv)}
        qc_num = feature_map.bind_parameters(param_dict)
    except Exception:
        # fallback: make a fresh feature_map and assign parameter values via interpolate (works for common maps)
        qc_num = feature_map.bind_parameters(xv) if hasattr(feature_map, "bind_parameters") else feature_map

    # Transpile for simulator
    tqc = transpile(qc_num, backend=backend)
    result = backend.run(tqc).result()
    sv = result.get_statevector(tqc)
    # convert to numpy complex vector
    return np.asarray(sv)

# compute statevectors for train+test sets
X_all = np.vstack([X_train_pca, X_test_pca])
n_train = X_train_pca.shape[0]
n_all = X_all.shape[0]

# Optional: cache statevectors to disk to avoid recomputing
sv_cache = "quantum_reduced/statevectors.npy"
if os.path.exists(sv_cache):
    statevectors = np.load(sv_cache, allow_pickle=False)
else:
    statevectors = []
    t0 = time.time()
    for i, x in enumerate(X_all):
        sv = get_statevector(x)
        statevectors.append(sv)
    statevectors = np.vstack([sv.reshape(1, -1) for sv in statevectors])  # shape (n_all, dim)
    np.save(sv_cache, statevectors)
    print("Computed and cached statevectors in {:.1f}s".format(time.time() - t0))

# build fidelity kernel matrix K_ij = |<psi_i | psi_j>|^2
dim = statevectors.shape[1]
K = np.zeros((n_all, n_all), dtype=float)
for i in range(n_all):
    vi = statevectors[i].view(np.complex128)
    for j in range(i, n_all):
        vj = statevectors[j].view(np.complex128)
        fid = np.abs(np.vdot(vi, vj))**2
        K[i, j] = fid
        K[j, i] = fid

# split into train/train and test/train kernels for sklearn SVC
K_train = K[:n_train, :n_train]
K_test = K[n_train:, :n_train]

# train SVM with precomputed kernel
svc = SVC(kernel="precomputed")
svc.fit(K_train, y_train)

y_pred_qsvc = svc.predict(K_test)

# evaluate
from sklearn.metrics import classification_report
print("\nQSVM (manual kernel) metrics:")
print(classification_report(y_test, y_pred_qsvc, digits=4))

# save kernel matrices for inspection
np.save("quantum_reduced/K_train.npy", K_train)
np.save("quantum_reduced/K_test.npy", K_test)

