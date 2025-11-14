import pandas as pd
import numpy as np
import json
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data/data.csv")

# Encode categorical columns automatically
cat_cols = ["Sex","Ethnicity","Jaundice","Family_mem_with_ASD","Who_completed_the_test","ASD_traits"]
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and labels
X = df.drop(columns=["ASD_traits"]).values.astype(float)
y = df["ASD_traits"].values.astype(int)

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Keep only first 4 features (quantum circuits can't handle 17 directly)
# Using PCA is possible, but for LR-equivalent VQC, 4 dims is common.
X = X[:, :4]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -------------------------------
# Build Quantum Logistic Regression (VQC)
# -------------------------------
num_qubits = X_train.shape[1]
dev = qml.device("default.qubit", wires=num_qubits)

def feature_encoding(x):
    for i in range(num_qubits):
        qml.RX(x[i], wires=i)

def variational_layer(weights):
    for i in range(num_qubits):
        qml.Rot(weights[i,0], weights[i,1], weights[i,2], wires=i)
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i+1])

@qml.qnode(dev)
def circuit(x, weights):
    feature_encoding(x)
    variational_layer(weights)
    return qml.expval(qml.PauliZ(0))

# Loss = binary cross entropy-like
def loss(weights, X, y):
    predictions = [circuit(x, weights) for x in X]
    predictions = (1 + pnp.array(predictions)) / 2  # convert Z expval → [0,1]
    return pnp.mean((predictions - y) ** 2)  # MSE as surrogate LR loss

opt = qml.AdamOptimizer(stepsize=0.05)
weights = pnp.random.randn(num_qubits, 3, requires_grad=True)

epochs = 25

for epoch in range(epochs):
    weights = opt.step(lambda w: loss(w, X_train, y_train), weights)
    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss = {loss(weights, X_train, y_train):.4f}")

# -------------------------------
# Inference
# -------------------------------
def predict(X):
    preds = []
    for x in X:
        p = circuit(x, weights)
        p = (1 + p) / 2
        preds.append(1 if p > 0.5 else 0)
    return np.array(preds)

y_pred = predict(X_test)

# -------------------------------
# Metrics
# -------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1
}

print("\n=== Results (Quantum Logistic Regression) ===")
print(results)

# Save results to LR folder
with open("LR/results_lr.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved to LR/results_lr.json")
